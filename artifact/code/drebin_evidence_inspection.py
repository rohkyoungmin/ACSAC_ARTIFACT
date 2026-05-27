#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREBIN XAI Case Study
- features/drebin_gw_features.p / drebin_mw_features.p 로드
- 메인 데이터셋으로 학습한 cluster에 DREBIN gw/mw 샘플을 할당
- cluster별 gw/mw purity + top features 분석
- Proposed 모델의 ClusterXAI (C2) contribution 정성 검증

실행: python drebin_evidence_inspection.py
결과: results/xai/drebin/drebin_xai_results.csv
소요시간 예상: ~15-30분
"""
import sys
import json
import pickle

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, '.')
from common import *

DREBIN_GW_PATH = "./artifact/data/features/drebin_gw_features.p"
DREBIN_MW_PATH = "./artifact/data/features/drebin_mw_features.p"


def load_drebin_features(hash_dim: int):
    """
    drebin_gw/mw_features.p 로드.
    각 파일은 List[lil_matrix], shape (1, 236363).
    hash_dim(262144)에 zero-pad 후 CSR로 반환.
    gw=0, mw=1 레이블.
    """
    with open(DREBIN_GW_PATH, "rb") as f:
        gw_list = pickle.load(f)
    with open(DREBIN_MW_PATH, "rb") as f:
        mw_list = pickle.load(f)

    feat_dim = gw_list[0].shape[1]
    pad      = hash_dim - feat_dim

    def stack_and_pad(lst):
        X = sp.vstack([m.tocsr() for m in lst], format="csr")
        if pad > 0:
            X = sp.hstack(
                [X, sp.csr_matrix((X.shape[0], pad), dtype=X.dtype)],
                format="csr")
        return X

    X_gw = stack_and_pad(gw_list)
    X_mw = stack_and_pad(mw_list)
    X    = sp.vstack([X_gw, X_mw], format="csr")
    y    = np.array([0] * len(gw_list) + [1] * len(mw_list), dtype=np.int32)

    print(f"  [DREBIN] GW: {len(gw_list):,}  MW: {len(mw_list):,}  "
          f"Total: {len(y):,}  (feat_dim: {feat_dim:,} → padded: {hash_dim:,})")
    return X, y


def run_drebin_xai(cfg: Config) -> List[Dict]:
    """
    1. 메인 데이터셋(월별)으로 Proposed 모델 학습 (train split까지만)
    2. 테스트 기간 동안 cluster 업데이트
    3. DREBIN gw/mw 샘플을 학습된 cluster에 할당
    4. cluster별 gw/mw purity + top features 분석
    """

    print("\n[1] Loading main dataset and training...")
    rows, y, months = load_dataset(cfg)
    m2idx, ordered, train_m, val_m, test_m, train_idx, val_idx = \
        make_temporal_split(months, cfg)

    b2n     = build_feature_index(rows, train_idx, cfg.hash_dim)
    X_train = build_hashed_csr(rows, train_idx, cfg.hash_dim)
    y_train = y[train_idx]

    embedder = SemanticEmbedder(cfg.svd_components, cfg.random_state)
    embedder.fit(X_train)

    global_mean = np.asarray(X_train.mean(axis=0)).flatten()
    xai = ClusterXAI(global_mean, b2n, cfg.xai_top_k, cfg.xai_min_delta)
    xai.set_embedder(embedder)

    cluster_mgr = ClusterManager(cfg, xai)
    Z_train     = embedder.transform(X_train)
    cluster_mgr.fit_initial(Z_train, X_train, y_train)

    rng = np.random.default_rng(cfg.random_state)
    print("  Updating clusters through test period...")
    for t_idx, mk in enumerate(test_m):
        idx_m = m2idx.get(mk, [])
        if not idx_m:
            continue
        Xt = build_hashed_csr(rows, idx_m, cfg.hash_dim)
        Zt = embedder.transform(Xt)
        member_rows = [rows[i] for i in idx_m]
        cluster_mgr.assign_and_update(Zt, Xt, t_idx, member_rows=member_rows)

    print(f"  Clusters after full training: {len(cluster_mgr.centroids)}")

    print("\n[2] Loading DREBIN gw/mw features...")
    X_drebin, y_drebin = load_drebin_features(cfg.hash_dim)

    Z_drebin    = embedder.transform(X_drebin)
    C           = np.stack(cluster_mgr.centroids)
    sims        = Z_drebin @ C.T
    assignments = sims.argmax(axis=1)

    print("\n[3] Analyzing cluster gw/mw purity...")
    results = []

    for cid in range(len(cluster_mgr.centroids)):
        mask    = assignments == cid
        n_total = int(mask.sum())
        if n_total < 3:
            continue

        labels_in = y_drebin[mask]
        n_mw      = int((labels_in == 1).sum())
        n_gw      = int((labels_in == 0).sum())
        dominant  = "malware" if n_mw >= n_gw else "goodware"
        purity    = max(n_mw, n_gw) / n_total

        top_feats = cluster_mgr.top_features[cid][:5]
        feat_str  = "; ".join(
            f"{nm}({'+' if d > 0 else ''}{d:.3f})"
            for nm, d in top_feats
        ) if top_feats else "N/A"

        ld    = cluster_mgr.label_dist[cid]
        total = ld.sum()
        mal_ratio = float(ld[1] / total) if total > 0 else 0.0

        results.append({
            "cluster_id"      : cid,
            "n_total"         : n_total,
            "n_malware"       : n_mw,
            "n_goodware"      : n_gw,
            "dominant_label"  : dominant,
            "purity"          : round(purity, 4),
            "n_train_samples" : int(total),
            "mal_ratio_train" : round(mal_ratio, 4),
            "top_features"    : feat_str,
            "cluster_age"     : cluster_mgr.age[cid],
        })

    return results


def main():
    cfg = Config()
    os.makedirs(f"{cfg.xai_dir}/drebin", exist_ok=True)

    cfg, _ = load_validation_tuned_params(cfg, granularity="monthly")

    with ExperimentTimer("drebin_evidence_inspection", cfg.out_dir):
        print(f"\n{'='*60}")
        print(f"  DREBIN XAI Case Study")
        print(f"  Validating ClusterXAI (C2) against DREBIN gw/mw samples")
        print(f"{'='*60}")

        results = run_drebin_xai(cfg)

        if not results:
            print("\n  [WARN] No results produced. Check DREBIN feature paths.")
            return

        out_path = f"{cfg.xai_dir}/drebin/drebin_xai_results.csv"
        df = pd.DataFrame(results)
        df.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")
        print_xai_table(df, title="DREBIN XAI — ClusterXAI gw/mw discrimination",
                        kind="marvin")


if __name__ == "__main__":
    main()
