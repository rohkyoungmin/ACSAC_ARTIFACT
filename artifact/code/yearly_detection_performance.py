#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yearly Dataset Experiment — Realistic Class Imbalance (9:1 GW:MW)
비교: Static vs FullUpdate vs Proposed  (말웨어 비율 ~10%)

Motivation (Arp et al., USENIX Security 2022):
  실제 배포 환경의 malware base rate는 ~5-10% (P1/P8 pitfall).
  exp_07의 50:50 균형 데이터셋은 낙관적 성능을 보고할 수 있으므로
  GW:MW = 9:1 (malware ≈ 10%)로 재샘플링한 동일 실험을 추가 수행.

실행: python yearly_detection_performance.py
결과: results/performance/yearly/yearly_imbalanced_results.csv
      results/xai/yearly/xai_snapshots_yearly_imbalanced.csv
소요시간 예상: ~15-30분

참고: yearly validation split에서 선택된 파라미터를 사용
      (results/hyperparameters/yearly_validation_tuned_params.json).
"""
import sys
import json
import time
sys.path.insert(0, '.')
from common import *
from common import _fresh_model, _run_selective_update_core


# ── Static (yearly) ───────────────────────────────────────
def run_static_yearly(X_train, y_train, y2idx, test_years, rows, y, cfg):
    model = _fresh_model(cfg)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    results = []
    for yr in test_years:
        t0  = time.perf_counter()
        idx = y2idx.get(yr, [])
        if not idx:
            continue
        Xt = build_hashed_csr(rows, idx, cfg.hash_dim)
        yt = y[idx]
        m  = compute_metrics(yt, model.predict(Xt),
                             model.predict_proba(Xt)[:, 1])
        m.update({"month": str(yr), "method": "Static",
                  "updated": 0, "total": len(yt),
                  "time_sec": round(time.perf_counter() - t0, 4)})
        results.append(m)
    return results


# ── FullUpdate (yearly) ───────────────────────────────────
def run_full_update_yearly(X_train, y_train, y2idx, test_years, rows, y, cfg):
    model = _fresh_model(cfg)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    results = []
    for yr in test_years:
        t0  = time.perf_counter()
        idx = y2idx.get(yr, [])
        if not idx:
            continue
        Xt = build_hashed_csr(rows, idx, cfg.hash_dim)
        yt = y[idx]
        m  = compute_metrics(yt, model.predict(Xt),
                             model.predict_proba(Xt)[:, 1])
        m.update({"month": str(yr), "method": "FullUpdate",
                  "updated": len(yt), "total": len(yt),
                  "time_sec": round(time.perf_counter() - t0, 4)})
        results.append(m)
        model.partial_fit(Xt, yt)
    return results


# ── Proposed (yearly) ─────────────────────────────────────
def run_proposed_yearly(X_train, y_train, y2idx, test_years,
                        rows, y, embedder, xai, cfg, rng):
    y2idx_str  = {str(yr): idxs for yr, idxs in y2idx.items()}
    test_y_str = [str(yr) for yr in test_years]
    results, xai_snaps = _run_selective_update_core(
        X_train, y_train, y2idx_str, test_y_str,
        rows, y, embedder, xai, cfg, rng,
        method_name="Proposed",
        use_cluster_selection=True,
        use_drift_loss=True,
    )
    return results, xai_snaps


def main():
    cfg = Config()
    os.makedirs(f"{cfg.perf_dir}/yearly", exist_ok=True)
    os.makedirs(f"{cfg.xai_dir}/yearly", exist_ok=True)

    cfg, _ = load_validation_tuned_params(cfg, granularity="yearly")

    rng = np.random.default_rng(cfg.random_state)

    with ExperimentTimer("yearly_detection_performance", cfg.out_dir):
        print(f"\n{'='*60}")
        print(f"  Yearly Experiment — Realistic Imbalance (GW:MW = 9:1)")
        print(f"  Static vs FullUpdate vs Proposed")
        print(f"  (Arp et al. 2022 P1/P8: realistic malware base rate ~10%)")
        print(f"{'='*60}")

        print("\n[1] Loading AndroZoo-Year (imbalanced 9:1)...")
        rows, y, years = load_androzoo_year_imbalanced(cfg, gw_mw_ratio=9.0)

        print("\n[2] Yearly temporal split...")
        y2idx, train_idx, val_idx, test_years = make_yearly_split(years, cfg)

        print("\n[3] Feature hashing + embedder setup...")
        X_train = build_hashed_csr(rows, train_idx, cfg.hash_dim)
        y_train = y[train_idx]
        print(f"  X_train: {X_train.shape}  nnz={X_train.nnz:,}  "
              f"mal_rate={y_train.mean():.3f}")

        b2n         = build_feature_index(rows, train_idx, cfg.hash_dim)
        global_mean = np.asarray(X_train.mean(axis=0)).flatten()

        embedder = SemanticEmbedder(cfg.svd_components, cfg.random_state)
        embedder.fit(X_train)

        xai = ClusterXAI(global_mean, b2n, cfg.xai_top_k, cfg.xai_min_delta)
        xai.set_embedder(embedder)

        print(f"\n[4] Running experiments on test years: {test_years}")

        print("  -- Static --")
        st_res = run_static_yearly(
            X_train, y_train, y2idx, test_years, rows, y, cfg)

        print("  -- FullUpdate --")
        fu_res = run_full_update_yearly(
            X_train, y_train, y2idx, test_years, rows, y, cfg)

        print("  -- Proposed --")
        pr_res, xai_snaps = run_proposed_yearly(
            X_train, y_train, y2idx, test_years,
            rows, y, embedder, xai, cfg,
            rng=np.random.default_rng(cfg.random_state))

        # Performance results
        all_results = st_res + fu_res + pr_res
        out_path    = f"{cfg.perf_dir}/yearly/yearly_imbalanced_results.csv"
        df = to_result_df(all_results)
        df.to_csv(out_path, index=False)
        print(f"\n  Saved performance: {out_path}")
        print_results_table(
            df, title="Yearly (9:1 Imbalanced) — Static vs FullUpdate vs Proposed")

        # XAI snapshots
        if xai_snaps:
            xai_path = f"{cfg.xai_dir}/yearly/xai_snapshots_yearly_imbalanced.csv"
            pd.DataFrame(xai_snaps).to_csv(xai_path, index=False)
            print(f"  Saved XAI snapshots: {xai_path}")

        print(f"\n  Performance → {cfg.perf_dir}/yearly/")
        print(f"  XAI        → {cfg.xai_dir}/yearly/")


if __name__ == "__main__":
    main()
