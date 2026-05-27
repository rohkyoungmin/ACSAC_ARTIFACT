#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yearly 9:1 Imbalanced Ablation Study
Static / FullUpdate / Ablation-Baseline / Ablation-+Cluster /
Ablation-+DriftLoss / Proposed  (GW:MW = 9:1, ~10% malware)

실행: python yearly_ablation.py
결과: results/performance/yearly/yearly_imbalanced_ablation_results.csv
소요시간 예상: ~30-60분 (ablation 4종 × yearly)
"""
import sys
import json
import time
sys.path.insert(0, '.')
from common import *
from common import _fresh_model, _run_selective_update_core


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


def run_ablation_yearly(X_train, y_train, y2idx, test_years,
                        rows, y, embedder, xai, cfg, rng,
                        method_name, use_cluster_selection, use_drift_loss):
    y2idx_str  = {str(yr): idxs for yr, idxs in y2idx.items()}
    test_y_str = [str(yr) for yr in test_years]
    results, _ = _run_selective_update_core(
        X_train, y_train, y2idx_str, test_y_str,
        rows, y, embedder, xai, cfg, rng,
        method_name=method_name,
        use_cluster_selection=use_cluster_selection,
        use_drift_loss=use_drift_loss,
    )
    return results


def main():
    cfg = Config()
    os.makedirs(f"{cfg.perf_dir}/yearly", exist_ok=True)

    cfg, _ = load_validation_tuned_params(cfg, granularity="yearly")

    rng_seed = cfg.random_state

    with ExperimentTimer("yearly_ablation", cfg.out_dir):
        print(f"\n{'='*60}")
        print(f"  Yearly Imbalanced Ablation Study (GW:MW = 9:1)")
        print(f"  Static / FullUpdate / Baseline / +Cluster / +DriftLoss / Proposed")
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

        print(f"\n[4] Running ablation variants on test years: {test_years}")

        print("  -- Static --")
        st_res = run_static_yearly(X_train, y_train, y2idx, test_years, rows, y, cfg)

        print("  -- FullUpdate --")
        fu_res = run_full_update_yearly(X_train, y_train, y2idx, test_years, rows, y, cfg)

        print("  -- Ablation-Baseline (random sel, uniform loss) --")
        ab_res = run_ablation_yearly(
            X_train, y_train, y2idx, test_years, rows, y, embedder, xai, cfg,
            rng=np.random.default_rng(rng_seed),
            method_name="Ablation-Baseline",
            use_cluster_selection=False, use_drift_loss=False)

        print("  -- Ablation-+Cluster (cluster sel, uniform loss) --")
        ac_res = run_ablation_yearly(
            X_train, y_train, y2idx, test_years, rows, y, embedder, xai, cfg,
            rng=np.random.default_rng(rng_seed),
            method_name="Ablation-+Cluster",
            use_cluster_selection=True, use_drift_loss=False)

        print("  -- Ablation-+DriftLoss (random sel, drift-reg loss) --")
        ad_res = run_ablation_yearly(
            X_train, y_train, y2idx, test_years, rows, y, embedder, xai, cfg,
            rng=np.random.default_rng(rng_seed),
            method_name="Ablation-+DriftLoss",
            use_cluster_selection=False, use_drift_loss=True)

        print("  -- Proposed (all contributions) --")
        pr_res = run_ablation_yearly(
            X_train, y_train, y2idx, test_years, rows, y, embedder, xai, cfg,
            rng=np.random.default_rng(rng_seed),
            method_name="Proposed",
            use_cluster_selection=True, use_drift_loss=True)

        all_results = st_res + fu_res + ab_res + ac_res + ad_res + pr_res
        out_path    = f"{cfg.perf_dir}/yearly/yearly_imbalanced_ablation_results.csv"
        df = to_result_df(all_results)
        df.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")

        print_results_table(df, title="Yearly Imbalanced (9:1) — Full Ablation")


if __name__ == "__main__":
    main()
