#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster Contribution Analysis
"Why is 33% more informative?" — per-cluster selection ratio vs performance contribution

분석 내용:
  A. Per-cluster selection allocation 추적 (drift-critical vs stable)
  B. Cluster type별 선택 비율 vs 다음 달 F1 변화 상관관계
  C. Proposed vs Ablation-Baseline: 선택된 샘플의 정보량 비교
     - mal_rate of selected samples (class representativeness)
     - New-cluster sample ratio (novelty coverage)
     - Loss reduction per selected sample

실행: python cluster_review_behavior.py
결과: results/analysis/cluster_contribution.csv            (natural)
      results/analysis/cluster_type_selection.csv          (natural)
      results/analysis/informative_comparison.csv          (natural)
      results/analysis/monthly_dc_analysis.csv             (natural)
      results/analysis/cluster_contribution_imbalanced.csv (9:1)
      results/analysis/cluster_type_selection_imbalanced.csv
      results/analysis/informative_comparison_imbalanced.csv
      results/analysis/monthly_dc_analysis_imbalanced.csv
소요시간 예상: ~60-80분 (monthly 4회 실행)
"""
import sys
import json
import time
sys.path.insert(0, '.')
from common import *
from common import _fresh_model, _apply_update


def run_with_cluster_tracking(
    X_train, y_train, month_to_idx, test_months,
    rows, y, embedder, xai, cfg, rng,
    method_name, use_cluster_selection, use_drift_loss,
):
    """
    _run_selective_update_core와 동일 로직이지만
    per-cluster selection 상세를 매 월 기록한다.
    """
    cfg_run = Config(**{k: v for k, v in cfg.__dict__.items()})
    cfg_run.loss_mode = "drift_reg" if use_drift_loss else "standard"

    model = _fresh_model(cfg_run)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))

    cluster_mgr = None
    if use_cluster_selection:
        cluster_mgr = ClusterManager(cfg_run, xai)
        Z_train     = embedder.transform(X_train)
        cluster_mgr.fit_initial(Z_train, y_train)

    energy_train = model.decision_function(X_train)
    drift_mon = DriftMonitor(
        energy_train,
        ks_alpha            = cfg_run.ks_alpha,
        ks_min_effect       = cfg_run.ks_min_effect,
        perf_ema_alpha      = cfg_run.perf_ema_alpha,
        burst_threshold     = cfg_run.burst_threshold,
        combined_threshold  = cfg_run.combined_threshold,
        weights             = cfg_run.drift_weights,
        emergency_threshold = cfg_run.emergency_threshold,
    )

    perf_results  = []   # standard monthly metrics
    cluster_alloc = []   # per-cluster allocation detail
    prev_f1       = None

    for t_idx, mk in enumerate(test_months):
        idx_m = month_to_idx.get(mk, [])
        if not idx_m:
            continue

        t0  = time.perf_counter()
        Xt  = build_hashed_csr(rows, idx_m, cfg_run.hash_dim)
        yt  = y[idx_m]

        y_pred  = model.predict(Xt)
        y_prob  = model.predict_proba(Xt)[:, 1]
        metrics = compute_metrics(yt, y_pred, y_prob)

        loss_before = compute_loss_profile(model, Xt, yt)
        energy_t    = model.decision_function(Xt)
        drift_info  = drift_mon.update(energy_t)
        drift_flag  = drift_info["drift_flag_applied"]
        emergency   = drift_info.get("emergency_flag", False)

        actual_frac = cfg_run.max_update_frac
        if drift_flag or emergency:
            actual_frac = min(1.0, cfg_run.max_update_frac * cfg_run.drift_budget_scale)

        n_new, n_pruned = 0, 0
        assignment = np.full(len(yt), -1, dtype=np.int32)
        per_cl     = {}

        if use_cluster_selection and cluster_mgr is not None:
            Zt = embedder.transform(Xt)
            member_rows = [rows[i] for i in idx_m]
            assignment, n_new = cluster_mgr.assign(Zt, t_idx, member_rows=member_rows)
            drift_mon.notify_births(n_new)
            n_pruned = cluster_mgr.prune(t_idx)

            X_sel, y_sel, sel_w, per_cl = cluster_mgr.select_update_samples(
                Xt, yt, assignment, drift_flag, actual_frac, rng)

            # ── Per-cluster allocation detail ──────────────
            K = len(cluster_mgr.centroids)
            n_total_month = len(yt)
            budget = max(1, int(n_total_month * actual_frac))

            for cid in range(K):
                n_in_cluster   = int((assignment == cid).sum())
                n_selected     = per_cl.get(cid, 0)
                is_new_cluster = bool(cluster_mgr.is_new[cid])
                ld             = cluster_mgr.label_dist[cid]
                total_ld       = ld.sum()
                mal_ratio_cid  = float(ld[1] / total_ld) if total_ld > 0 else 0.0

                # drift-critical: is_new OR drift_flag active
                is_drift_critical = is_new_cluster or drift_flag

                cluster_alloc.append({
                    "month"            : mk,
                    "method"           : method_name,
                    "cluster_id"       : cid,
                    "n_in_cluster"     : n_in_cluster,
                    "n_selected"       : n_selected,
                    "selection_rate"   : round(n_selected / n_in_cluster, 4) if n_in_cluster > 0 else 0.0,
                    "budget_share"     : round(n_selected / budget, 4) if budget > 0 else 0.0,
                    "is_new_cluster"   : is_new_cluster,
                    "is_drift_critical": is_drift_critical,
                    "drift_flag"       : drift_flag,
                    "cluster_mal_ratio": round(mal_ratio_cid, 4),
                    "cluster_age"      : cluster_mgr.age[cid],
                    "n_clusters_total" : K,
                    "drift_score"      : drift_info["drift_score"],
                    "f1_this_month"    : round(metrics["f1"], 4),
                    "f1_prev_month"    : round(prev_f1, 4) if prev_f1 is not None else None,
                    "f1_delta"         : round(metrics["f1"] - prev_f1, 4) if prev_f1 is not None else None,
                })
        else:
            # Baseline: uniform random
            budget  = max(1, int(len(yt) * actual_frac))
            chosen  = rng.choice(len(yt), size=min(budget, len(yt)), replace=False)
            X_sel   = Xt[chosen]
            y_sel   = yt[chosen]
            sel_w   = np.ones(len(y_sel), dtype=np.float64)

        _apply_update(model, X_sel, y_sel, sel_w,
                      drift_info["drift_score"], cfg_run)
        loss_after = compute_loss_profile(model, Xt, yt)

        row = {
            "month"              : mk,
            "method"             : method_name,
            "f1"                 : metrics["f1"],
            "auc_pr"             : metrics["auc_pr"],
            "fpr_at_tpr90"       : metrics["fpr_at_tpr90"],
            "updated"            : X_sel.shape[0],
            "total"              : len(yt),
            "actual_frac"        : actual_frac,
            "drift_score"        : drift_info["drift_score"],
            "drift_detected"     : drift_info["drift_detected"],
            "n_new_clusters"     : n_new,
            "loss_before"        : round(loss_before["loss_mean"], 6),
            "loss_after"         : round(loss_after["loss_mean"], 6),
            "loss_delta"         : round(loss_after["loss_mean"] - loss_before["loss_mean"], 6),
            "loss_per_sample"    : round((loss_before["loss_mean"] - loss_after["loss_mean"]) /
                                         max(X_sel.shape[0], 1), 6),
            "mal_rate_selected"  : compute_label_efficiency(y_sel),
            "f1_prev"            : prev_f1,
            "f1_change"          : round(metrics["f1"] - prev_f1, 4) if prev_f1 is not None else None,
        }
        perf_results.append(row)
        prev_f1 = metrics["f1"]

    return perf_results, cluster_alloc


def run_condition(label, rows, y, months, cfg, out_dir, suffix=""):
    """
    One full cluster-contribution analysis pass on a single dataset condition.
    suffix: "" for natural, "_imbalanced" for 9:1.
    """
    print(f"\n{'='*60}")
    print(f"  CONDITION: {label}")
    print(f"{'='*60}")

    print(f"\n[{label}] Temporal split...")
    m2idx, ordered, train_m, val_m, test_m, train_idx, val_idx = \
        make_temporal_split(months, cfg)

    print(f"\n[{label}] Building features + embedder...")
    b2n     = build_feature_index(rows, train_idx, cfg.hash_dim)
    X_train = build_hashed_csr(rows, train_idx, cfg.hash_dim)
    y_train = y[train_idx]
    print(f"  X_train: {X_train.shape}  nnz={X_train.nnz:,}  mal_rate={y_train.mean():.4f}")

    global_mean = np.asarray(X_train.mean(axis=0)).flatten()
    embedder    = SemanticEmbedder(cfg.svd_components, cfg.random_state)
    embedder.fit(X_train)

    xai = ClusterXAI(global_mean, b2n, cfg.xai_top_k, cfg.xai_min_delta)
    xai.set_embedder(embedder)

    # Run Proposed with per-cluster tracking
    print(f"\n[{label}] Running Proposed with per-cluster allocation tracking...")
    prop_perf, prop_alloc = run_with_cluster_tracking(
        X_train, y_train, m2idx, test_m,
        rows, y, embedder, xai, cfg,
        rng=np.random.default_rng(cfg.random_state),
        method_name="Proposed",
        use_cluster_selection=True,
        use_drift_loss=True,
    )

    # Run Baseline for comparison
    print(f"\n[{label}] Running Ablation-Baseline for comparison...")
    base_perf, _ = run_with_cluster_tracking(
        X_train, y_train, m2idx, test_m,
        rows, y, embedder, xai, cfg,
        rng=np.random.default_rng(cfg.random_state),
        method_name="Ablation-Baseline",
        use_cluster_selection=False,
        use_drift_loss=False,
    )

    # Save raw results
    df_alloc = pd.DataFrame(prop_alloc)
    alloc_path = os.path.join(out_dir, f"cluster_contribution{suffix}.csv")
    df_alloc.to_csv(alloc_path, index=False)
    print(f"  Saved: {alloc_path}  ({len(df_alloc)} rows)")

    df_perf = pd.DataFrame(prop_perf + base_perf)
    comp_path = os.path.join(out_dir, f"informative_comparison{suffix}.csv")
    df_perf.to_csv(comp_path, index=False)
    print(f"  Saved: {comp_path}")

    # Analysis A: Cluster type selection ratios
    print(f"\n{'='*60}")
    print(f"  ANALYSIS A [{label}]: Selection ratio by cluster type")
    print(f"{'='*60}")

    if not df_alloc.empty:
        new_cl    = df_alloc[df_alloc.is_new_cluster == True]
        drift_cl  = df_alloc[(df_alloc.is_new_cluster == False) & (df_alloc.drift_flag == True)]
        stable_cl = df_alloc[(df_alloc.is_new_cluster == False) & (df_alloc.drift_flag == False)]

        for lbl, sub in [("New clusters    (weight=2.0)", new_cl),
                          ("Drift clusters  (weight=1.5)", drift_cl),
                          ("Stable clusters (weight=0.5)", stable_cl)]:
            if sub.empty:
                print(f"  {lbl}: no data")
                continue
            print(f"  {lbl}:")
            print(f"    avg selection_rate  = {sub.selection_rate.mean():.4f}")
            print(f"    avg budget_share    = {sub.budget_share.mean():.4f}")
            print(f"    n cluster-months    = {len(sub)}")

        type_summary = []
        for lbl, cat, sub in [
            ("New",    "new",    new_cl),
            ("Drift",  "drift",  drift_cl),
            ("Stable", "stable", stable_cl),
        ]:
            if sub.empty:
                continue
            type_summary.append({
                "cluster_type"          : lbl,
                "n_cluster_months"      : len(sub),
                "mean_selection_rate"   : round(sub.selection_rate.mean(), 4),
                "mean_budget_share"     : round(sub.budget_share.mean(), 4),
                "mean_n_selected"       : round(sub.n_selected.mean(), 2),
                "mean_cluster_mal_ratio": round(sub.cluster_mal_ratio.mean(), 4),
            })
        df_type = pd.DataFrame(type_summary)
        type_path = os.path.join(out_dir, f"cluster_type_selection{suffix}.csv")
        df_type.to_csv(type_path, index=False)
        print(f"\n  Saved: {type_path}")
        print(df_type.to_string(index=False))

    # Analysis B: Drift-critical budget share vs F1 change
    print(f"\n{'='*60}")
    print(f"  ANALYSIS B [{label}]: Drift-critical budget share vs F1 change")
    print(f"{'='*60}")

    if not df_alloc.empty:
        monthly_dc = df_alloc.groupby("month").apply(lambda g: pd.Series({
            "dc_budget_share"    : g[g.is_drift_critical == True]["budget_share"].sum(),
            "new_budget_share"   : g[g.is_new_cluster == True]["budget_share"].sum(),
            "stable_budget_share": g[g.is_drift_critical == False]["budget_share"].sum(),
            "n_new_clusters"     : int(g.is_new_cluster.sum()),
            "f1"                 : g.f1_this_month.iloc[0],
            "f1_delta"           : g.f1_delta.iloc[0],
            "drift_score"        : g.drift_score.iloc[0],
        })).reset_index()

        valid = monthly_dc.dropna(subset=["f1_delta", "dc_budget_share"])
        if len(valid) > 3:
            from scipy.stats import pearsonr, spearmanr
            r_p, p_p = pearsonr(valid.dc_budget_share, valid.f1_delta)
            r_s, p_s = spearmanr(valid.dc_budget_share, valid.f1_delta)
            print(f"  dc_budget_share vs f1_delta:  Pearson r={r_p:.4f} (p={p_p:.4f}),  Spearman ρ={r_s:.4f} (p={p_s:.4f})")

            r_p2, p_p2 = pearsonr(valid.new_budget_share, valid.f1_delta)
            print(f"  new_budget_share vs f1_delta: Pearson r={r_p2:.4f} (p={p_p2:.4f})")

        dc_path = os.path.join(out_dir, f"monthly_dc_analysis{suffix}.csv")
        monthly_dc.to_csv(dc_path, index=False)
        print(f"  Saved: {dc_path}")

    # Analysis C: Proposed vs Baseline informativeness
    print(f"\n{'='*60}")
    print(f"  ANALYSIS C [{label}]: Proposed vs Baseline — selection informativeness")
    print(f"{'='*60}")

    df_p = pd.DataFrame(prop_perf)
    df_b = pd.DataFrame(base_perf)

    print(f"  {'Metric':<30} {'Proposed':>12} {'Baseline':>12} {'Delta':>10}")
    print(f"  {'-'*66}")

    metrics_to_compare = [
        ("Mean F1",                 "f1"),
        ("Mean AUC-PR",             "auc_pr"),
        ("Mean mal_rate_selected",  "mal_rate_selected"),
        ("Mean loss_per_sample",    "loss_per_sample"),
        ("Mean |loss_delta|",       "loss_delta"),
    ]
    for lbl, col in metrics_to_compare:
        if col not in df_p.columns:
            continue
        pv = df_p[col].abs().mean() if col == "loss_delta" else df_p[col].mean()
        bv = df_b[col].abs().mean() if col == "loss_delta" else df_b[col].mean()
        print(f"  {lbl:<30} {pv:>12.4f} {bv:>12.4f} {pv-bv:>+10.4f}")

    print(f"\n  Label cost: Proposed={df_p.updated.sum()/df_p.total.sum()*100:.1f}%  "
          f"Baseline={df_b.updated.sum()/df_b.total.sum()*100:.1f}%")

    drift_months_p  = df_p[df_p.drift_detected == True]
    stable_months_p = df_p[df_p.drift_detected == False]
    drift_months_b  = df_b[df_b.drift_detected == True]
    stable_months_b = df_b[df_b.drift_detected == False]

    print(f"\n  Performance on DRIFT months (drift_detected=True):")
    print(f"    Proposed  F1: {drift_months_p.f1.mean():.4f}  (n={len(drift_months_p)} months)")
    print(f"    Baseline  F1: {drift_months_b.f1.mean():.4f}  (n={len(drift_months_b)} months)")
    print(f"\n  Performance on STABLE months:")
    print(f"    Proposed  F1: {stable_months_p.f1.mean():.4f}  (n={len(stable_months_p)} months)")
    print(f"    Baseline  F1: {stable_months_b.f1.mean():.4f}  (n={len(stable_months_b)} months)")


def main():
    cfg = Config()
    out_dir = os.path.join(cfg.out_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    cfg, _ = load_validation_tuned_params(cfg, granularity="monthly")

    with ExperimentTimer("cluster_review_behavior", cfg.out_dir):
        print(f"\n{'='*60}")
        print(f"  Cluster Contribution Analysis")
        print(f"  Why is 33% more informative?")
        print(f"{'='*60}")

        # ── CONDITION 1: Natural distribution ─────────────
        print("\n[1] Loading dataset (natural distribution)...")
        rows_n, y_n, months_n = load_dataset(cfg)
        run_condition(
            label="Monthly Natural",
            rows=rows_n, y=y_n, months=months_n,
            cfg=cfg, out_dir=out_dir, suffix="",
        )

        # ── CONDITION 2: Imbalanced 9:1 ───────────────────
        print("\n[2] Loading dataset (9:1 imbalanced)...")
        rows_i, y_i, months_i = load_dataset_imbalanced(cfg, gw_mw_ratio=9.0)
        run_condition(
            label="Monthly Imbalanced (9:1)",
            rows=rows_i, y=y_i, months=months_i,
            cfg=cfg, out_dir=out_dir, suffix="_imbalanced",
        )

        print(f"\n{'='*60}")
        print(f"  All outputs saved to: {out_dir}/")
        print(f"  Natural:    cluster_contribution.csv, cluster_type_selection.csv,")
        print(f"              informative_comparison.csv, monthly_dc_analysis.csv")
        print(f"  Imbalanced: cluster_contribution_imbalanced.csv, etc.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
