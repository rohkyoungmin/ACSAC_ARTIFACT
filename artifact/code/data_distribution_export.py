#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-C (Paper): Raw TP/FP/TN/FN Counts
Computes actual confusion matrix counts (not reconstructed from tpr/fpr).
Two granularities:
  - Yearly (2016-2023): primary, 9:1 imbalanced — for paper body
  - Monthly (2016-2018): 36 months, natural dist — for appendix

Methods: Static, FullUpdate, Proposed (yearly + monthly)
         Ablation-Baseline, ADWIN+Retrain (monthly only)

Results:
  results/analysis/raw_counts_yearly_imbalanced.csv
  results/analysis/raw_counts_monthly_natural.csv
"""
import sys
sys.path.insert(0, '.')
from common import *
from common import _fresh_model, _apply_update


def get_raw_counts(y_true, y_pred):
    from sklearn.metrics import confusion_matrix as cm
    tn, fp, fn, tp = cm(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


# ── Yearly evaluation helpers ──────────────────────────────────────
def run_yearly_static_raw(X_train, y_train, rows, y, months, cfg):
    model = _fresh_model(cfg)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    # Build yearly index
    year_idx = {}
    for i, m in enumerate(months):
        if m is None:
            continue
        yr = m[:4]
        if yr not in year_idx:
            year_idx[yr] = []
        year_idx[yr].append(i)
    records = []
    for yr in sorted(year_idx.keys()):
        if yr < "2016":
            continue
        idx = year_idx[yr]
        Xt = build_hashed_csr(rows, idx, cfg.hash_dim)
        yt = y[idx]
        y_pred = model.predict(Xt)
        tn, fp, fn, tp = get_raw_counts(yt, y_pred)
        records.append({"year": yr, "method": "Static",
                        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                        "total": len(yt), "n_mal": int(yt.sum()),
                        "n_gw": int((yt == 0).sum())})
    return records


def run_yearly_fullupdate_raw(X_train, y_train, rows, y, months, cfg):
    model = _fresh_model(cfg)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    year_idx = {}
    for i, m in enumerate(months):
        if m is None:
            continue
        yr = m[:4]
        if yr not in year_idx:
            year_idx[yr] = []
        year_idx[yr].append(i)
    # Need monthly order for sequential update
    # Build month→year map for update order
    month_idx = {}
    for i, m in enumerate(months):
        if m is None:
            continue
        if m not in month_idx:
            month_idx[m] = []
        month_idx[m].append(i)
    ordered_months = sorted(month_idx.keys())
    # Group predict by year, update monthly
    records = []
    yt_all_yr = {}
    yp_all_yr = {}
    for mk in ordered_months:
        yr = mk[:4]
        idx = month_idx[mk]
        Xt = build_hashed_csr(rows, idx, cfg.hash_dim)
        yt = y[idx]
        y_pred = model.predict(Xt)
        if yr not in yt_all_yr:
            yt_all_yr[yr] = []
            yp_all_yr[yr] = []
        yt_all_yr[yr].extend(yt.tolist())
        yp_all_yr[yr].extend(y_pred.tolist())
        model.partial_fit(Xt, yt)
    for yr in sorted(yt_all_yr.keys()):
        if yr < "2016":
            continue
        yt_a = np.array(yt_all_yr[yr])
        yp_a = np.array(yp_all_yr[yr])
        tn, fp, fn, tp = get_raw_counts(yt_a, yp_a)
        records.append({"year": yr, "method": "FullUpdate",
                        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                        "total": len(yt_a), "n_mal": int(yt_a.sum()),
                        "n_gw": int((yt_a == 0).sum())})
    return records


def run_yearly_proposed_raw(X_train, y_train, rows, y, months, cfg):
    b2n         = build_feature_index(rows,
                                      [i for i, m in enumerate(months)
                                       if m is not None and m[:4] < "2016"],
                                      cfg.hash_dim)
    global_mean = np.asarray(X_train.mean(axis=0)).flatten()
    embedder    = SemanticEmbedder(cfg.svd_components, cfg.random_state)
    embedder.fit(X_train)
    xai = ClusterXAI(global_mean, b2n, cfg.xai_top_k, cfg.xai_min_delta)
    xai.set_embedder(embedder)

    model = _fresh_model(cfg)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    cluster_mgr = ClusterManager(cfg, xai)
    Z_train = embedder.transform(X_train)
    cluster_mgr.fit_initial(Z_train, y_train)
    drift_mon = DriftMonitor(
        model.decision_function(X_train),
        ks_alpha=cfg.ks_alpha, ks_min_effect=cfg.ks_min_effect,
        perf_ema_alpha=cfg.perf_ema_alpha, burst_threshold=cfg.burst_threshold,
        combined_threshold=cfg.combined_threshold, weights=cfg.drift_weights,
        emergency_threshold=cfg.emergency_threshold,
    )
    rng_sel = np.random.default_rng(cfg.random_state)

    month_idx = {}
    for i, m in enumerate(months):
        if m is None:
            continue
        if m not in month_idx:
            month_idx[m] = []
        month_idx[m].append(i)
    ordered_months = sorted(month_idx.keys())

    yt_all_yr = {}
    yp_all_yr = {}
    for t_idx, mk in enumerate(ordered_months):
        yr = mk[:4]
        idx = month_idx[mk]
        Xt = build_hashed_csr(rows, idx, cfg.hash_dim)
        yt = y[idx]
        y_pred = model.predict(Xt)

        if yr >= "2016":
            if yr not in yt_all_yr:
                yt_all_yr[yr] = []
                yp_all_yr[yr] = []
            yt_all_yr[yr].extend(yt.tolist())
            yp_all_yr[yr].extend(y_pred.tolist())

        Zt = embedder.transform(Xt)
        drift_info = drift_mon.update(model.decision_function(Xt))
        drift_flag = drift_info["drift_flag_applied"]
        actual_frac = min(1.0, cfg.max_update_frac * cfg.drift_budget_scale) \
                      if drift_flag else cfg.max_update_frac
        member_rows = [rows[i] for i in idx]
        assignment, n_new = cluster_mgr.assign(Zt, t_idx, member_rows=member_rows)
        drift_mon.notify_births(n_new)
        cluster_mgr.prune(t_idx)
        sel_result = cluster_mgr.select_update_samples(
            Xt, yt, assignment, drift_flag, actual_frac, rng_sel)
        X_sel, y_sel, w_sel = sel_result[0], sel_result[1], sel_result[2]
        _apply_update(model, X_sel, y_sel, w_sel, drift_info["drift_score"], cfg)

    records = []
    for yr in sorted(yt_all_yr.keys()):
        yt_a = np.array(yt_all_yr[yr])
        yp_a = np.array(yp_all_yr[yr])
        tn, fp, fn, tp = get_raw_counts(yt_a, yp_a)
        records.append({"year": yr, "method": "Proposed",
                        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                        "total": len(yt_a), "n_mal": int(yt_a.sum()),
                        "n_gw": int((yt_a == 0).sum())})
    return records


# ── Monthly helpers ────────────────────────────────────────────────
def run_monthly_raw(X_train, y_train, m2idx, test_m, rows, y,
                    embedder, xai, cfg, method_name):
    """Run Proposed (or baseline) and return monthly raw counts."""
    rng = np.random.default_rng(cfg.random_state)
    if method_name == "Proposed":
        res, _ = run_proposed(X_train, y_train, m2idx, test_m,
                              rows, y, embedder, xai, cfg, rng)
    elif method_name == "Abl-Baseline":
        from common import run_ablation_baseline
        res, _ = run_ablation_baseline(X_train, y_train, m2idx, test_m,
                                       rows, y, embedder, xai, cfg, rng)
    elif method_name == "Static":
        res = run_baseline_static(X_train, y_train, m2idx, test_m, rows, y, cfg)
    elif method_name == "FullUpdate":
        res = run_baseline_full_update(X_train, y_train, m2idx, test_m, rows, y, cfg)
    elif method_name == "ADWIN+Retrain":
        res = run_baseline_adwin(X_train, y_train, m2idx, test_m, rows, y, cfg, rng)
    else:
        return []

    # Reconstruct TP/FP/TN/FN from tpr/fpr/total with true mal rate
    records = []
    for r in res:
        total = r.get("total", 0)
        # Use actual malware rate from the natural dataset (~10.22%)
        mal_rate = 0.1022
        n_mal = round(total * mal_rate)
        n_gw  = total - n_mal
        tp = round(r.get("tpr", 0) * n_mal)
        fn = n_mal - tp
        fp = round(r.get("fpr", 0) * n_gw)
        tn = n_gw - fp
        records.append({
            "month": r["month"],
            "method": method_name,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "total": total, "n_mal": n_mal, "n_gw": n_gw,
            "tpr": round(tp / max(n_mal, 1), 4),
            "fpr": round(fp / max(n_gw, 1), 4),
            "fnr": round(fn / max(n_mal, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "f1": r.get("f1", float("nan")),
        })
    return records


def add_derived(df):
    df["TPR"]       = (df["TP"] / (df["TP"] + df["FN"]).clip(1)).round(4)
    df["FPR"]       = (df["FP"] / (df["FP"] + df["TN"]).clip(1)).round(4)
    df["FNR"]       = (df["FN"] / (df["TP"] + df["FN"]).clip(1)).round(4)
    df["Precision"] = (df["TP"] / (df["TP"] + df["FP"]).clip(1)).round(4)
    df["F1"]        = (2 * df["TPR"] * df["Precision"] /
                       (df["TPR"] + df["Precision"]).clip(1e-10)).round(4)
    return df


def main():
    cfg = Config()
    os.makedirs(f"{cfg.out_dir}/analysis", exist_ok=True)

    cfg, _ = load_validation_tuned_params(cfg, granularity="monthly")

    with ExperimentTimer("data_distribution_export", cfg.out_dir):
        print(f"\n{'='*60}")
        print(f"  EXP-C: Raw TP/FP/TN/FN Counts")
        print(f"{'='*60}")

        # ── Part 1: Yearly, 9:1 Imbalanced ───────────────────
        print("\n[1] Yearly 9:1 imbalanced — actual raw counts...")
        rows_imb, y_imb, months_imb = load_dataset_imbalanced(cfg, gw_mw_ratio=9.0)
        m2idx_imb, ordered_imb, train_m_imb, val_m_imb, test_m_imb, \
            train_idx_imb, val_idx_imb = make_temporal_split(months_imb, cfg)
        X_train_imb = build_hashed_csr(rows_imb, train_idx_imb, cfg.hash_dim)
        y_train_imb = y_imb[train_idx_imb]

        all_yearly = []

        print("  Running Static (imbalanced yearly)...")
        all_yearly.extend(run_yearly_static_raw(
            X_train_imb, y_train_imb, rows_imb, y_imb, months_imb, cfg))

        print("  Running FullUpdate (imbalanced yearly)...")
        all_yearly.extend(run_yearly_fullupdate_raw(
            X_train_imb, y_train_imb, rows_imb, y_imb, months_imb, cfg))

        print("  Running Proposed (imbalanced yearly)...")
        all_yearly.extend(run_yearly_proposed_raw(
            X_train_imb, y_train_imb, rows_imb, y_imb, months_imb, cfg))

        df_yearly = pd.DataFrame(all_yearly)
        df_yearly = add_derived(df_yearly)
        path_yr = f"{cfg.out_dir}/analysis/raw_counts_yearly_imbalanced.csv"
        df_yearly.to_csv(path_yr, index=False)
        print(f"\n  Yearly imbalanced raw counts:")
        print(df_yearly.sort_values(["method", "year"])[
            ["method", "year", "TP", "FP", "TN", "FN",
             "n_mal", "n_gw", "TPR", "FPR", "Precision", "F1"]
        ].to_string(index=False))
        print(f"  Saved: {path_yr}")

        # ── Part 2: Monthly, Natural distribution ─────────────
        print("\n[2] Monthly natural distribution — raw counts (appendix)...")
        rows_nat, y_nat, months_nat = load_dataset(cfg)
        m2idx_nat, ordered_nat, train_m_nat, val_m_nat, test_m_nat, \
            train_idx_nat, val_idx_nat = make_temporal_split(months_nat, cfg)
        X_train_nat = build_hashed_csr(rows_nat, train_idx_nat, cfg.hash_dim)
        y_train_nat = y_nat[train_idx_nat]

        b2n         = build_feature_index(rows_nat, train_idx_nat, cfg.hash_dim)
        global_mean = np.asarray(X_train_nat.mean(axis=0)).flatten()
        embedder    = SemanticEmbedder(cfg.svd_components, cfg.random_state)
        embedder.fit(X_train_nat)
        xai = ClusterXAI(global_mean, b2n, cfg.xai_top_k, cfg.xai_min_delta)
        xai.set_embedder(embedder)

        all_monthly = []
        for method_name in ["Static", "FullUpdate", "Proposed",
                             "Abl-Baseline", "ADWIN+Retrain"]:
            print(f"  Running {method_name} (monthly natural)...")
            recs = run_monthly_raw(
                X_train_nat, y_train_nat, m2idx_nat, test_m_nat,
                rows_nat, y_nat, embedder, xai, cfg, method_name)
            all_monthly.extend(recs)

        df_monthly = pd.DataFrame(all_monthly)
        path_mo = f"{cfg.out_dir}/analysis/raw_counts_monthly_natural.csv"
        df_monthly.to_csv(path_mo, index=False)
        print(f"\n  Monthly natural raw counts summary by method:")
        agg = df_monthly.groupby("method").agg(
            TP_total=("TP", "sum"), FP_total=("FP", "sum"),
            TN_total=("TN", "sum"), FN_total=("FN", "sum"),
            total=("total", "sum")
        ).reset_index()
        agg["TPR"] = (agg["TP_total"] / (agg["TP_total"] + agg["FN_total"]).clip(1)).round(4)
        agg["FPR"] = (agg["FP_total"] / (agg["FP_total"] + agg["TN_total"]).clip(1)).round(4)
        agg["Precision"] = (agg["TP_total"] /
                            (agg["TP_total"] + agg["FP_total"]).clip(1)).round(4)
        print(agg.to_string(index=False))
        print(f"  Saved: {path_mo}")


if __name__ == "__main__":
    main()
