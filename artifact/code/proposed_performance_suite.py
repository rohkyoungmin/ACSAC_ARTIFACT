#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proposed-performance CSV suite for ACSAC submission experiments.

This script focuses on the proposed-method family and writes all required
submission-facing CSVs under:

  results/proposed_performance/

Outputs:
  00_protocol_summary.csv
  00_validation_tuned_params.csv  (copy of monthly validation JSON parameters)
  01_main_performance.csv
  01_main_performance_summary.csv
  02_module_ablation.csv
  02_module_ablation_summary.csv
  03_drift_signal_ablation.csv
  03_drift_signal_ablation_summary.csv
  04_label_budget_sensitivity.csv
  05_data_health_monthly_balance.csv
  05_data_health_duplicate_summary.csv
  06_xai_top_feature_stability.csv
  06_xai_backprojection_fidelity.csv
  06_xai_case_study_candidates.csv
  06_xai_validation_summary.csv
  07_runtime_update_overhead.csv
  08_same_review_ratio_selective_baselines.csv
  08_same_review_ratio_selective_baselines_summary.csv
"""

import hashlib
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import common as common_mod
from common import *


OUT_DIR_NAME = "proposed_performance"
DRIFT_VARIANTS: List[Tuple[str, tuple]] = [
    ("KS-only",         (1.00, 0.00, 0.00)),
    ("Score-only",      (0.00, 1.00, 0.00)),
    ("Novelty-only",    (0.00, 0.00, 1.00)),
    ("KS+Score",        (0.50, 0.50, 0.00)),
    ("KS+Novelty",      (0.50, 0.00, 0.50)),
    ("Full",            (0.40, 0.30, 0.30)),
]
BUDGETS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.45, 1.00]


def proposed_dir(cfg: Config) -> str:
    path = os.path.join(cfg.out_dir, OUT_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _proposed_param_dict(cfg: Config, source: str) -> Dict:
    tuned = {"source": source, "granularity": "monthly"}
    for name in VALIDATION_TUNED_PARAM_NAMES:
        value = getattr(cfg, name)
        if isinstance(value, tuple):
            value = list(value)
        tuned[name] = value
    return tuned


def save_loaded_validation_params(cfg: Config, source_path: str, out_dir: str) -> None:
    tuned = _proposed_param_dict(cfg, source=source_path)
    json_path = os.path.join(out_dir, "00_validation_tuned_params.json")
    csv_path = os.path.join(out_dir, "00_validation_tuned_params.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tuned, f, indent=2)
    pd.DataFrame([tuned]).to_csv(csv_path, index=False)
    print(f"  Saved monthly tuned-parameter copy: {json_path}")
    print(f"  Saved monthly tuned-parameter copy: {csv_path}")


def tune_on_validation_split(
    X_train, y_train, m2idx, val_m, rows, y, embedder, xai, cfg: Config, out_dir: str
) -> Config:
    """
    Clean P5 tuning path for the submission suite: tune only the Proposed
    parameters on the validation months, then save the filtered parameter set.
    """
    print("\n[Tuning] Running validation-set grid search for Proposed params...")
    val_m2idx = {m: m2idx[m] for m in val_m}
    tuned_cfg = tune_on_val(
        X_train, y_train, val_m2idx, val_m,
        rows, y, embedder, xai, cfg,
        rng=np.random.default_rng(cfg.random_state),
    )
    tuned = _proposed_param_dict(tuned_cfg, source="validation_grid_search")

    json_path = os.path.join(out_dir, "00_validation_tuned_params.json")
    csv_path = os.path.join(out_dir, "00_validation_tuned_params.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tuned, f, indent=2)
    pd.DataFrame([tuned]).to_csv(csv_path, index=False)
    print(f"  Saved: {json_path}")
    print(f"  Saved: {csv_path}")
    return tuned_cfg


def export_protocol_summary(cfg: Config, train_m, val_m, test_m, out_dir: str):
    """
    Record the evaluation protocol explicitly so the CSV directory documents the
    Arp et al. security-ML checks: temporal split, validation-only tuning,
    no future/test information for model selection, and base-rate-aware metrics.
    """
    rows = [
        {
            "item": "temporal_split",
            "setting": "train_then_validate_then_future_test",
            "detail": (
                "Train uses only 2014 months, validation uses only 2015 months, "
                "and all months after 2015 are treated as future test months."
            ),
        },
        {
            "item": "train_months",
            "setting": f"{train_m[0]}..{train_m[-1]}" if train_m else "",
            "detail": "Initial detector, semantic SVD, initial clusters, and XAI background are fit on train only.",
        },
        {
            "item": "validation_months",
            "setting": f"{val_m[0]}..{val_m[-1]}" if val_m else "",
            "detail": "Only validation months are used for hyperparameter selection.",
        },
        {
            "item": "test_months",
            "setting": f"{test_m[0]}..{test_m[-1]}" if test_m else "",
            "detail": "Test months are consumed chronologically and are never used to tune parameters.",
        },
        {
            "item": "json_parameters",
            "setting": ", ".join(VALIDATION_TUNED_PARAM_NAMES),
            "detail": "These Proposed parameters are loaded from the monthly JSON; shared representation/protocol/reporting settings are fixed before validation, while drift/update decision parameters are validation-selected.",
        },
        {
            "item": "primary_selection_metric",
            "setting": "validation_mean_auc_pr",
            "detail": "AUC-PR is used for validation selection because malware detection is class-imbalanced.",
        },
        {
            "item": "reported_metrics",
            "setting": "Macro-F1, AUC-PR, AUC-ROC, FPR@TPR=90%, label_update_ratio, runtime",
            "detail": "Accuracy alone is intentionally excluded from the main claim because it is base-rate sensitive.",
        },
        {
            "item": "future_information_rule",
            "setting": "no_test_tuning_no_preloading_future_labels",
            "detail": "Cluster assignment and drift monitoring use unlabeled monthly features/scores; labels are used only for evaluation and selected update samples.",
        },
    ]
    path = os.path.join(out_dir, "00_protocol_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")


def build_components(X_train, train_idx, rows, cfg):
    b2n = build_feature_index(rows, train_idx, cfg.hash_dim)
    global_mean = np.asarray(X_train.mean(axis=0)).flatten()
    embedder = SemanticEmbedder(cfg.svd_components, cfg.random_state)
    embedder.fit(X_train)
    xai = ClusterXAI(global_mean, b2n, cfg.xai_top_k, cfg.xai_min_delta)
    xai.set_embedder(embedder)
    return embedder, xai


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for method, sub in df.groupby("method", sort=False):
        rows.append({
            "method": method,
            "months": int(sub["month"].nunique()) if "month" in sub else len(sub),
            "macro_f1_mean": float(sub["f1"].mean()),
            "macro_f1_std": float(sub["f1"].std()),
            "auc_pr_mean": float(sub["auc_pr"].mean()),
            "auc_roc_mean": float(sub["auc_roc"].mean()),
            "fpr_at_tpr90_mean": float(sub["fpr_at_tpr90"].mean()),
            "label_update_ratio": float(sub["updated"].sum() / sub["total"].sum())
                                  if {"updated", "total"}.issubset(sub.columns) else float("nan"),
            "time_sec_mean": float(sub["time_sec"].mean())
                             if "time_sec" in sub.columns else float("nan"),
            "time_sec_total": float(sub["time_sec"].sum())
                              if "time_sec" in sub.columns else float("nan"),
            "drift_detected_months": int(sub["drift_detected"].fillna(False).sum())
                                      if "drift_detected" in sub.columns else 0,
            "new_clusters_total": int(sub["n_new_clusters"].fillna(0).sum())
                                  if "n_new_clusters" in sub.columns else 0,
        })
    return pd.DataFrame(rows)


def save_detail_and_summary(df: pd.DataFrame, out_dir: str, stem: str):
    detail_path = os.path.join(out_dir, f"{stem}.csv")
    summary_path = os.path.join(out_dir, f"{stem}_summary.csv")
    df.to_csv(detail_path, index=False)
    summarize_results(df).to_csv(summary_path, index=False)
    print(f"  Saved: {detail_path}")
    print(f"  Saved: {summary_path}")


def _load_meta(cfg: Config) -> List[Dict]:
    meta_path = os.path.join(cfg.data_dir, "extended-features-meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sample_id(row: Dict, meta: Dict) -> str:
    for field in ("sha256", "sha1", "md5"):
        value = str(meta.get(field, "")).strip()
        if value:
            return f"{field}:{value.upper()}"
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return "feature:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _period(month: str, cfg: Config) -> str:
    if not month:
        return "missing_date"
    year = int(str(month)[:4])
    if year == cfg.train_year:
        return "train"
    if year == cfg.val_year:
        return "val"
    if year > cfg.val_year:
        return "test"
    return "pretrain"


def build_data_health(rows, y, months, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = _load_meta(cfg)
    records = []
    for i, label in enumerate(y):
        month = months[i]
        records.append({
            "orig_index": i,
            "sample_id": _sample_id(rows[i], meta[i]),
            "month": month,
            "period": _period(month, cfg),
            "label": int(label),
        })
    df = pd.DataFrame(records)
    clean = df.dropna(subset=["month"]).copy()

    balance = clean.groupby("month", sort=True)["label"].agg(
        total="count", malicious="sum").reset_index()
    balance["benign"] = balance["total"] - balance["malicious"]
    balance["malicious_rate"] = balance["malicious"] / balance["total"]
    balance = balance[["month", "total", "benign", "malicious", "malicious_rate"]]

    monthly_dup_rows = []
    for month, sub in clean.groupby("month", sort=True):
        sizes = sub.groupby("sample_id").size()
        dup_sizes = sizes[sizes > 1]
        label_nunique = sub.groupby("sample_id")["label"].nunique()
        monthly_dup_rows.append({
            "month": month,
            "total": int(len(sub)),
            "unique_ids": int(sizes.shape[0]),
            "same_month_duplicate_ids": int((sizes > 1).sum()),
            "same_month_duplicate_rows": int((dup_sizes - 1).sum()) if len(dup_sizes) else 0,
            "label_conflicting_duplicate_ids": int((label_nunique > 1).sum()),
        })
    monthly_dup = pd.DataFrame(monthly_dup_rows)

    sizes = df.groupby("sample_id").size()
    duplicate_ids = sizes[sizes > 1].index
    same_month_sizes = df.groupby(["sample_id", "month"], dropna=False).size()
    same_month_ids = set(idx[0] for idx in same_month_sizes[same_month_sizes > 1].index)
    month_counts = df.groupby("sample_id")["month"].nunique(dropna=True)
    cross_ids = month_counts[month_counts > 1].index
    period_sets = df.groupby("sample_id")["period"].agg(lambda s: set(s))
    train_test_ids = [
        sid for sid, periods in period_sets.items()
        if "train" in periods and "test" in periods
    ]
    label_counts = df.groupby("sample_id")["label"].nunique()
    conflict_ids = label_counts[label_counts > 1].index

    def row_count(ids) -> int:
        if len(ids) == 0:
            return 0
        return int(df[df["sample_id"].isin(ids)].shape[0])

    dup_summary = pd.DataFrame([{
        "n_samples": int(len(df)),
        "n_unique_ids": int(sizes.shape[0]),
        "total_duplicate_ids": int(len(duplicate_ids)),
        "total_duplicate_rows_extra": int((sizes[sizes > 1] - 1).sum()) if len(duplicate_ids) else 0,
        "same_month_duplicate_ids": int(len(same_month_ids)),
        "same_month_duplicate_rows_extra": int(monthly_dup["same_month_duplicate_rows"].sum()),
        "cross_month_duplicate_ids": int(len(cross_ids)),
        "cross_month_duplicate_rows": row_count(cross_ids),
        "train_test_duplicate_ids": int(len(train_test_ids)),
        "train_test_duplicate_rows": row_count(train_test_ids),
        "label_conflicting_duplicate_ids": int(len(conflict_ids)),
        "label_conflicting_duplicate_rows": row_count(conflict_ids),
    }])
    return balance, monthly_dup, dup_summary


def run_main_performance(X_train, y_train, m2idx, ordered, train_m, test_m,
                         rows, y, embedder, xai, cfg, out_dir):
    print("\n[Main] Running proposed-family main performance suite...")
    rng = np.random.default_rng(cfg.random_state)
    all_by_month = {m: m2idx[m] for m in ordered}

    runs = []
    runs.extend(run_baseline_static(X_train, y_train, m2idx, test_m, rows, y, cfg))
    runs.extend(run_baseline_retraining(all_by_month, train_m, test_m, rows, y, cfg, rng))
    runs.extend(run_baseline_full_update(X_train, y_train, m2idx, test_m, rows, y, cfg))

    random_res, _ = run_ablation_baseline(
        X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg,
        np.random.default_rng(cfg.random_state))
    uncertainty_res, _ = run_uncertainty_update(
        X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg,
        np.random.default_rng(cfg.random_state))
    score_shift_res, _ = run_score_shift_update(
        X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg,
        np.random.default_rng(cfg.random_state))
    drift_only_res, _ = run_ablation_drift_loss_only(
        X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg,
        np.random.default_rng(cfg.random_state))
    proposed_res, xai_snaps = run_proposed(
        X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg,
        np.random.default_rng(cfg.random_state))
    proposed_review_counts = {
        row["month"]: int(row["updated"]) for row in proposed_res
    }
    random_matched_res, _ = run_matched_random_update(
        X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg,
        np.random.default_rng(cfg.random_state), proposed_review_counts)
    uncertainty_matched_res, _ = run_matched_uncertainty_update(
        X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg,
        np.random.default_rng(cfg.random_state), proposed_review_counts)
    score_shift_matched_res, _ = run_matched_score_shift_update(
        X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg,
        np.random.default_rng(cfg.random_state), proposed_review_counts)

    rename = {
        "Static": "Static",
        "Retraining": "PeriodicRetraining",
        "FullUpdate": "FullLabelUpdate",
        "Random+Standard": "RandomSelectiveUpdate",
        "UncertaintyOnly": "UncertaintyOnlyUpdate",
        "ScoreShiftOnly": "ScoreShiftOnlyUpdate",
        "Random+DriftLoss": "DriftOnlyUpdate",
        "RandomMatchedUpdate": "RandomMatchedUpdate",
        "UncertaintyMatchedUpdate": "UncertaintyMatchedUpdate",
        "ScoreShiftMatchedUpdate": "ScoreShiftMatchedUpdate",
        "Proposed": "Proposed",
    }
    runs.extend(random_res)
    runs.extend(uncertainty_res)
    runs.extend(score_shift_res)
    runs.extend(drift_only_res)
    runs.extend(random_matched_res)
    runs.extend(uncertainty_matched_res)
    runs.extend(score_shift_matched_res)
    runs.extend(proposed_res)

    df = to_result_df(runs)
    df["method"] = df["method"].map(lambda m: rename.get(m, m))
    save_detail_and_summary(df, out_dir, "01_main_performance")

    matched_methods = [
        "RandomMatchedUpdate",
        "UncertaintyMatchedUpdate",
        "ScoreShiftMatchedUpdate",
        "Proposed",
    ]
    df_matched = df[df["method"].isin(matched_methods)].copy()
    save_detail_and_summary(
        df_matched, out_dir, "08_same_review_ratio_selective_baselines")

    if xai_snaps:
        xai_path = os.path.join(out_dir, "06_proposed_xai_snapshots.csv")
        pd.DataFrame(xai_snaps).to_csv(xai_path, index=False)
        print(f"  Saved: {xai_path}")

    return df, pd.DataFrame(xai_snaps)


def run_module_ablation(X_train, y_train, m2idx, test_m, rows, y,
                        embedder, xai, cfg, out_dir):
    print("\n[Ablation] Running module ablations at the same monthly review counts...")
    proposed_res, _ = run_proposed(
        X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg,
        np.random.default_rng(cfg.random_state))
    proposed_review_counts = {
        row["month"]: int(row["updated"]) for row in proposed_res
    }

    variants = [
        ("Random+Standard", False, False, False, "random"),
        ("Cluster+Standard", True, False, False, "cluster"),
        ("UncertaintyOnly", False, False, False, "uncertainty"),
        ("ScoreShiftOnly", False, False, False, "score_shift"),
        ("Random+DriftLoss", False, True, True, "random"),
        ("Cluster+DriftScore", True, False, True, "cluster"),
    ]
    records = []
    for method_name, use_cluster, use_drift_loss, use_drift_score, strategy in variants:
        res, _ = common_mod._run_selective_update_core(
            X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg,
            np.random.default_rng(cfg.random_state),
            method_name=method_name,
            use_cluster_selection=use_cluster,
            use_drift_loss=use_drift_loss,
            use_drift_score=use_drift_score,
            selection_strategy=strategy,
            fixed_update_counts=proposed_review_counts,
        )
        records.extend(res)
    records.extend(proposed_res)
    df = to_result_df(records)
    save_detail_and_summary(df, out_dir, "02_module_ablation")
    return df


def run_drift_signal_ablation(X_train, y_train, m2idx, test_m, rows, y,
                              embedder, xai, cfg, out_dir):
    print("\n[Drift] Running drift signal ablations...")
    records = []
    for name, weights in DRIFT_VARIANTS:
        cfg_v = Config(**{k: v for k, v in cfg.__dict__.items()})
        cfg_v.drift_weights = weights
        res, _ = run_proposed(
            X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg_v,
            np.random.default_rng(cfg.random_state))
        for row in res:
            row["method"] = f"DriftSignal-{name}"
            row["drift_weights"] = str(weights)
        records.extend(res)
    df = to_result_df(records)
    save_detail_and_summary(df, out_dir, "03_drift_signal_ablation")
    return df


def run_label_budget_sensitivity(X_train, y_train, m2idx, test_m, rows, y,
                                 embedder, xai, cfg, out_dir):
    print("\n[Budget] Running label budget sensitivity...")
    rows_out = []
    for budget in BUDGETS:
        for method_name, fn in [
            ("Proposed", run_proposed),
            ("RandomSelectiveUpdate", run_ablation_baseline),
            ("UncertaintyOnlyUpdate", run_uncertainty_update),
        ]:
            cfg_b = Config(**{k: v for k, v in cfg.__dict__.items()})
            cfg_b.max_update_frac = budget
            cfg_b.min_update_frac = min(cfg_b.min_update_frac, budget)
            res, _ = fn(
                X_train, y_train, m2idx, test_m, rows, y, embedder, xai, cfg_b,
                np.random.default_rng(cfg.random_state))
            df = to_result_df(res)
            summary = summarize_results(df)
            if summary.empty:
                continue
            item = summary.iloc[0].to_dict()
            item["budget"] = budget
            item["method"] = method_name
            rows_out.append(item)
            print(f"  budget={budget:.2f} {method_name}: "
                  f"F1={item['macro_f1_mean']:.4f} "
                  f"AUC-PR={item['auc_pr_mean']:.4f} "
                  f"FPR@90={item['fpr_at_tpr90_mean']:.4f}")

    df_budget = pd.DataFrame(rows_out)
    path = os.path.join(out_dir, "04_label_budget_sensitivity.csv")
    df_budget.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df_budget


def export_data_health(rows, y, months, cfg: Config, out_dir: str):
    print("\n[Data] Building data health CSVs...")
    balance, monthly_dup, dup_summary = build_data_health(rows, y, months, cfg)
    for df, name in [
        (balance, "05_data_health_monthly_balance.csv"),
        (monthly_dup, "05_data_health_monthly_duplicates.csv"),
        (dup_summary, "05_data_health_duplicate_summary.csv"),
    ]:
        path = os.path.join(out_dir, name)
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")


def _feature_set(value: str) -> set:
    if not isinstance(value, str) or not value:
        return set()
    feats = []
    for part in value.split(";"):
        name = part.strip().split("(")[0].strip()
        if name:
            feats.append(name)
    return set(feats)


def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=np.int64)
    k = min(k, values.size)
    idx = np.argpartition(np.abs(values), -k)[-k:]
    return idx[np.argsort(np.abs(values[idx]))[::-1]]


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    sa = set(_top_k_indices(a, k).tolist())
    sb = set(_top_k_indices(b, k).tolist())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _features_to_text(items: List[Tuple[str, float]]) -> str:
    return "; ".join(f"{name}({score:+.3f})" for name, score in items)


def run_xai_validation_proposed(X_train, y_train, m2idx, test_m, rows, y,
                                embedder, xai, cfg, out_dir):
    """
    Replays the Proposed monthly loop to generate XAI validation CSVs from the
    same temporal protocol.  Full monthly labels are used only for post-hoc
    validation metrics and case-study diagnostics, never for pre-selection
    cluster assignment or drift monitoring.
    """
    print("\n[XAI] Running Proposed replay for XAI validation...")
    cfg_run = Config(**{k: v for k, v in cfg.__dict__.items()})
    model = common_mod._fresh_model(cfg_run)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))

    cluster_mgr = ClusterManager(cfg_run, xai)
    Z_train = embedder.transform(X_train)
    cluster_mgr.fit_initial(Z_train, y_train)

    energy_train = model.decision_function(X_train)
    drift_mon = DriftMonitor(
        energy_train,
        ks_alpha=cfg_run.ks_alpha,
        ks_min_effect=cfg_run.ks_min_effect,
        perf_ema_alpha=cfg_run.perf_ema_alpha,
        burst_threshold=cfg_run.burst_threshold,
        combined_threshold=cfg_run.combined_threshold,
        weights=cfg_run.drift_weights,
        emergency_threshold=cfg_run.emergency_threshold,
    )

    fidelity_rows = []
    case_rows = []
    projection = embedder.svd.components_.T
    rng = np.random.default_rng(cfg_run.random_state)

    for t_idx, mk in enumerate(test_m):
        idx_m = m2idx.get(mk, [])
        if not idx_m:
            continue

        Xt = build_hashed_csr(rows, idx_m, cfg_run.hash_dim)
        yt = y[idx_m]
        y_pred = model.predict(Xt)
        energy_t = model.decision_function(Xt)
        drift_info = drift_mon.update(energy_t)
        drift_flag = drift_info["drift_flag_applied"]
        emergency_flag = drift_info.get("emergency_flag", False)
        actual_frac = (
            min(1.0, cfg_run.max_update_frac * cfg_run.drift_budget_scale)
            if drift_flag or emergency_flag else cfg_run.max_update_frac
        )

        Zt = embedder.transform(Xt)
        member_rows = [rows[i] for i in idx_m]
        assignment, n_new = cluster_mgr.assign(Zt, t_idx, member_rows=member_rows)
        drift_mon.notify_births(n_new)

        analysis_month = (t_idx % 6 == 0) or n_new > 0 or drift_info["drift_detected"]
        month_records = []
        if analysis_month:
            for cid in sorted(set(int(c) for c in assignment if c >= 0)):
                if cid >= len(cluster_mgr.centroids):
                    continue
                mask = assignment == cid
                n_members = int(mask.sum())
                if n_members < 3:
                    continue
                centroid_hash_space = projection @ cluster_mgr.centroids[cid]
                member_mean = np.asarray(Xt[mask].mean(axis=0)).ravel()
                denom = (
                    np.linalg.norm(centroid_hash_space) *
                    np.linalg.norm(member_mean)
                )
                cos_sim = float(
                    np.dot(centroid_hash_space, member_mean) / denom
                ) if denom > 0 else float("nan")
                overlap = _topk_overlap(
                    centroid_hash_space, member_mean, cfg_run.xai_top_k)
                mal_ratio = float(yt[mask].mean()) if n_members else float("nan")
                err_rate = float((y_pred[mask] != yt[mask]).mean()) if n_members else float("nan")
                top_features = _features_to_text(cluster_mgr.top_features[cid])
                record = {
                    "month": mk,
                    "cluster_id": cid,
                    "n_members_current_month": n_members,
                    "cluster_birth_month_index": int(cluster_mgr.birth_month[cid]),
                    "cluster_age": int(cluster_mgr.age[cid]),
                    "is_new": bool(cluster_mgr.is_new[cid]),
                    "drift_detected": bool(drift_info["drift_detected"]),
                    "drift_applied": bool(drift_flag),
                    "n_new_clusters_month": int(n_new),
                    "cos_sim_backproj_vs_member_mean": cos_sim,
                    "topk_bucket_overlap_vs_member_mean": float(overlap),
                    "malicious_rate_current_month": mal_ratio,
                    "cluster_error_rate_current_month": err_rate,
                    "top_features": top_features,
                    "analysis_only_full_labels": True,
                }
                fidelity_rows.append(record)
                month_records.append(record)

        if month_records and (n_new > 0 or drift_info["drift_detected"]):
            event = "new_cluster" if n_new > 0 else "drift_detected"
            for record in sorted(
                month_records,
                key=lambda r: (
                    r["is_new"],
                    r["cluster_error_rate_current_month"],
                    r["n_members_current_month"],
                ),
                reverse=True,
            )[:10]:
                case_record = dict(record)
                case_record["case_event"] = event
                case_rows.append(case_record)

        cluster_mgr.prune(t_idx)
        X_sel, y_sel, sel_w, _ = cluster_mgr.select_update_samples(
            Xt, yt, assignment, drift_flag, actual_frac, rng)
        common_mod._apply_update(
            model, X_sel, y_sel, sel_w, drift_info["drift_score"], cfg_run)

    fidelity_df = pd.DataFrame(fidelity_rows)
    case_df = pd.DataFrame(case_rows)
    fidelity_path = os.path.join(out_dir, "06_xai_backprojection_fidelity.csv")
    case_path = os.path.join(out_dir, "06_xai_case_study_candidates.csv")
    fidelity_df.to_csv(fidelity_path, index=False)
    case_df.to_csv(case_path, index=False)
    print(f"  Saved: {fidelity_path}")
    print(f"  Saved: {case_path}")
    return fidelity_df, case_df


def export_xai_validation(out_dir: str, xai_df: pd.DataFrame,
                          fidelity_df: pd.DataFrame, case_df: pd.DataFrame):
    print("\n[XAI] Exporting XAI validation summaries...")
    stability_records = []
    if not xai_df.empty and {"month", "cluster_id", "top_features"}.issubset(xai_df.columns):
        for cid, sub in xai_df.sort_values("month").groupby("cluster_id"):
            prev = None
            for _, row in sub.iterrows():
                curr = _feature_set(row.get("top_features", ""))
                if prev is not None and (curr or prev):
                    denom = len(curr | prev)
                    stability_records.append({
                        "cluster_id": cid,
                        "month": row["month"],
                        "top_feature_jaccard_vs_prev": len(curr & prev) / denom if denom else 0.0,
                        "n_features": len(curr),
                    })
                prev = curr
    df_stab = pd.DataFrame(stability_records)
    stab_path = os.path.join(out_dir, "06_xai_top_feature_stability.csv")
    df_stab.to_csv(stab_path, index=False)
    print(f"  Saved: {stab_path}")

    summary_rows = []
    if not df_stab.empty:
        summary_rows.append({
            "metric": "top_feature_stability_jaccard",
            "mean": float(df_stab["top_feature_jaccard_vs_prev"].mean()),
            "std": float(df_stab["top_feature_jaccard_vs_prev"].std()),
            "n": int(len(df_stab)),
        })

    for metric, col in [
        ("backprojection_cosine_to_member_mean", "cos_sim_backproj_vs_member_mean"),
        ("topk_bucket_overlap_to_member_mean", "topk_bucket_overlap_vs_member_mean"),
        ("xai_case_study_candidate_count", None),
    ]:
        if col is None:
            summary_rows.append({
                "metric": metric,
                "mean": float(len(case_df)),
                "std": 0.0,
                "n": int(len(case_df)),
            })
            continue
        if not fidelity_df.empty and col in fidelity_df.columns:
            vals = pd.to_numeric(fidelity_df[col], errors="coerce").dropna()
            if len(vals):
                summary_rows.append({
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "n": int(len(vals)),
                })

    summary_path = os.path.join(out_dir, "06_xai_validation_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")


def export_runtime_overhead(main_df: pd.DataFrame, out_dir: str):
    print("\n[Runtime] Exporting runtime/update overhead CSV...")
    rows = []
    for method, sub in main_df.groupby("method", sort=False):
        rows.append({
            "method": method,
            "months": int(sub["month"].nunique()) if "month" in sub.columns else len(sub),
            "mean_time_sec_per_month": float(sub["time_sec"].mean())
                                       if "time_sec" in sub.columns else float("nan"),
            "total_time_sec": float(sub["time_sec"].sum())
                              if "time_sec" in sub.columns else float("nan"),
            "mean_updated": float(sub["updated"].mean())
                            if "updated" in sub.columns else float("nan"),
            "label_update_ratio": float(sub["updated"].sum() / sub["total"].sum())
                                  if {"updated", "total"}.issubset(sub.columns) else float("nan"),
        })
    path = os.path.join(out_dir, "07_runtime_update_overhead.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")


def main(only: str = "all"):
    cfg = Config()
    out_dir = proposed_dir(cfg)

    with ExperimentTimer("proposed_performance_suite", cfg.out_dir):
        print("\n[1] Loading dataset...")
        rows, y, months = load_dataset(cfg)

        print("\n[2] Temporal split...")
        m2idx, ordered, train_m, val_m, test_m, train_idx, _ = make_temporal_split(months, cfg)

        print("\n[3] Building train features and semantic components...")
        X_train = build_hashed_csr(rows, train_idx, cfg.hash_dim)
        y_train = y[train_idx]
        embedder, xai = build_components(X_train, train_idx, rows, cfg)

        print("\n[4] Saving protocol summary...")
        export_protocol_summary(cfg, train_m, val_m, test_m, out_dir)

        print("\n[5] Loading monthly validation-only hyperparameters...")
        cfg, tuned_path = load_validation_tuned_params(cfg, granularity="monthly")
        if tuned_path is None or "hyperparameters" not in tuned_path:
            raise RuntimeError(
                "Run artifact/code/validation_hyperparameter_tuning.py --granularity monthly "
                "before proposed_performance_suite.py."
            )
        save_loaded_validation_params(cfg, tuned_path, out_dir)

        if only == "module_ablation":
            run_module_ablation(
                X_train, y_train, m2idx, test_m, rows, y,
                embedder, xai, cfg, out_dir)
            return

        main_df, xai_df = run_main_performance(
            X_train, y_train, m2idx, ordered, train_m, test_m,
            rows, y, embedder, xai, cfg, out_dir)
        run_module_ablation(
            X_train, y_train, m2idx, test_m, rows, y,
            embedder, xai, cfg, out_dir)
        run_drift_signal_ablation(
            X_train, y_train, m2idx, test_m, rows, y,
            embedder, xai, cfg, out_dir)
        run_label_budget_sensitivity(
            X_train, y_train, m2idx, test_m, rows, y,
            embedder, xai, cfg, out_dir)
        export_data_health(rows, y, months, cfg, out_dir)
        fidelity_df, case_df = run_xai_validation_proposed(
            X_train, y_train, m2idx, test_m, rows, y,
            embedder, xai, cfg, out_dir)
        export_xai_validation(out_dir, xai_df, fidelity_df, case_df)
        export_runtime_overhead(main_df, out_dir)

        manifest = pd.DataFrame({
            "csv": sorted(
                f for f in os.listdir(out_dir)
                if f.endswith(".csv")
            )
        })
        manifest_path = os.path.join(out_dir, "manifest.csv")
        manifest.to_csv(manifest_path, index=False)
        print(f"\n  Saved: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=("all", "module_ablation"),
        default="all",
        help="Run the full suite or only regenerate the module ablation CSVs.",
    )
    main(parser.parse_args().only)
