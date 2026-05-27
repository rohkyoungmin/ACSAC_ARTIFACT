#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export paper-facing baseline tables split by monthly/yearly granularity.

Outputs:
  results/performance/baseline_tables/monthly_baseline_comparison.csv
  results/performance/baseline_tables/monthly_baseline_comparison_summary.csv
  results/performance/baseline_tables/yearly_baseline_comparison.csv
  results/performance/baseline_tables/yearly_baseline_comparison_summary.csv
"""

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from common import Config  # noqa: E402


OUT_SUBDIR = "baseline_tables"


def _read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  [SKIP] missing: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source_file"] = path
    print(f"  Loaded: {path} ({len(df)} rows)")
    return df


def _first_existing(row: pd.Series, names: List[str], default=np.nan):
    for name in names:
        if name in row.index:
            return row[name]
    return default


def normalize_detail(
    df: pd.DataFrame,
    granularity: str,
    dataset_setting: str,
    table_group: str,
    display_from: str = "method",
    method_from: str = "method",
    inspired_by_default: str = "",
) -> pd.DataFrame:
    rows: List[Dict] = []
    if df.empty:
        return pd.DataFrame()

    for _, row in df.iterrows():
        method = str(_first_existing(row, [method_from, "method", "selection_variant"], ""))
        display = str(_first_existing(row, [display_from, method_from, "method"], method))
        period = str(_first_existing(row, ["month", "year"], ""))
        total = _first_existing(row, ["total"], np.nan)
        updated = _first_existing(row, ["updated"], np.nan)
        try:
            review_ratio = float(updated) / float(total) if float(total) > 0 else np.nan
        except Exception:
            review_ratio = _first_existing(row, ["review_ratio", "label_update_ratio"], np.nan)

        rows.append({
            "granularity": granularity,
            "dataset_setting": dataset_setting,
            "table_group": table_group,
            "period": period,
            "display_name": display,
            "method": method,
            "selection_variant": _first_existing(row, ["selection_variant"], display),
            "inspired_by": _first_existing(row, ["inspired_by"], inspired_by_default),
            "uses_cluster": _first_existing(row, ["uses_cluster"], np.nan),
            "total": total,
            "updated": updated,
            "review_ratio": review_ratio,
            "f1": _first_existing(row, ["f1", "macro_f1_mean"], np.nan),
            "tpr": _first_existing(row, ["tpr"], np.nan),
            "fpr": _first_existing(row, ["fpr"], np.nan),
            "fpr_at_tpr90": _first_existing(row, ["fpr_at_tpr90", "fpr_at_tpr90_mean"], np.nan),
            "auc_roc": _first_existing(row, ["auc_roc", "auc_roc_mean"], np.nan),
            "auc_pr": _first_existing(row, ["auc_pr", "auc_pr_mean"], np.nan),
            "mal_rate_selected": _first_existing(row, ["mal_rate_selected"], np.nan),
            "time_sec": _first_existing(row, ["time_sec"], np.nan),
            "source_file": _first_existing(row, ["source_file"], ""),
        })
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        "granularity", "dataset_setting", "table_group", "display_name",
        "method", "selection_variant", "inspired_by", "uses_cluster",
    ]
    for keys, sub in detail.groupby(group_cols, dropna=False, sort=False):
        item = dict(zip(group_cols, keys))
        total_sum = pd.to_numeric(sub["total"], errors="coerce").sum()
        updated_sum = pd.to_numeric(sub["updated"], errors="coerce").sum()
        item.update({
            "periods": int(sub["period"].nunique()),
            "total": float(total_sum),
            "updated": float(updated_sum),
            "review_ratio": float(updated_sum / total_sum) if total_sum > 0 else np.nan,
            "f1_mean": float(pd.to_numeric(sub["f1"], errors="coerce").mean()),
            "f1_std": float(pd.to_numeric(sub["f1"], errors="coerce").std()),
            "auc_pr_mean": float(pd.to_numeric(sub["auc_pr"], errors="coerce").mean()),
            "auc_roc_mean": float(pd.to_numeric(sub["auc_roc"], errors="coerce").mean()),
            "fpr_at_tpr90_mean": float(pd.to_numeric(sub["fpr_at_tpr90"], errors="coerce").mean()),
            "mal_rate_selected_mean": float(pd.to_numeric(sub["mal_rate_selected"], errors="coerce").mean()),
            "time_sec_total": float(pd.to_numeric(sub["time_sec"], errors="coerce").sum()),
        })
        rows.append(item)
    return pd.DataFrame(rows)


def build_monthly() -> pd.DataFrame:
    parts = []

    main_path = "results/proposed_performance/01_main_performance.csv"
    main = _read(main_path)
    if not main.empty:
        parts.append(normalize_detail(
            main,
            granularity="monthly",
            dataset_setting="extended_features_natural",
            table_group="existing_main_baselines",
            display_from="method",
        ))

        proposed = main[main["method"] == "Proposed"].copy()
        if not proposed.empty:
            proposed["selection_variant"] = "Cluster-priority (ours)"
            proposed["inspired_by"] = "this work"
            proposed["uses_cluster"] = True
            parts.append(normalize_detail(
                proposed,
                granularity="monthly",
                dataset_setting="extended_features_natural",
                table_group="selection_criterion_33pct",
                display_from="selection_variant",
                inspired_by_default="this work",
            ))

    sel_path = "results/performance/monthly/selection_criterion_ablation.csv"
    sel = _read(sel_path)
    if not sel.empty:
        parts.append(normalize_detail(
            sel,
            granularity="monthly",
            dataset_setting="extended_features_natural",
            table_group="selection_criterion_33pct",
            display_from="selection_variant",
            method_from="selection_variant",
        ))

    noncluster_path = "results/performance/monthly/noncluster_selective_baselines_proposed_ratio.csv"
    noncluster = _read(noncluster_path)
    if not noncluster.empty:
        parts.append(normalize_detail(
            noncluster,
            granularity="monthly",
            dataset_setting="extended_features_natural",
            table_group="noncluster_selective_33pct",
            display_from="method",
        ))

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_yearly() -> pd.DataFrame:
    configs = [
        (
            "results/performance/yearly/yearly_results.csv",
            "androzoo_year_balanced",
            "existing_yearly_baselines",
        ),
        (
            "results/performance/yearly/yearly_imbalanced_results.csv",
            "androzoo_year_imbalanced_9to1",
            "existing_yearly_baselines",
        ),
        (
            "results/performance/yearly/yearly_imbalanced_ablation_results.csv",
            "androzoo_year_imbalanced_9to1",
            "yearly_ablation_baselines",
        ),
    ]
    parts = []
    for path, dataset_setting, table_group in configs:
        df = _read(path)
        if df.empty:
            continue
        parts.append(normalize_detail(
            df,
            granularity="yearly",
            dataset_setting=dataset_setting,
            table_group=table_group,
            display_from="method",
        ))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    cfg = Config()
    out_dir = os.path.join(cfg.perf_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print("\n[Monthly] Building merged monthly baseline CSV...")
    monthly = build_monthly()
    monthly_path = os.path.join(out_dir, "monthly_baseline_comparison.csv")
    monthly_summary_path = os.path.join(out_dir, "monthly_baseline_comparison_summary.csv")
    monthly.to_csv(monthly_path, index=False)
    summarize(monthly).to_csv(monthly_summary_path, index=False)
    print(f"  Saved: {monthly_path}")
    print(f"  Saved: {monthly_summary_path}")

    print("\n[Yearly] Building merged yearly baseline CSV...")
    yearly = build_yearly()
    yearly_path = os.path.join(out_dir, "yearly_baseline_comparison.csv")
    yearly_summary_path = os.path.join(out_dir, "yearly_baseline_comparison_summary.csv")
    yearly.to_csv(yearly_path, index=False)
    summarize(yearly).to_csv(yearly_summary_path, index=False)
    print(f"  Saved: {yearly_path}")
    print(f"  Saved: {yearly_summary_path}")

    print("\n[Done]")
    print(f"  Monthly rows: {len(monthly)}")
    print(f"  Yearly rows : {len(yearly)}")


if __name__ == "__main__":
    main()
