#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation-only hyperparameter tuning for the ACSAC artifact.

The paper experiments must not tune on test months/years. This script selects
the Proposed-method hyperparameters using only the validation split, then writes
granularity-specific JSON files consumed by the other experiment scripts:

  results/hyperparameters/monthly_validation_tuned_params.json
  results/hyperparameters/yearly_validation_tuned_params.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from common import *  # noqa: F401,F403
from common import _run_selective_update_core


PARAM_STAGES: List[Tuple[str, List[str]]] = [
    ("paper_tuned_parameters", [
        "ks_min_effect",
        "combined_threshold",
        "drift_lambda",
        "new_cluster_dist",
    ]),
]


DEFAULT_GRID: Dict[str, List] = {
    "hash_dim": [2 ** 16, 2 ** 18],
    "svd_components": [32, 64],
    "n_clusters_init": [10, 20, 40],
    "cluster_ttl": [2, 3, 6],
    "new_cluster_dist": [0.25, 0.35, 0.45],
    "ks_min_effect": [0.05, 0.10, 0.15],
    "perf_ema_alpha": [0.20, 0.30, 0.40],
    "burst_threshold": [2.0, 3.0, 4.0],
    "drift_weights": [
        (0.40, 0.30, 0.30),
        (0.50, 0.25, 0.25),
        (0.33, 0.33, 0.34),
    ],
    "combined_threshold": [0.25, 0.35, 0.45],
    "max_update_frac": [0.30, 0.3327381546720546],
    "drift_budget_scale": [1.0, 1.5, 2.0],
    "cluster_priority_new": [1.5, 2.0, 2.5],
    "cluster_priority_drift": [1.0, 1.5, 2.0],
    "cluster_priority_stable": [0.25, 0.50, 0.75],
    "drift_lambda": [0.5, 1.0, 2.0],
    "classifier_alpha": [1e-5, 1e-4],
    "xai_top_k": [10],
}


def _clone_cfg(cfg: Config) -> Config:
    return Config(**{k: v for k, v in cfg.__dict__.items()})


def _json_value(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _candidate_key(value) -> str:
    return json.dumps(_json_value(value), sort_keys=True)


def _unique_candidates(current_value, candidates: List) -> List:
    values = [current_value] + list(candidates)
    seen, out = set(), []
    for value in values:
        key = _candidate_key(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _set_param(cfg: Config, name: str, value) -> None:
    if name == "drift_weights":
        value = tuple(float(v) for v in value)
    setattr(cfg, name, value)


def _build_components(rows, train_idx, X_train, cfg: Config):
    b2n = build_feature_index(rows, train_idx, cfg.hash_dim)
    global_mean = np.asarray(X_train.mean(axis=0)).flatten()
    embedder = SemanticEmbedder(cfg.svd_components, cfg.random_state)
    embedder.fit(X_train)
    xai = ClusterXAI(global_mean, b2n, cfg.xai_top_k, cfg.xai_min_delta)
    xai.set_embedder(embedder)
    return embedder, xai


def _get_eval_components(data: Dict, cfg: Config, cache: Dict):
    key = (cfg.hash_dim, cfg.svd_components, cfg.xai_top_k, cfg.xai_min_delta)
    if key not in cache:
        cache.clear()
        X_train = build_hashed_csr(data["rows"], data["train_idx"], cfg.hash_dim)
        y_train = data["y"][data["train_idx"]]
        embedder, xai = _build_components(data["rows"], data["train_idx"], X_train, cfg)
        cache[key] = (X_train, y_train, embedder, xai)
    return cache[key]


def _summarize_validation(results: List[Dict], elapsed_sec: float) -> Dict:
    if not results:
        return {
            "val_auc_pr_mean": float("nan"),
            "val_macro_f1_mean": float("nan"),
            "val_fpr_at_tpr90_mean": float("nan"),
            "val_auc_roc_mean": float("nan"),
            "val_label_update_ratio": float("nan"),
            "val_periods": 0,
            "runtime_sec": round(elapsed_sec, 4),
        }
    df = pd.DataFrame(results)
    ratio = float(df["updated"].sum() / df["total"].sum()) \
        if {"updated", "total"}.issubset(df.columns) and df["total"].sum() > 0 else float("nan")
    return {
        "val_auc_pr_mean": float(df["auc_pr"].mean()),
        "val_macro_f1_mean": float(df["f1"].mean()),
        "val_fpr_at_tpr90_mean": float(df["fpr_at_tpr90"].mean()),
        "val_auc_roc_mean": float(df["auc_roc"].mean()),
        "val_label_update_ratio": ratio,
        "val_periods": int(df["month"].nunique()) if "month" in df else len(df),
        "runtime_sec": round(elapsed_sec, 4),
    }


def _rank(summary: Dict) -> Tuple[float, float, float]:
    auc_pr = summary.get("val_auc_pr_mean", float("nan"))
    f1 = summary.get("val_macro_f1_mean", float("nan"))
    fpr90 = summary.get("val_fpr_at_tpr90_mean", float("nan"))
    auc_pr = auc_pr if np.isfinite(auc_pr) else -1.0
    f1 = f1 if np.isfinite(f1) else -1.0
    fpr_component = -fpr90 if np.isfinite(fpr90) else -1e9
    return (float(auc_pr), float(f1), float(fpr_component))


def load_monthly_validation_data(cfg: Config) -> Dict:
    rows, y, months = load_dataset(cfg)
    m2idx, ordered, train_m, val_m, test_m, train_idx, val_idx = make_temporal_split(months, cfg)
    return {
        "granularity": "monthly",
        "rows": rows,
        "y": y,
        "period_to_idx": m2idx,
        "ordered_periods": ordered,
        "train_periods": train_m,
        "val_periods": val_m,
        "test_periods": test_m,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def load_yearly_validation_data(cfg: Config, gw_mw_ratio: float) -> Dict:
    rows, y, years = load_androzoo_year_imbalanced(cfg, gw_mw_ratio=gw_mw_ratio)
    y2idx, train_idx, val_idx, test_years = make_yearly_split(years, cfg)
    val_year = cfg.val_year + 1
    return {
        "granularity": "yearly",
        "rows": rows,
        "y": y,
        "period_to_idx": {str(yr): idxs for yr, idxs in y2idx.items()},
        "ordered_periods": [str(yr) for yr in sorted(y2idx.keys())],
        "train_periods": [str(yr) for yr in sorted(y2idx.keys()) if yr <= cfg.val_year],
        "val_periods": [str(val_year)],
        "test_periods": [str(yr) for yr in test_years],
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def score_validation(data: Dict, cfg: Config, cache: Dict) -> Dict:
    X_train, y_train, embedder, xai = _get_eval_components(data, cfg, cache)
    val_map = {p: data["period_to_idx"][p] for p in data["val_periods"] if p in data["period_to_idx"]}
    if not val_map:
        raise RuntimeError(f"No validation periods available for {data['granularity']}")
    t0 = time.perf_counter()
    results, _ = _run_selective_update_core(
        X_train, y_train, val_map, list(val_map.keys()),
        data["rows"], data["y"], embedder, xai, cfg,
        rng=np.random.default_rng(cfg.random_state),
        method_name="ProposedValidation",
        use_cluster_selection=True,
        use_drift_loss=True,
        use_drift_score=True,
    )
    return _summarize_validation(results, time.perf_counter() - t0)


def staged_validation_tune(granularity: str, data: Dict, cfg: Config) -> Tuple[Config, pd.DataFrame]:
    current = _clone_cfg(cfg)
    cache: Dict = {}
    trace_rows: List[Dict] = []
    evaluation_id = 0

    print(f"\n[Tuning:{granularity}] staged validation grid search")
    print(f"  Validation periods: {', '.join(data['val_periods'])}")
    print("  Selection metric: mean validation AUC-PR, then macro-F1, then lower FPR@TPR=90%")

    for stage, params in PARAM_STAGES:
        print(f"\n  [Stage] {stage}")
        for param in params:
            candidates = _unique_candidates(getattr(current, param), DEFAULT_GRID[param])
            best_rank = None
            best_value = getattr(current, param)
            row_ids = []

            for value in candidates:
                evaluation_id += 1
                trial = _clone_cfg(current)
                _set_param(trial, param, value)
                summary = score_validation(data, trial, cache)
                rank = _rank(summary)
                selected_now = best_rank is None or rank > best_rank
                if selected_now:
                    best_rank = rank
                    best_value = value

                row = {
                    "evaluation_id": evaluation_id,
                    "granularity": granularity,
                    "stage": stage,
                    "parameter": param,
                    "candidate_value": _candidate_key(value),
                    "selected_for_parameter": False,
                    "active_params_json": json.dumps(
                        {name: _json_value(getattr(trial, name)) for name in VALIDATION_TUNED_PARAM_NAMES},
                        sort_keys=True,
                    ),
                }
                row.update(summary)
                trace_rows.append(row)
                row_ids.append(len(trace_rows) - 1)
                print(
                    f"    {param}={_candidate_key(value)} "
                    f"AUC-PR={summary['val_auc_pr_mean']:.4f} "
                    f"F1={summary['val_macro_f1_mean']:.4f}"
                )

            _set_param(current, param, best_value)
            for row_id in row_ids:
                if trace_rows[row_id]["candidate_value"] == _candidate_key(best_value):
                    trace_rows[row_id]["selected_for_parameter"] = True
            print(f"    -> selected {param}={_candidate_key(best_value)}")

    return current, pd.DataFrame(trace_rows)


def selected_param_record(cfg: Config, granularity: str, trace: pd.DataFrame, data: Dict) -> Dict:
    record = {
        "source": "validation_staged_grid_search",
        "granularity": granularity,
        "selection_metric": "mean_validation_auc_pr_then_macro_f1_then_lower_fpr_at_tpr90",
        "search_strategy": "validation_grid_over_paper_tuned_parameters",
        "fixed_before_validation": {
            name: _json_value(getattr(cfg, name))
            for name in VALIDATION_FIXED_PARAM_NAMES
        },
        "fixed_parameter_justification": (
            "Fixed parameters are shared representation, protocol, reporting, "
            "or base-classifier settings set before validation. Validation "
            "selection is limited to the four paper-tuned Proposed parameters."
        ),
        "validation_selected_parameters": list(VALIDATION_SELECTED_PARAM_NAMES),
        "train_periods": data["train_periods"],
        "validation_periods": data["val_periods"],
        "test_periods_excluded": data["test_periods"],
        "num_evaluations": int(len(trace)),
    }
    for name in VALIDATION_TUNED_PARAM_NAMES:
        record[name] = _json_value(getattr(cfg, name))
    return record


def write_outputs(record: Dict, trace: pd.DataFrame, out_dir: str, archive_dir: str = "") -> None:
    dirs = [out_dir]
    if archive_dir:
        dirs.append(archive_dir)

    for target_dir in dirs:
        os.makedirs(target_dir, exist_ok=True)
        stem = f"{record['granularity']}_validation"
        json_path = os.path.join(target_dir, f"{stem}_tuned_params.json")
        csv_path = os.path.join(target_dir, f"{stem}_tuned_params.csv")
        trace_path = os.path.join(target_dir, f"{stem}_tuning_trace.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        pd.DataFrame([record]).to_csv(csv_path, index=False)
        trace.to_csv(trace_path, index=False)
        print(f"  Saved: {json_path}")
        print(f"  Saved: {csv_path}")
        print(f"  Saved: {trace_path}")


def run_one(granularity: str, args) -> None:
    cfg = Config(out_dir=args.out_dir)
    hyper_dir = os.path.join(cfg.out_dir, "hyperparameters")
    archive_dir = "" if args.no_archive_copy else args.archive_dir

    if granularity == "monthly":
        data = load_monthly_validation_data(cfg)
    elif granularity == "yearly":
        data = load_yearly_validation_data(cfg, gw_mw_ratio=args.gw_mw_ratio)
    else:
        raise ValueError(granularity)

    tuned_cfg, trace = staged_validation_tune(granularity, data, cfg)
    record = selected_param_record(tuned_cfg, granularity, trace, data)
    write_outputs(record, trace, hyper_dir, archive_dir=archive_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--granularity",
        choices=["monthly", "yearly", "both"],
        default="both",
        help="Which validation split to tune.",
    )
    parser.add_argument("--out-dir", default="./results", help="Fresh rerun output root.")
    parser.add_argument(
        "--archive-dir",
        default="./artifact/results/full_csv/hyperparameters",
        help="Optional artifact CSV/JSON mirror directory.",
    )
    parser.add_argument("--no-archive-copy", action="store_true")
    parser.add_argument("--gw-mw-ratio", type=float, default=9.0)
    args = parser.parse_args()

    targets = ["monthly", "yearly"] if args.granularity == "both" else [args.granularity]
    t0 = time.perf_counter()
    for granularity in targets:
        run_one(granularity, args)
    os.makedirs(args.out_dir, exist_ok=True)
    timing_path = os.path.join(args.out_dir, "timing_summary.csv")
    row = pd.DataFrame([{
        "exp_name": "validation_hyperparameter_tuning",
        "wall_time_sec": round(time.perf_counter() - t0, 4),
        "peak_mem_mb": float("nan"),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }])
    row.to_csv(timing_path, mode="a", header=not os.path.exists(timing_path), index=False)


if __name__ == "__main__":
    main()
