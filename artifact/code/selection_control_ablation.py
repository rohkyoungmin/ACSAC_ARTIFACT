#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-24: Selection criterion ablation at the same review budget.

All variants use:
  - monthly extended-features dataset;
  - strict predict-before-update protocol;
  - same review budget ratio, defaulting to Proposed's overall label ratio;
  - standard online update after selected labels are reviewed.

Selection variants:
  Random selection               naive baseline
  Uncertainty-based selection    classic active learning
  Distance-to-centroid selection CADE-style latent distance score

The output markdown also appends the existing Proposed reference row as
"Cluster-priority (ours)".

Outputs:
  results/performance/monthly/selection_criterion_ablation.csv
  results/performance/monthly/selection_criterion_ablation_summary.csv
  documents/SELECTION_CRITERION_ABLATION.md
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, ".")
from common import (  # noqa: E402
    Config,
    ExperimentTimer,
    SemanticEmbedder,
    build_hashed_csr,
    compute_label_efficiency,
    compute_loss_profile,
    compute_metrics,
    load_dataset,
    make_temporal_split,
    to_result_df,
    _fresh_model,
)


DEFAULT_REVIEW_RATIO = 0.3327381546720546

VARIANT_META = {
    "Random selection": "naive baseline",
    "Uncertainty-based selection": "classic AL [Chen'23-style]",
    "Distance-to-centroid selection": "CADE-style drift score [10]",
    "Cluster-priority (ours)": "this work",
}


def _fmt(x, digits=3) -> str:
    if pd.isna(x):
        return "nan"
    return f"{float(x):.{digits}f}"


def select_random(n: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n, size=min(budget, n), replace=False) if budget > 0 else np.array([], dtype=np.int64)


def select_uncertainty(y_prob: np.ndarray, budget: int) -> np.ndarray:
    uncertainty = 1.0 - 2.0 * np.abs(y_prob - 0.5)
    return np.argsort(uncertainty)[::-1][:min(budget, len(y_prob))]


def fit_centroid_state(embedder: SemanticEmbedder, X_train: csr_matrix, y_train: np.ndarray) -> Dict:
    z_train = embedder.transform(X_train)
    return {
        "z_bank": z_train,
        "y_bank": y_train.copy(),
    }


def centroid_scores(embedder: SemanticEmbedder, state: Dict, X: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    z = embedder.transform(X)
    labels = np.array(sorted(np.unique(state["y_bank"])))
    centroids = []
    for label in labels:
        centroids.append(state["z_bank"][state["y_bank"] == label].mean(axis=0))
    C = np.vstack(centroids)
    dist = np.vstack([np.linalg.norm(z - c, axis=1) for c in C]).T
    min_dist = dist.min(axis=1)
    closest = labels[dist.argmin(axis=1)]
    return min_dist, closest


def select_centroid_distance(scores: np.ndarray, budget: int) -> np.ndarray:
    return np.argsort(scores)[::-1][:min(budget, len(scores))]


def update_centroid_state(embedder: SemanticEmbedder, state: Dict, X_sel: csr_matrix, y_sel: np.ndarray) -> None:
    if len(y_sel) == 0:
        return
    state["z_bank"] = np.vstack([state["z_bank"], embedder.transform(X_sel)])
    state["y_bank"] = np.concatenate([state["y_bank"], y_sel])


def fit_conformal_state(model, X_train: csr_matrix, y_train: np.ndarray) -> Dict:
    scores = model.decision_function(X_train)
    return {
        "ncm_1": -scores[y_train == 1],
        "ncm_0": scores[y_train == 0],
    }


def conformal_credibility(state: Dict, scores: np.ndarray) -> np.ndarray:
    ncm_test_1 = -scores
    ncm_test_0 = scores
    p1 = np.array([(state["ncm_1"] >= v).mean() for v in ncm_test_1])
    p0 = np.array([(state["ncm_0"] >= v).mean() for v in ncm_test_0])
    return np.maximum(p1, p0)


def select_conformal_rejection(credibility: np.ndarray, budget: int) -> np.ndarray:
    return np.argsort(credibility)[:min(budget, len(credibility))]


def update_conformal_state(state: Dict, scores_sel: np.ndarray, y_sel: np.ndarray) -> None:
    if len(y_sel) == 0:
        return
    if np.any(y_sel == 1):
        state["ncm_1"] = np.concatenate([state["ncm_1"], -scores_sel[y_sel == 1]])
    if np.any(y_sel == 0):
        state["ncm_0"] = np.concatenate([state["ncm_0"], scores_sel[y_sel == 0]])


def run_variant(
    variant: str,
    X_train: csr_matrix,
    y_train: np.ndarray,
    month_to_idx: Dict[str, List[int]],
    test_months: List[str],
    rows: List[Dict],
    y: np.ndarray,
    cfg: Config,
    review_ratio: float,
    rng: np.random.Generator,
    embedder: SemanticEmbedder = None,
) -> List[Dict]:
    model = _fresh_model(cfg)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))

    centroid_state = fit_centroid_state(embedder, X_train, y_train) \
        if variant == "Distance-to-centroid selection" else None
    conformal_state = fit_conformal_state(model, X_train, y_train) \
        if variant == "Conformal-rejection-priority" else None

    results = []
    for month in test_months:
        idx = month_to_idx.get(month, [])
        if not idx:
            continue

        t0 = time.perf_counter()
        Xt = build_hashed_csr(rows, idx, cfg.hash_dim)
        yt = y[idx]
        budget = min(int(len(yt) * review_ratio), len(yt))

        y_prob = model.predict_proba(Xt)[:, 1]
        y_pred = (y_prob >= 0.5).astype(np.int64)
        metrics = compute_metrics(yt, y_pred, y_prob)
        loss_before = compute_loss_profile(model, Xt, yt)["loss_mean"]
        scores = model.decision_function(Xt)

        aux = {}
        if variant == "Random selection":
            chosen = select_random(len(yt), budget, rng)
        elif variant == "Uncertainty-based selection":
            chosen = select_uncertainty(y_prob, budget)
        elif variant == "Distance-to-centroid selection":
            dist, closest = centroid_scores(embedder, centroid_state, Xt)
            chosen = select_centroid_distance(dist, budget)
            aux["selection_score_mean"] = float(np.mean(dist[chosen])) if len(chosen) else float("nan")
            aux["closest_malware_rate"] = float(np.mean(closest == 1))
        elif variant == "Conformal-rejection-priority":
            credibility = conformal_credibility(conformal_state, scores)
            chosen = select_conformal_rejection(credibility, budget)
            aux["selection_score_mean"] = float(np.mean(1.0 - credibility[chosen])) if len(chosen) else float("nan")
        else:
            raise ValueError(f"Unknown variant: {variant}")

        X_sel = Xt[chosen]
        y_sel = yt[chosen]
        if len(y_sel):
            model.partial_fit(X_sel, y_sel)
            if centroid_state is not None:
                update_centroid_state(embedder, centroid_state, X_sel, y_sel)
            if conformal_state is not None:
                update_conformal_state(conformal_state, model.decision_function(X_sel), y_sel)

        loss_after = compute_loss_profile(model, Xt, yt)["loss_mean"]
        row = {
            "month": month,
            "method": variant,
            "selection_variant": variant,
            "inspired_by": VARIANT_META[variant],
            "total": int(len(yt)),
            "updated": int(len(chosen)),
            "review_ratio_target": float(review_ratio),
            "actual_frac": float(len(chosen) / len(yt)) if len(yt) else 0.0,
            "uses_cluster": False,
            "loss_mode": "standard",
            "loss_before": float(loss_before),
            "loss_after": float(loss_after),
            "loss_delta": float(loss_after - loss_before),
            "mal_rate_selected": compute_label_efficiency(y_sel),
            "time_sec": round(time.perf_counter() - t0, 4),
        }
        row.update(aux)
        row.update(metrics)
        results.append(row)
    return results


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in detail.groupby("method", sort=False):
        rows.append({
            "selection_variant": method,
            "inspired_by": VARIANT_META[method],
            "uses_cluster": bool(sub["uses_cluster"].iloc[0]),
            "review_ratio": float(sub["updated"].sum() / sub["total"].sum()),
            "months": int(sub["month"].nunique()),
            "f1": float(sub["f1"].mean()),
            "f1_std": float(sub["f1"].std()),
            "auc_pr": float(sub["auc_pr"].mean()),
            "auc_roc": float(sub["auc_roc"].mean()),
            "fpr_at_tpr90": float(sub["fpr_at_tpr90"].mean()),
            "mal_rate_selected": float(sub["mal_rate_selected"].mean()),
            "time_sec_total": float(sub["time_sec"].sum()),
        })
    return pd.DataFrame(rows)


def append_proposed_reference(summary: pd.DataFrame) -> pd.DataFrame:
    path = os.path.join("results", "proposed_performance", "01_main_performance_summary.csv")
    if not os.path.exists(path):
        return summary
    ref = pd.read_csv(path)
    ref = ref[ref["method"] == "Proposed"]
    if ref.empty:
        return summary
    r = ref.iloc[0]
    proposed = pd.DataFrame([{
        "selection_variant": "Cluster-priority (ours)",
        "inspired_by": VARIANT_META["Cluster-priority (ours)"],
        "uses_cluster": True,
        "review_ratio": float(r["label_update_ratio"]),
        "months": int(r["months"]),
        "f1": float(r["macro_f1_mean"]),
        "f1_std": float(r["macro_f1_std"]),
        "auc_pr": float(r["auc_pr_mean"]),
        "auc_roc": float(r["auc_roc_mean"]),
        "fpr_at_tpr90": float(r["fpr_at_tpr90_mean"]),
        "mal_rate_selected": float("nan"),
        "time_sec_total": float(r["time_sec_total"]),
    }])
    return pd.concat([summary, proposed], ignore_index=True)


def write_markdown(summary_with_ref: pd.DataFrame, md_path: str,
                   detail_path: str, summary_path: str, review_ratio: float) -> None:
    lines = [
        "# Selection Criterion Ablation",
        "",
        "All non-cluster variants use the same predict-before-update protocol and the same 33% review budget.",
        f"Target review budget: `{review_ratio:.6f}`.",
        "",
        f"- Detail CSV: `{detail_path}`",
        f"- Summary CSV: `{summary_path}`",
        "",
        "```text",
        "Selection criterion ablation (same predict-before-update protocol, same 33% budget)",
        "─────────────────────────────────────────────────────────────────────────────",
        "Selection variant              | F1    | Inspired by",
        "─────────────────────────────────────────────────────────────────────────────",
    ]
    order = [
        "Random selection",
        "Uncertainty-based selection",
        "Distance-to-centroid selection",
        "Cluster-priority (ours)",
    ]
    for name in order:
        row = summary_with_ref[summary_with_ref["selection_variant"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(f"{name:<30} | {_fmt(r['f1'])} | {r['inspired_by']}")
    lines.extend([
        "─────────────────────────────────────────────────────────────────────────────",
        "```",
        "",
        "## Full Metrics",
        "",
        "| Selection variant | Review ratio | F1 | AUC-PR | FPR@90%TPR | Selected malware rate | Inspired by |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for name in order:
        row = summary_with_ref[summary_with_ref["selection_variant"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"| {name} | {_fmt(r['review_ratio'], 4)} | {_fmt(r['f1'])} | "
            f"{_fmt(r['auc_pr'])} | {_fmt(r['fpr_at_tpr90'])} | "
            f"{_fmt(r['mal_rate_selected'], 4)} | {r['inspired_by']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `Distance-to-centroid selection` uses fixed semantic embeddings and class centroids, then prioritizes samples farthest from the nearest class centroid. This is a CADE-style distance signal adapted to the binary monthly setting without CADE's family-label task.",
        "- `Cluster-priority (ours)` is the existing Proposed reference row from `results/proposed_performance/01_main_performance_summary.csv`; use it as the paper method row, not as a pure selection-only row.",
        "",
    ])

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--review-ratio", type=float, default=DEFAULT_REVIEW_RATIO)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    out_dir = os.path.join(cfg.perf_dir, "monthly")
    os.makedirs(out_dir, exist_ok=True)

    detail_path = os.path.join(out_dir, "selection_criterion_ablation.csv")
    summary_path = os.path.join(out_dir, "selection_criterion_ablation_summary.csv")
    md_path = os.path.join("documents", "SELECTION_CRITERION_ABLATION.md")

    with ExperimentTimer("selection_control_ablation", cfg.out_dir):
        print(f"\n{'=' * 70}")
        print("  EXP-24: Selection criterion ablation")
        print(f"{'=' * 70}")

        print("\n[1] Loading monthly dataset...")
        rows, y, months = load_dataset(cfg)

        print("\n[2] Temporal split...")
        m2idx, _, _, _, test_m, train_idx, _ = make_temporal_split(months, cfg)

        print("\n[3] Feature hashing...")
        X_train = build_hashed_csr(rows, train_idx, cfg.hash_dim)
        y_train = y[train_idx]
        print(f"  X_train: {X_train.shape}  nnz={X_train.nnz:,}")
        print(f"  Review budget target: {args.review_ratio:.6f}")

        print("\n[4] Fitting semantic embedder for centroid-distance baseline...")
        embedder = SemanticEmbedder(cfg.svd_components, cfg.random_state)
        embedder.fit(X_train)

        all_rows = []
        variants = [
            "Random selection",
            "Uncertainty-based selection",
            "Distance-to-centroid selection",
        ]
        for variant in variants:
            print(f"\n[5] Running {variant}...")
            all_rows.extend(run_variant(
                variant, X_train, y_train, m2idx, test_m, rows, y, cfg,
                args.review_ratio, np.random.default_rng(cfg.random_state),
                embedder=embedder,
            ))

        detail = to_result_df(all_rows)
        detail.to_csv(detail_path, index=False)
        summary = summarize(detail)
        summary_with_ref = append_proposed_reference(summary)
        summary_with_ref.to_csv(summary_path, index=False)
        write_markdown(summary_with_ref, md_path, detail_path, summary_path,
                       args.review_ratio)

        print(f"\n  Saved detail : {detail_path}")
        print(f"  Saved summary: {summary_path}")
        print(f"  Saved report : {md_path}")
        print("\n" + summary_with_ref.to_string(index=False))


if __name__ == "__main__":
    main()
