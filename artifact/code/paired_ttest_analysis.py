#!/usr/bin/env python3
"""Paper-style paired t-tests for Tables VI-VII.

The submitted paper reports a two-sided paired t-test over chronological
period-level F1 values, together with a 95% confidence interval for the paired
mean difference.  This script applies that same test to the archived CSVs used
for Tables VI and VII.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "artifact" / "results" / "full_csv"
OUT = ROOT / "artifact" / "results" / "paper_tables"


def _metric_series(
    df: pd.DataFrame,
    period_col: str,
    metric_col: str = "f1",
) -> pd.DataFrame:
    out = df[[period_col, metric_col]].copy()
    out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce")
    return out.dropna(subset=[period_col, metric_col]).sort_values(period_col)


def _paired_ttest_row(
    *,
    table: str,
    setting: str,
    method_a: str,
    method_b: str,
    a: pd.DataFrame,
    b: pd.DataFrame,
    period_col: str,
    metric_col: str = "f1",
) -> dict:
    left = _metric_series(a, period_col, metric_col).rename(columns={metric_col: "a"})
    right = _metric_series(b, period_col, metric_col).rename(columns={metric_col: "b"})
    paired = left.merge(right, on=period_col, how="inner").sort_values(period_col)
    diffs = paired["a"].to_numpy(dtype=float) - paired["b"].to_numpy(dtype=float)
    n = int(len(diffs))

    if n < 2:
        t_stat = p_value = mean_diff = std_diff = ci_low = ci_high = np.nan
        dfree = n - 1
    else:
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        dfree = n - 1
        if std_diff == 0.0:
            t_stat = 0.0 if mean_diff == 0.0 else np.inf * np.sign(mean_diff)
            p_value = 1.0 if mean_diff == 0.0 else 0.0
            ci_low = ci_high = mean_diff
        else:
            t_stat, p_value = stats.ttest_rel(paired["a"], paired["b"])
            se = std_diff / np.sqrt(n)
            margin = stats.t.ppf(0.975, dfree) * se
            ci_low = mean_diff - margin
            ci_high = mean_diff + margin

    return {
        "table": table,
        "setting": setting,
        "method_a": method_a,
        "method_b": method_b,
        "comparison": f"{method_a} - {method_b}",
        "metric": metric_col,
        "test": "two-sided paired t-test over periods",
        "n_periods": n,
        "df": dfree,
        "method_a_mean": float(paired["a"].mean()) if n else np.nan,
        "method_b_mean": float(paired["b"].mean()) if n else np.nan,
        "mean_diff": mean_diff,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else t_stat,
        "p_value": float(p_value) if np.isfinite(p_value) else p_value,
        "significant_p05": bool(p_value < 0.05) if np.isfinite(p_value) else False,
        "periods": ";".join(str(x) for x in paired[period_col].tolist()),
    }


def _method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return df.loc[df["method"] == method].copy()


def _selection(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    return df.loc[df["selection_variant"] == variant].copy()


def build_rows() -> list[dict]:
    rows: list[dict] = []

    monthly = pd.read_csv(FULL / "proposed_performance" / "01_main_performance.csv")
    yearly = pd.read_csv(
        FULL / "performance" / "baseline_tables" / "yearly_baseline_comparison.csv"
    )
    yearly = yearly[
        (yearly["dataset_setting"] == "androzoo_year_imbalanced_9to1")
        & (yearly["table_group"] == "yearly_ablation_baselines")
    ].copy()

    # Table VI: Proposed against the baselines shown in the detection table.
    table_vi = [
        ("Monthly", "Proposed", "Static", _method(monthly, "Proposed"), _method(monthly, "Static"), "month"),
        (
            "Monthly",
            "Proposed",
            "FullUpdate",
            _method(monthly, "Proposed"),
            _method(monthly, "FullLabelUpdate"),
            "month",
        ),
        ("Yearly", "Proposed", "Static", _method(yearly, "Proposed"), _method(yearly, "Static"), "period"),
        (
            "Yearly",
            "Proposed",
            "FullUpdate",
            _method(yearly, "Proposed"),
            _method(yearly, "FullUpdate"),
            "period",
        ),
    ]
    for setting, method_a, method_b, a, b, period_col in table_vi:
        rows.append(
            _paired_ttest_row(
                table="VI",
                setting=setting,
                method_a=method_a,
                method_b=method_b,
                a=a,
                b=b,
                period_col=period_col,
            )
        )

    # Table VII: same-budget cluster-free selection controls vs Proposed.
    selection = pd.read_csv(
        FULL / "performance" / "monthly" / "selection_criterion_ablation.csv"
    )
    proposed = _method(monthly, "Proposed")
    for variant, display in [
        ("Random selection", "RandomSelective"),
        ("Uncertainty-based selection", "UncertaintyOnly"),
        ("Distance-to-centroid selection", "Distance-to-centroid"),
    ]:
        rows.append(
            _paired_ttest_row(
                table="VII",
                setting="Monthly",
                method_a="Proposed",
                method_b=display,
                a=proposed,
                b=_selection(selection, variant),
                period_col="month",
            )
        )

    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(build_rows())

    float_cols = [
        "method_a_mean",
        "method_b_mean",
        "mean_diff",
        "ci_95_low",
        "ci_95_high",
        "t_stat",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(6)
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce").map(
        lambda x: float(f"{x:.12g}") if pd.notna(x) else np.nan
    )

    combined = OUT / "statistical_tests_paired_ttest.csv"
    df.to_csv(combined, index=False)
    for table in ["VI", "VII"]:
        df.loc[df["table"] == table].to_csv(
            OUT / f"table_{table.lower()}_paired_ttests.csv", index=False
        )

    print(f"Saved: {combined}")
    print(df.drop(columns=["periods"]).to_string(index=False))


if __name__ == "__main__":
    main()
