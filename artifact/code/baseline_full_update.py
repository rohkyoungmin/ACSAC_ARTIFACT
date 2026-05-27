#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline: Full Update (all samples, uniform loss — oracle labelling-cost upper bound)
실행: python baseline_full_update.py
결과: results/performance/monthly/full_update_results.csv
소요시간 예상: ~3-8분
"""
import sys
sys.path.insert(0, '.')
from common import *


def main():
    cfg = Config()
    os.makedirs(f"{cfg.perf_dir}/monthly", exist_ok=True)

    with ExperimentTimer("baseline_full_update", cfg.out_dir):
        print(f"\n{'='*60}")
        print(f"  Baseline: FullUpdate  (all samples, uniform loss)")
        print(f"{'='*60}")

        print("\n[1] Loading dataset...")
        rows, y, months = load_dataset(cfg)

        print("\n[2] Temporal split...")
        m2idx, ordered, train_m, val_m, test_m, train_idx, val_idx = \
            make_temporal_split(months, cfg)

        print("\n[3] Feature hashing...")
        X_train = build_hashed_csr(rows, train_idx, cfg.hash_dim)
        y_train = y[train_idx]
        print(f"  X_train: {X_train.shape}  nnz={X_train.nnz:,}")

        print("\n[4] Running FullUpdate baseline...")
        results = run_baseline_full_update(
            X_train, y_train, m2idx, test_m, rows, y, cfg)

        out_path = f"{cfg.perf_dir}/monthly/full_update_results.csv"
        df = to_result_df(results)
        df.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")
        print_results_table(df, title="FullUpdate — all samples, uniform loss")


if __name__ == "__main__":
    main()
