#!/usr/bin/env python3
"""Lightweight claim validator for the ACSAC AE package."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def validate_claim(claim_dir: Path) -> None:
    expected = claim_dir / "expected"
    csvs = sorted(expected.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No expected CSV files found in {expected}")

    print(f"Claim: {claim_dir.name}")
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        print(f"- {csv_path.name}: {len(df)} rows, columns={list(df.columns)}")
        print(df.to_string(index=False))
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("claim_dir", help="Path to a claims/claim*/ directory")
    args = parser.parse_args()
    validate_claim(Path(args.claim_dir).resolve())


if __name__ == "__main__":
    main()
