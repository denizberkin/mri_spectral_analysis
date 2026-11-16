#!/usr/bin/env python3
"""Aggregate the leaderboard scores of the paper rerun: per-arm mean +- sd and the seed-matched MIM gain.

    python scripts/summarise_rerun.py [--csv rerun/leaderboard_scores.csv]

Fill rerun/leaderboard_scores.csv by hand from the leaderboard (nrmse lower is better, ssim higher,
lpips lower). Rows with an empty score are ignored. With n=3 the sd is a spread, not a significance test:
report the per-seed values next to it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("nrmse", "ssim", "lpips")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ROOT / "rerun" / "leaderboard_scores.csv")
    args = parser.parse_args()

    table = pd.read_csv(args.csv)
    table = table.dropna(subset=list(METRICS), how="all")
    if table.empty:
        raise SystemExit("no scores filled in yet")

    print("per run")
    print(table[["arm", "seed", *METRICS]].to_string(index=False))

    print("\nper arm (mean +- sd over seeds, n = number of seeds)")
    for arm, group in table.groupby("arm"):
        cells = [f"{m} {group[m].mean():.4f} +- {group[m].std(ddof=1) if len(group) > 1 else float('nan'):.4f}"
                 for m in METRICS if group[m].notna().any()]
        print(f"  {arm:8s} n={len(group)}  " + "   ".join(cells))

    arms = set(table["arm"])
    if {"mim", "scratch"} <= arms:
        wide = table.pivot(index="seed", columns="arm", values=list(METRICS))
        print("\nseed-matched difference, mim - scratch (positive ssim = MIM better; negative nrmse/lpips = MIM better)")
        for metric in METRICS:
            diff = (wide[(metric, "mim")] - wide[(metric, "scratch")]).dropna()
            if diff.empty:
                continue
            spread = f"{diff.std(ddof=1):.4f}" if len(diff) > 1 else "nan"
            print(f"  {metric:6s} per seed {[round(float(d), 4) for d in diff]}  mean {diff.mean():+.4f}  sd {spread}")


if __name__ == "__main__":
    main()
