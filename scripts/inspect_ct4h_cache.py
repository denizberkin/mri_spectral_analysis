#!/usr/bin/env python
"""Look at the cached CT4H data before training anything.

    python scripts/inspect_ct4h_cache.py --cache_dir $CT/cache --out_dir $CT/inspect

Writes into out_dir and prints:
  1. hu_histogram.png + percentiles: where the phantom lives in HU (decides the HU window).
  2. z_coverage.png + the slice range containing tissue (decides data.params.z_range).
  3. pairs_grid.png: real source / target / difference slices for a few dose pairs.
  4. identity_baseline.csv: error of "return the source unchanged" for all 20 dose pairs, against one noisy
     target repeat and against the repeat mean. A trained model has to beat these numbers.
Everything is computed from the cache; nothing is written to it.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct4h.data.cached_dataset import CT4HPairDataset, unit_to_hu  # noqa: E402
from ct4h.data.utils import DOSES, HU_RANGE, SCANNER_SPLIT, dose_pairs  # noqa: E402

AIR = -900.0          # below this a voxel counts as air
TISSUE = -500.0       # above this a voxel counts as phantom / body
SOFT = (-160.0, 240.0)


def read_index(cache_dir: Path) -> list[dict]:
    with open(cache_dir / "cache_index.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["dose_mgy"], row["repeat"] = int(row["dose_mgy"]), int(row["repeat"])
    return rows


def hu_and_coverage(cache_dir: Path, rows: list[dict], out_dir: Path) -> None:
    """HU histogram and per-slice tissue fraction from one 10 mGy FBP volume per scanner."""
    chosen = {}
    for row in rows:
        if row["recon"] == "FBP" and row["dose_mgy"] == 10 and row["repeat"] == 1:
            chosen[row["scanner"]] = row
    if not chosen:
        print("no 10 mGy FBP repeat-1 volumes found, skipping histogram and coverage")
        return
    edges = np.arange(HU_RANGE[0], HU_RANGE[1] + 10, 10.0)
    fig_h, ax_h = plt.subplots(figsize=(9, 4))
    fig_z, ax_z = plt.subplots(figsize=(9, 4))
    print("\n== HU distribution (10 mGy FBP rep 1, every 5th slice) and tissue z-range ==")
    print(f"{'scanner':8s}{'air%':>7s}{'p1':>8s}{'p50':>8s}{'p99':>8s}{'p99.9':>8s}{'in soft%':>10s}   z with tissue (>1% voxels)")
    for scanner, row in sorted(chosen.items()):
        volume = np.load(cache_dir / row["path"], mmap_mode="r")
        sample = np.asarray(volume[::5])
        counts, _ = np.histogram(sample, bins=edges)
        ax_h.plot(edges[:-1], counts + 1, label=scanner, lw=0.8)
        flat = sample.ravel()
        body = flat[flat > AIR]
        soft = ((flat >= SOFT[0]) & (flat <= SOFT[1])).mean() * 100
        p = np.percentile(body, [1, 50, 99, 99.9]) if body.size else [np.nan] * 4
        tissue_fraction = (np.asarray(volume) > TISSUE).mean(axis=(1, 2))
        ax_z.plot(tissue_fraction, label=scanner, lw=0.8)
        z_with = np.nonzero(tissue_fraction > 0.01)[0]
        z_text = f"{z_with.min()}..{z_with.max()} of {volume.shape[0]}" if z_with.size else "none"
        print(f"{scanner:8s}{(flat <= AIR).mean() * 100:7.1f}{p[0]:8.0f}{p[1]:8.0f}{p[2]:8.0f}{p[3]:8.0f}{soft:10.1f}   {z_text}")
    ax_h.set(yscale="log", xlabel="HU", ylabel="voxels + 1", title="HU histogram (10 HU bins)")
    ax_h.legend(ncol=5, fontsize=7)
    ax_z.set(xlabel="slice index z", ylabel="fraction of voxels > -500 HU", title="Tissue per slice")
    ax_z.legend(ncol=5, fontsize=7)
    fig_h.tight_layout(); fig_z.tight_layout()
    fig_h.savefig(out_dir / "hu_histogram.png", dpi=130)
    fig_z.savefig(out_dir / "z_coverage.png", dpi=130)
    plt.close("all")
    print("  'in soft%' = share of all voxels inside the soft-tissue window (-160..240 HU).")
    print("  Use the z ranges above for data.params.z_range if a lot of the volume is empty.")


def pairs_grid(cache_dir: Path, split_mode: str, out_dir: Path) -> None:
    cases = [(1, 14), (6, 14), (14, 1)]
    rows, cols = len(cases) * 1, 3
    fig, axes = plt.subplots(rows * 2, cols, figsize=(3.6 * cols, 3.4 * rows * 2))
    for r, (src, tgt) in enumerate(cases):
        direction = "up" if tgt > src else "down"
        ds = CT4HPairDataset(
            cache_dir, "val", split_mode, directions=direction, doses=[src, tgt], neighbours=0,
            length=3, deterministic=True, seed=3, pad_multiple=1,
        )
        for c in range(cols):
            item = ds[c]
            low, high = ds.hu_window
            source = unit_to_hu((item["source"][0].numpy() + 1) / 2, low, high)
            target = unit_to_hu((item["target"][0].numpy() + 1) / 2, low, high)
            ax = axes[2 * r, c]
            ax.imshow(source, cmap="gray", vmin=SOFT[0], vmax=SOFT[1])
            ax.set_title(f"{item['scanner']} z={item['slice_idx']}  src {src} mGy", fontsize=8)
            ax.axis("off")
            ax = axes[2 * r + 1, c]
            ax.imshow(target, cmap="gray", vmin=SOFT[0], vmax=SOFT[1])
            ax.set_title(f"target {tgt} mGy (rep {item['target_repeat']})", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"source (top) vs target (bottom) of each pair, window {SOFT[0]:.0f}..{SOFT[1]:.0f} HU", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "pairs_grid.png", dpi=110)
    plt.close(fig)


def identity_baseline(cache_dir: Path, split_mode: str, samples: int, out_dir: Path) -> None:
    """MAE (HU) and PSNR of returning the source slice, per dose pair, on the validation split."""
    print(f"\n== identity baseline, split=val ({split_mode}), {samples} draws per pair ==")
    print("MAE in HU over the whole slice (air included), PSNR with peak 3000 HU, "
          "'soft' = MAE after clipping both images to -160..240 HU.")
    header = ["source_mgy", "target_mgy", "direction", "mae_repeat", "mae_mean", "soft_mae_repeat", "soft_mae_mean",
              "psnr_repeat", "psnr_mean"]
    table = []
    print(f"{'pair':>9s}{'MAE rep':>10s}{'MAE mean':>10s}{'soft rep':>10s}{'soft mean':>10s}{'PSNR rep':>10s}{'PSNR mean':>10s}")
    for src, tgt in dose_pairs("any"):
        direction = "up" if tgt > src else "down"
        values = {}
        for mode in ("repeat", "mean"):
            ds = CT4HPairDataset(
                cache_dir, "val", split_mode, directions=direction, doses=[src, tgt], neighbours=0,
                target_mode=mode, length=samples, deterministic=True, seed=5, pad_multiple=1,
            )
            low, high = ds.hu_window
            mae, soft, mse = [], [], []
            for i in range(samples):
                item = ds[i]
                a = unit_to_hu((item["source"][0].numpy() + 1) / 2, low, high)
                b = unit_to_hu((item["target"][0].numpy() + 1) / 2, low, high)
                mae.append(np.abs(a - b).mean())
                soft.append(np.abs(np.clip(a, *SOFT) - np.clip(b, *SOFT)).mean())
                mse.append(((a - b) ** 2).mean())
            peak = high - low
            values[mode] = (np.mean(mae), np.mean(soft), 10 * np.log10(peak**2 / np.mean(mse)))
        table.append([src, tgt, direction, *[f"{values['repeat'][k]:.3f}" for k in (0, 1)],
                      *[f"{values['mean'][k]:.3f}" for k in (0, 1)],
                      f"{values['repeat'][2]:.2f}", f"{values['mean'][2]:.2f}"])
        # reorder to header: mae_repeat, mae_mean, soft_repeat, soft_mean, psnr_repeat, psnr_mean
        r, m = values["repeat"], values["mean"]
        table[-1] = [src, tgt, direction, f"{r[0]:.3f}", f"{m[0]:.3f}", f"{r[1]:.3f}", f"{m[1]:.3f}",
                     f"{r[2]:.2f}", f"{m[2]:.2f}"]
        print(f"{f'{src}->{tgt}':>9s}{r[0]:10.2f}{m[0]:10.2f}{r[1]:10.2f}{m[1]:10.2f}{r[2]:10.2f}{m[2]:10.2f}")
    with open(out_dir / "identity_baseline.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split_mode", default="scanner", choices=["scanner", "repeat"])
    parser.add_argument("--samples", type=int, default=100, help="draws per dose pair for the identity baseline")
    args = parser.parse_args()

    cache_dir, out_dir = Path(args.cache_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_index(cache_dir)
    print(f"{len(rows)} volumes in the cache; validation scanners ({args.split_mode} split): "
          f"{SCANNER_SPLIT['val'] if args.split_mode == 'scanner' else 'all, repeat 8'}")
    hu_and_coverage(cache_dir, rows, out_dir)
    pairs_grid(cache_dir, args.split_mode, out_dir)
    identity_baseline(cache_dir, args.split_mode, args.samples, out_dir)
    print(f"\nfigures and csv in {out_dir}")


if __name__ == "__main__":
    main()
