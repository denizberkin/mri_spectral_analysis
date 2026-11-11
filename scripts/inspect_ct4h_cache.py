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
from ct4h.data.utils import (  # noqa: E402
    DOSES, HU_RANGE, SCANNER_SPLIT, SOFT_WINDOW, dose_pairs, read_cache_index, split_selection,
)

AIR = -900.0          # below this a voxel counts as air
TISSUE = -500.0       # above this a voxel counts as phantom / body
SOFT = SOFT_WINDOW


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
    print(f"{'scanner':8s}{'air%':>7s}{'p1':>8s}{'p50':>8s}{'p99':>8s}{'p99.9':>8s}{'in soft%':>10s}   z with >90% tissue voxels")
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
        z_with = np.nonzero(tissue_fraction > 0.9)[0]
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
    print("  Slices outside these ranges are nearly empty or wrapped round by the authors' circular roll;")
    print("  take data.params.z_range = [max of the firsts + 1, min of the lasts] as a safe common range.")


def pairs_grid(cache_dir: Path, split_mode: str, out_dir: Path, z_range) -> None:
    cases = [(1, 14), (6, 14), (14, 1)]
    cols = 3
    fig, axes = plt.subplots(len(cases) * 2, cols, figsize=(3.6 * cols, 3.4 * len(cases) * 2))
    for r, (src, tgt) in enumerate(cases):
        direction = "up" if tgt > src else "down"
        ds = CT4HPairDataset(
            cache_dir, "val", split_mode, directions=direction, doses=[src, tgt], neighbours=0,
            length=3, deterministic=True, seed=3, pad_multiple=1, z_range=z_range,
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


def identity_baseline(cache_dir: Path, split_mode: str, samples: int, out_dir: Path, z_range) -> None:
    """MAE (HU) and PSNR of returning the source slice, per dose pair, on the validation split."""
    print(f"\n== identity baseline, split=val ({split_mode}), {samples} draws per pair, z {z_range[0]}..{z_range[1]} ==")
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
                target_mode=mode, length=samples, deterministic=True, seed=5, pad_multiple=1, z_range=z_range,
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
        r, m = values["repeat"], values["mean"]
        table.append([src, tgt, direction, f"{r[0]:.3f}", f"{m[0]:.3f}", f"{r[1]:.3f}", f"{m[1]:.3f}",
                     f"{r[2]:.2f}", f"{m[2]:.2f}"])
        print(f"{f'{src}->{tgt}':>9s}{r[0]:10.2f}{m[0]:10.2f}{r[1]:10.2f}{m[1]:10.2f}{r[2]:10.2f}{m[2]:10.2f}")
    with open(out_dir / "identity_baseline.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(table)


def same_dose_floor(cache_dir: Path, rows: list[dict], split_mode: str, samples: int, out_dir: Path, z_range) -> None:
    """MAE between two different repeats of the same dose on the same scanner (validation split).

    Both volumes went through the same registration to the same reference, so this is the noise of two images plus
    whatever misalignment two registrations leave. For independent noise the noise of ONE image is about this
    divided by sqrt(2) (the last column), and no dose-to-dose model can do better than that against a single
    noisy repeat.
    """
    scanners, repeats = split_selection(split_mode, "val")
    if len(repeats) < 2:
        print(f"\n== same-dose floor: the val split of the {split_mode!r} mode has one repeat ({repeats}), "
              "so there is no second repeat of the same dose to compare with. Skipped "
              "(use --split_mode scanner for this table). ==")
        return
    z_lo, z_hi = z_range
    rng = np.random.default_rng(7)
    by_key: dict[tuple[str, int], dict[int, str]] = {}
    for row in rows:
        if row["recon"] == "FBP" and row["scanner"] in scanners and row["repeat"] in repeats:
            by_key.setdefault((row["scanner"], row["dose_mgy"]), {})[row["repeat"]] = row["path"]
    print(f"\n== same-dose floor, split=val ({split_mode}), {samples} slices per scanner and dose ==")
    print(f"{'dose':>6s}{'MAE':>9s}{'soft MAE':>10s}{'soft/sqrt2':>12s}   (HU, two repeats of the same dose)")
    table = []
    for dose in DOSES:
        full, soft = [], []
        for scanner in scanners:
            files = by_key.get((scanner, dose), {})
            if len(files) < 2:
                continue
            for _ in range(samples):
                first, second = rng.choice(sorted(files), size=2, replace=False)
                z = int(rng.integers(z_lo, z_hi))
                a = np.load(cache_dir / files[int(first)], mmap_mode="r")[z].astype(np.float32)
                b = np.load(cache_dir / files[int(second)], mmap_mode="r")[z].astype(np.float32)
                full.append(np.abs(a - b).mean())
                soft.append(np.abs(np.clip(a, *SOFT) - np.clip(b, *SOFT)).mean())
        if not full:
            continue
        table.append([dose, f"{np.mean(full):.3f}", f"{np.mean(soft):.3f}", f"{np.mean(soft) / np.sqrt(2):.3f}"])
        print(f"{dose:>6d}{np.mean(full):9.2f}{np.mean(soft):10.2f}{np.mean(soft) / np.sqrt(2):12.2f}")
    with open(out_dir / "same_dose_floor.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dose_mgy", "mae", "soft_mae", "soft_mae_per_image"])
        writer.writerows(table)


def z_montage(cache_dir: Path, rows: list[dict], out_dir: Path, step: int = 12) -> None:
    """Every ``step``-th slice of one 14 mGy volume, soft-tissue window, labelled with z."""
    wanted = [r for r in rows if r["recon"] == "FBP" and r["dose_mgy"] == 14 and r["repeat"] == 1]
    if not wanted:
        print("no 14 mGy FBP repeat-1 volume found, skipping the montage")
        return
    row = sorted(wanted, key=lambda r: r["scanner"])[0]
    volume = np.load(cache_dir / row["path"], mmap_mode="r")
    zs = list(range(0, volume.shape[0], step))
    cols = 6
    n_rows = -(-len(zs) // cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(2.6 * cols, 2.3 * n_rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, z in zip(axes.ravel(), zs):
        ax.imshow(np.asarray(volume[z]), cmap="gray", vmin=SOFT[0], vmax=SOFT[1])
        ax.set_title(f"z={z}", fontsize=8)
    fig.suptitle(f"{row['scanner']} 14 mGy FBP rep 1, every {step}th slice, window {SOFT[0]:.0f}..{SOFT[1]:.0f} HU")
    fig.tight_layout()
    fig.savefig(out_dir / "z_montage.png", dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split_mode", default="scanner", choices=["scanner", "repeat"])
    parser.add_argument("--samples", type=int, default=100, help="draws per dose pair for the identity baseline")
    parser.add_argument("--z_lo", type=int, default=11, help="first slice used for the baselines (matches data.params.z_range)")
    parser.add_argument("--z_hi", type=int, default=304, help="end (exclusive) of the slice range for the baselines")
    args = parser.parse_args()

    cache_dir, out_dir = Path(args.cache_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_cache_index(cache_dir)
    print(f"{len(rows)} volumes in the cache; validation scanners ({args.split_mode} split): "
          f"{SCANNER_SPLIT['val'] if args.split_mode == 'scanner' else 'all, repeat 8'}")
    z_range = (args.z_lo, args.z_hi)
    hu_and_coverage(cache_dir, rows, out_dir)
    z_montage(cache_dir, rows, out_dir)
    pairs_grid(cache_dir, args.split_mode, out_dir, z_range)
    identity_baseline(cache_dir, args.split_mode, args.samples, out_dir, z_range)
    same_dose_floor(cache_dir, rows, args.split_mode, max(args.samples // 4, 5), out_dir, z_range)
    print(f"\nfigures and csv in {out_dir}")


if __name__ == "__main__":
    main()
