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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ct4h.data.utils import DOSES, SCANNERS, SOFT_WINDOW, VENDORS, read_cache_index  # noqa: E402

SOFT = SOFT_WINDOW
FIGURES = ["dose", "scanner", "repeat", "recon", "model", "results", "noise", "identity"]


class Cache:
    """Read access to the int16 volumes through cache_index.csv."""

    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
        self.rows = read_cache_index(cache_dir)
        self.paths = {(r["scanner"], r["recon"], r["dose_mgy"], r["repeat"]): r["path"] for r in self.rows}

    def has(self, scanner: str, recon: str, dose: int, repeat: int = 1) -> bool:
        return (scanner, recon, dose, repeat) in self.paths

    def repeats(self, scanner: str, recon: str, dose: int) -> list[int]:
        return sorted(k[3] for k in self.paths if k[:3] == (scanner, recon, dose))

    def slice(self, scanner: str, recon: str, dose: int, repeat: int, z: int) -> np.ndarray:
        volume = np.load(self.dir / self.paths[(scanner, recon, dose, repeat)], mmap_mode="r")
        return np.asarray(volume[z], dtype=np.float32)

    def mean_slice(self, scanner: str, recon: str, dose: int, z: int) -> np.ndarray:
        return np.mean([self.slice(scanner, recon, dose, r, z) for r in self.repeats(scanner, recon, dose)], axis=0)


def show(ax, image, title: str = "", window=SOFT, cmap="gray", crop=None, fontsize=8) -> None:
    if crop is not None:
        top, left, size = crop
        image = image[top : top + size, left : left + size]
    ax.imshow(image, cmap=cmap, vmin=window[0], vmax=window[1], interpolation="nearest")
    ax.set_title(title, fontsize=fontsize)
    ax.axis("off")


def center_crop(image: np.ndarray, size: int) -> tuple[int, int, int]:
    return (image.shape[0] - size) // 2, (image.shape[1] - size) // 2, size


def fig_dose(cache: Cache, out: Path, scanner: str, z: int, zoom: int) -> None:
    doses = [d for d in DOSES if cache.has(scanner, "FBP", d)]
    fig, axes = plt.subplots(2, len(doses), figsize=(2.9 * len(doses), 5.4))
    for column, dose in enumerate(doses):
        image = cache.slice(scanner, "FBP", dose, 1, z)
        show(axes[0, column], image, f"{dose} mGy")
        show(axes[1, column], image, "", crop=center_crop(image, zoom))
    fig.suptitle(f"Scanner {scanner}, slice {z}, FBP, repeat 1, window {SOFT[0]:.0f}..{SOFT[1]:.0f} HU (bottom: {zoom}-pixel zoom)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "factors_dose.png", dpi=150)
    plt.close(fig)


def fig_scanner(cache: Cache, out: Path, z: int, dose: int) -> None:
    present = [s for s in SCANNERS if cache.has(s, "FBP", dose)]
    cols = 5
    rows = -(-len(present) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.35 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, scanner in zip(axes.ravel(), present):
        show(ax, cache.slice(scanner, "FBP", dose, 1, z), f"{scanner} ({VENDORS[scanner]})")
    fig.suptitle(f"The same slice ({z}) on every scanner, {dose} mGy, FBP, repeat 1", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "factors_scanner.png", dpi=140)
    plt.close(fig)


def fig_repeat(cache: Cache, out: Path, scanner: str, z: int) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 8.4))
    for row, dose in enumerate((1, 14)):
        repeats = cache.repeats(scanner, "FBP", dose)[:3]
        for column, repeat in enumerate(repeats):
            show(axes[row, column], cache.slice(scanner, "FBP", dose, repeat, z), f"{dose} mGy, repeat {repeat}")
        show(axes[row, 3], cache.mean_slice(scanner, "FBP", dose, z), f"{dose} mGy, mean of {len(cache.repeats(scanner, 'FBP', dose))} repeats")
    diff_window = (-100.0, 100.0)
    for column, dose in enumerate((1, 14)):
        a, b = cache.repeats(scanner, "FBP", dose)[:2]
        show(axes[2, column], cache.slice(scanner, "FBP", dose, a, z) - cache.slice(scanner, "FBP", dose, b, z),
             f"{dose} mGy: repeat {a} - repeat {b}", window=diff_window, cmap="coolwarm")
    one = cache.slice(scanner, "FBP", 14, 1, z) - cache.mean_slice(scanner, "FBP", 14, z)
    show(axes[2, 2], one, "14 mGy: repeat 1 - mean", window=diff_window, cmap="coolwarm")
    axes[2, 3].axis("off")
    fig.suptitle(f"Scanner {scanner}, slice {z}: repeats of one scan differ by noise only. Top rows window "
                 f"{SOFT[0]:.0f}..{SOFT[1]:.0f} HU, bottom row differences {diff_window[0]:.0f}..{diff_window[1]:.0f} HU", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "factors_repeat.png", dpi=140)
    plt.close(fig)


def registered_slice(reg_dir: Path, scanner: str, recon: str, dose: int, repeat: int, z: int) -> np.ndarray | None:
    """One slice straight from a registered NIfTI (same orientation as the cache: nibabel (x, y, z) -> (z, y, x)).
    Lets the reconstruction figure be made without caching IR and DL (about 42 GB for IR alone)."""
    import nibabel as nib

    path = reg_dir / f"{scanner}_{dose}mGy_{recon}_rep{repeat:02d}.nii.gz"
    if not path.is_file():
        return None
    volume = np.asanyarray(nib.load(str(path)).dataobj)
    return np.asarray(np.transpose(volume, (2, 1, 0))[z], dtype=np.float32)


def fig_recon(cache: Cache, out: Path, z: int, dose: int, reg_dir: Path | None = None) -> None:
    def get(scanner: str, recon: str):
        if reg_dir is not None:
            return registered_slice(reg_dir, scanner, recon, dose, 1, z)
        return cache.slice(scanner, recon, dose, 1, z) if cache.has(scanner, recon, dose) else None

    images = {s: [get(s, r) for r in ("FBP", "IR", "DL")] for s in SCANNERS}
    scanners = [s for s, row in images.items() if all(img is not None for img in row)]
    if not scanners:
        print(f"factors_recon: no scanner has FBP, IR and DL at {dose} mGy repeat 1 "
              f"({'in ' + str(reg_dir) if reg_dir else 'in the cache; pass --reg_dir to read the registered NIfTI files instead'}). Skipped.")
        return
    fig, axes = plt.subplots(len(scanners), 3, figsize=(9.5, 2.9 * len(scanners)), squeeze=False)
    for row, scanner in enumerate(scanners):
        for column, recon in enumerate(("FBP", "IR", "DL")):
            show(axes[row, column], images[scanner][column], f"{scanner} {recon}")
    fig.suptitle(f"Reconstruction of the same scan, {dose} mGy, slice {z}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "factors_recon.png", dpi=140)
    plt.close(fig)


def fig_model(cache: Cache, out: Path, checkpoint: str, cases: list[tuple[str, int, int]], z: int, device: str,
              hu_window: tuple[float, float]) -> None:
    import torch

    import evaluate_ct4h as ev
    from ct4h.data.cached_dataset import CT4HPairDataset, unit_to_hu

    model, uses_slice_pos = ev.load_model(checkpoint, torch.device(device))
    neighbours = (int(model.input_channels) - 1) // 2
    predict = ev.make_predict(model, uses_slice_pos, torch.device(device))
    low, high = hu_window
    error_window = (0.0, 80.0)
    fig, axes = plt.subplots(len(cases), 5, figsize=(14.5, 2.85 * len(cases)), squeeze=False)
    for row, (scanner, source_dose, target_dose) in enumerate(cases):
        direction = "up" if target_dose > source_dose else "down"
        dataset = CT4HPairDataset(
            cache.dir, "val", "scanner", directions=direction, doses=[source_dose, target_dose], neighbours=neighbours,
            hu_window=hu_window, target_mode="mean", pad_multiple=16, z_range=(z, z + 1), scanners=[scanner],
            length=1, deterministic=True, seed=1, include_slice_pos=uses_slice_pos)
        item = dataset[0]
        batch = {k: [v.numpy() if hasattr(v, "numpy") else v] for k, v in item.items()}
        height, width = int(item["orig_height"]), int(item["orig_width"])
        to_hu = lambda x: unit_to_hu((np.asarray(x)[:height, :width] + 1) / 2, low, high)  # noqa: E731
        source, target, output = to_hu(item["source"].numpy()[0]), to_hu(item["target"].numpy()[0]), to_hu(predict(batch)[0])
        mae = lambda a: float(np.mean(np.abs(np.clip(a, *SOFT) - np.clip(target, *SOFT))))  # noqa: E731
        show(axes[row, 0], source, f"{scanner}: source {source_dose} mGy")
        show(axes[row, 1], output, f"model output (told {target_dose} mGy)")
        show(axes[row, 2], target, f"target {target_dose} mGy (repeat mean)")
        show(axes[row, 3], np.abs(np.clip(source, *SOFT) - np.clip(target, *SOFT)), f"|source - target|, MAE {mae(source):.1f} HU",
             window=error_window, cmap="magma")
        show(axes[row, 4], np.abs(np.clip(output, *SOFT) - np.clip(target, *SOFT)), f"|output - target|, MAE {mae(output):.1f} HU",
             window=error_window, cmap="magma")
    fig.suptitle(f"Validation scanners (never trained on), slice {z}. Images {SOFT[0]:.0f}..{SOFT[1]:.0f} HU, errors 0..{error_window[1]:.0f} HU",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "model_examples.png", dpi=130)
    plt.close(fig)


def fig_results(csv_path: Path, out: Path) -> None:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if "mean_model_mae" not in rows[0]:
        print("results_pairs: the CSV has no model columns (it was made without a checkpoint). Skipped.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), sharey=True)
    for ax, direction in zip(axes, ("up", "down")):
        part = [r for r in rows if r["direction"] == direction]
        part.sort(key=lambda r: (int(r["target_mgy"]), int(r["source_mgy"])) if direction == "up" else (int(r["source_mgy"]), int(r["target_mgy"])))
        x = np.arange(len(part))
        identity = [float(r["mean_identity_mae"]) for r in part]
        model = [float(r["mean_model_mae"]) for r in part]
        ax.bar(x - 0.2, identity, 0.4, color="0.7", label="return the source")
        ax.bar(x + 0.2, model, 0.4, color="black", label="model")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['source_mgy']}→{r['target_mgy']}" for r in part], fontsize=8)
        ax.set_title("upward pairs (low dose to high dose)" if direction == "up" else "downward pairs (high dose to low dose)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    top = max(float(r["mean_identity_mae"]) for r in rows)
    axes[0].set_ylim(0, 1.25 * top)
    axes[0].set_ylabel("MAE vs repeat mean, soft window (HU)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "results_pairs.png", dpi=170)
    plt.close(fig)


def fig_noise(floor_csv: Path, out: Path, repeats: int = 10) -> None:
    """One image: soft MAE between two repeats / sqrt(2), as measured by inspect_ct4h_cache.py.
    Mean of N repeats: NOT measured here. It is the one-image value / sqrt(N), which holds only if the noise of
    different repeats is independent; it is drawn dashed and labelled as expected."""
    with open(floor_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dose = np.array([int(r["dose_mgy"]) for r in rows])
    one = np.array([float(r["soft_mae_per_image"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    grid = np.linspace(dose.min(), dose.max() + 1, 100)
    ax.plot(grid, one[0] * np.sqrt(dose[0] / grid), "--", color="0.6", lw=1, label=f"{one[0]:.1f}/\u221adose scaled to {dose[0]} mGy")
    ax.plot(dose, one, "o-", color="black", label="one image (measured)")
    ax.plot(dose, one / np.sqrt(repeats), "s--", color="#2a6fb0", label=f"mean of {repeats} repeats (expected, one image / \u221a{repeats})")
    for x, y in zip(dose, one):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xticks(dose)
    ax.set_xlabel("dose (mGy)")
    ax.set_ylabel("noise, soft-window MAE (HU)")
    ax.set_ylim(0, 1.12 * one.max())
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "noise_vs_dose.png", dpi=200)
    plt.close(fig)


def fig_identity(identity_csv: Path, out: Path, column: str = "soft_mae_mean") -> None:
    with open(identity_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    doses = sorted({int(r["source_mgy"]) for r in rows} | {int(r["target_mgy"]) for r in rows})
    grid = np.full((len(doses), len(doses)), np.nan)
    for r in rows:
        grid[doses.index(int(r["source_mgy"])), doses.index(int(r["target_mgy"]))] = float(r[column])
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=np.nanmax(grid) * 1.25)
    for i in range(len(doses)):
        for j in range(len(doses)):
            text = "-" if np.isnan(grid[i, j]) else f"{grid[i, j]:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if not np.isnan(grid[i, j]) and grid[i, j] > 0.6 * np.nanmax(grid) else "black")
    ax.set_xticks(range(len(doses)), doses)
    ax.set_yticks(range(len(doses)), doses)
    ax.set_xlabel("target dose (mGy)")
    ax.set_ylabel("source dose (mGy)")
    fig.tight_layout()
    fig.savefig(out / "identity_heatmap.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--only", nargs="+", choices=FIGURES, default=None, help="default: every figure that can be made")
    parser.add_argument("--scanner", default="A1", help="scanner for the dose and repeat figures")
    parser.add_argument("--z", type=int, default=126, help="slice index (126 shows the liver with lesions)")
    parser.add_argument("--zoom", type=int, default=120, help="pixels of the zoom crop in the dose figure")
    parser.add_argument("--recon_dose", type=int, default=6)
    parser.add_argument("--reg_dir", default=None, help="registered NIfTI folder: the reconstruction figure reads it directly "
                        "(no need to cache IR and DL)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pairs_csv", default=None)
    parser.add_argument("--floor_csv", default=None, help="same_dose_floor.csv from inspect_ct4h_cache.py")
    parser.add_argument("--identity_csv", default=None, help="identity_baseline.csv from inspect_ct4h_cache.py")
    parser.add_argument("--identity_column", default="soft_mae_mean", help="soft_mae_mean (vs repeat mean) or soft_mae_repeat")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hu_window", nargs=2, type=float, default=[-160.0, 240.0], help="window the model was trained with")
    parser.add_argument("--cases", nargs="+", default=["B1:1:14", "E1:3:14", "E1:10:14", "B1:14:1"],
                        help="scanner:source:target for the model figure")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(Path(args.cache_dir))
    needs = {"model": args.checkpoint, "results": args.pairs_csv, "noise": args.floor_csv, "identity": args.identity_csv}
    wanted = args.only or [f for f in FIGURES if needs.get(f, True)]
    for name in wanted:
        print("making", name, flush=True)
        if name == "dose":
            fig_dose(cache, out, args.scanner, args.z, args.zoom)
        elif name == "scanner":
            fig_scanner(cache, out, args.z, 10)
        elif name == "repeat":
            fig_repeat(cache, out, args.scanner, args.z)
        elif name == "recon":
            fig_recon(cache, out, args.z, args.recon_dose, Path(args.reg_dir) if args.reg_dir else None)
        elif name == "model":
            if not args.checkpoint:
                sys.exit("model figure needs --checkpoint")
            cases = [(c.split(":")[0], int(c.split(":")[1]), int(c.split(":")[2])) for c in args.cases]
            fig_model(cache, out, args.checkpoint, cases, args.z, args.device, tuple(args.hu_window))
        elif name == "results":
            if not args.pairs_csv:
                sys.exit("results figure needs --pairs_csv")
            fig_results(Path(args.pairs_csv), out)
        elif name == "noise":
            if not args.floor_csv:
                sys.exit("noise figure needs --floor_csv")
            fig_noise(Path(args.floor_csv), out)
        elif name == "identity":
            if not args.identity_csv:
                sys.exit("identity figure needs --identity_csv")
            fig_identity(Path(args.identity_csv), out, args.identity_column)
    print("figures in", out)


if __name__ == "__main__":
    main()
