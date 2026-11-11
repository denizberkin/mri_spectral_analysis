import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiment-pipeline"))

from ldct.data.cached_dataset import DEFAULT_HU_WINDOWS, window_to_unit
from ldct.data.utils import ANATOMIES, NUM_DOMAINS, get_domain, patients_in, read_index
from ldct.env import get_ldct_preprocessed_dir

METRICS = ["rmse", "mae", "psnr", "ssim"]
LABELS = {"rmse": "RMSE (HU)", "mae": "MAE (HU)", "psnr": "PSNR (dB)", "ssim": "SSIM"}
LOWER_IS_BETTER = {"rmse", "mae"}


def slice_metrics(x: np.ndarray, target: np.ndarray, width: float) -> dict:
    """Metrics of x against target; both are images in the [0, 1] windowed scale."""
    diff = x - target
    mse = float(np.mean(diff**2))
    return {
        "rmse": float(np.sqrt(mse) * width),
        "mae": float(np.mean(np.abs(diff)) * width),
        "psnr": float(10 * np.log10(1.0 / max(mse, 1e-12))),
        "ssim": float(structural_similarity(x, target, data_range=1.0)),
    }


def load_model(path: str, device):
    import torch
    from eval_pipeline.core.importing import import_module_from_path

    module = import_module_from_path(REPO / "experiment-pipeline" / "components" / "models" / "conditional_unet.py")
    state = torch.load(path, map_location=device, weights_only=True)
    state = state.get("model", state)
    model = module.unet_from_state_dict(state).to(device).eval()
    return model, any(k.startswith("slice_embedding.") for k in state), int(state["target_embedding.weight"].shape[0])


def predict(model, uses_slice_pos, images: np.ndarray, positions: np.ndarray, source_domain, target_domain, device, batch_size):
    """images: (N, H, W) in [0, 1]. Returns the prediction in [0, 1], same shape."""
    import torch

    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(images[start : start + batch_size]).unsqueeze(1).to(device) * 2.0 - 1.0
            n = len(x)
            slice_pos = torch.from_numpy(positions[start : start + batch_size]).float().to(device) if uses_slice_pos else None
            y = model(x, torch.full((n,), target_domain, dtype=torch.long, device=device),
                      torch.full((n,), source_domain, dtype=torch.long, device=device), slice_pos)
            out.append(((y.squeeze(1).float().cpu().numpy() + 1.0) / 2.0).clip(0.0, 1.0))
    return np.concatenate(out)


def save_example(path: Path, low, pred, full, window, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = window
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, image, name in zip(axes[:3], (low, pred, full), ("low dose (input)", "model output", "full dose (target)")):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
    err = (pred - full) * (hi - lo)
    axes[3].imshow(err, cmap="coolwarm", vmin=-(hi - lo) / 4, vmax=(hi - lo) / 4)
    axes[3].set_title(f"output - target (+/-{(hi - lo) / 4:.0f} HU)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anatomy", required=True, choices=ANATOMIES)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=None, help="Model .pt file. Omit to score only the do-nothing input.")
    parser.add_argument("--preprocessed_dir", default=None, help="Default: LDCT_PREPROCESSED_DIR.")
    parser.add_argument("--hu_window", nargs=2, type=float, default=None, help="Default: the anatomy's training window.")
    parser.add_argument("--domain_scheme", default="dose", choices=sorted(NUM_DOMAINS))
    parser.add_argument("--stride", type=int, default=2, help="Score every N-th slice. Use 1 for final numbers.")
    parser.add_argument("--all_slices", action="store_true", help="Chest: do not restrict to slices with lung.")
    parser.add_argument("--lung_fraction", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples", type=int, default=0, help="Save a picture for the first N patients.")
    parser.add_argument("--output", default=None, help="Per-patient CSV. Default: {preprocessed_dir}/eval/...")
    args = parser.parse_args()

    out_dir = Path(args.preprocessed_dir or get_ldct_preprocessed_dir() or "")
    if not str(out_dir):
        sys.exit("Pass --preprocessed_dir or set LDCT_PREPROCESSED_DIR.")
    window = tuple(args.hu_window or DEFAULT_HU_WINDOWS[args.anatomy])
    width = window[1] - window[0]
    patients = patients_in(read_index(out_dir / "index.csv"), args.anatomy, args.split)
    if not patients:
        sys.exit(f"No {args.split} patients for {args.anatomy}.")

    model = device = None
    uses_slice_pos = False
    tag = "identity"
    if args.checkpoint:
        import torch

        device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        model, uses_slice_pos, n_domains = load_model(args.checkpoint, device)
        if n_domains != NUM_DOMAINS[args.domain_scheme]:
            print(f"warning: the checkpoint has {n_domains} domains but scheme {args.domain_scheme!r} has "
                  f"{NUM_DOMAINS[args.domain_scheme]}; check --domain_scheme")
        ckpt = Path(args.checkpoint)
        tag = f"{ckpt.parent.parent.name}_{ckpt.stem.split('_')[-1]}"
    src_dom = get_domain(args.anatomy, "low", args.domain_scheme)
    tgt_dom = get_domain(args.anatomy, "full", args.domain_scheme)
    lung_only = args.anatomy == "chest" and not args.all_slices

    base = out_dir / args.anatomy
    rows = []
    for patient in patients:
        files = sorted((base / "low").glob(f"{patient}_s*.npz"))[:: args.stride]
        low, full, pos, names = [], [], [], []
        for f in files:
            with np.load(f) as npz:
                low_hu, slice_pos = npz["image"], float(npz["slice_pos"])
            with np.load(base / "full" / f.name) as npz:
                full_hu = npz["image"]
            if lung_only and ((full_hu > -950) & (full_hu < -600)).mean() < args.lung_fraction:
                continue
            low.append(window_to_unit(low_hu, *window))
            full.append(window_to_unit(full_hu, *window))
            pos.append(slice_pos)
            names.append(f.stem)
        if not low:
            print(f"{patient}: no slices to score, skipped")
            continue
        low, full, pos = np.stack(low), np.stack(full), np.array(pos, dtype=np.float32)
        pred = predict(model, uses_slice_pos, low, pos, src_dom, tgt_dom, device, args.batch_size) if model else None
        row = {"patient_id": patient, "n_slices": len(low)}
        for i in range(len(low)):
            for metric, value in slice_metrics(low[i], full[i], width).items():
                row.setdefault(f"{metric}_input", []).append(value)
            if pred is not None:
                for metric, value in slice_metrics(pred[i], full[i], width).items():
                    row.setdefault(f"{metric}_model", []).append(value)
        rows.append({k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in row.items()})
        if pred is not None and len(rows) <= args.examples:
            mid = len(low) // 2
            save_example(out_dir / "eval" / "examples" / f"{tag}_{names[mid]}.png", low[mid], pred[mid], full[mid],
                         window, f"{args.anatomy} {names[mid]} ({tag})")

    if not rows:
        sys.exit(f"No {args.anatomy} {args.split} slices were scored"
                 + (f" (every patient was dropped by the lung filter; try --all_slices or a lower "
                    f"--lung_fraction than {args.lung_fraction})" if lung_only else ""))
    scored = sum(r["n_slices"] for r in rows)
    print(f"\n{args.anatomy} {args.split}: {len(rows)} patients, {scored} slices"
          f"{' (slices with lung only)' if lung_only else ''}, window {window}, tag {tag}")
    header = f"{'':12}{'input':>20}" + (f"{'model':>20}{'change':>12}" if model else "")
    print(header)
    for metric in METRICS:
        a = np.array([r[f"{metric}_input"] for r in rows])
        line = f"{LABELS[metric]:12}{a.mean():>12.3f} ({a.std():.3f})"
        if model:
            b = np.array([r[f"{metric}_model"] for r in rows])
            better = (b < a) if metric in LOWER_IS_BETTER else (b > a)
            line += f"{b.mean():>12.3f} ({b.std():.3f}){b.mean() - a.mean():>+12.3f}   better in {better.sum()}/{len(rows)}"
        print(line)
    print("(mean over patients, standard deviation across patients in brackets)")

    output = Path(args.output) if args.output else out_dir / "eval" / f"{tag}_{args.anatomy}_{args.split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
