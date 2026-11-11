#!/usr/bin/env python
"""Score a dose-to-dose checkpoint, or the "return the source" baseline, on the CT4H validation/test split.

For every dose pair it compares, against the target-dose image of the same scanner and slice:
    identity   the source slice itself (what a model has to beat)
    model      the checkpoint's prediction from the source (needs --checkpoint)
against two targets:
    one noisy repeat   what training sees; the floor is the target's own noise
    repeat mean        the average of every repeat of the target dose in the split (low-noise reference)
Everything is computed inside --hu_window (clipped to it, data range = window width), which must be the window
the model was trained with. MAE is in HU, PSNR uses the window width as the peak. Draws are deterministic and
identical to scripts/inspect_ct4h_cache.py (same seed), so the identity columns match its soft-window columns.

    python scripts/evaluate_ct4h.py --split val                                        # baseline only
    python scripts/evaluate_ct4h.py --split val --checkpoint experiment-pipeline/runs/ct4h_smoke/artifacts/task3_unet_finetune_2.pt
    python scripts/evaluate_ct4h.py --split test --split_mode repeat --checkpoint ... --ssim
Up and down pairs are summarised separately: a deterministic regressor cannot add noise, so the down pairs
mostly measure how well it removes it. The seen/unseen-scanner comparison is --split_mode repeat vs scanner.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiment-pipeline"))

from ct4h.data.cached_dataset import CT4HPairDataset  # noqa: E402
from ct4h.data.utils import DOSES, NUM_DOMAINS, SCANNER_SPLIT, dose_pairs  # noqa: E402


# -- metrics (numpy only) ------------------------------------------------------------------------------------
def image_metrics(x: np.ndarray, target: np.ndarray, ssim: bool) -> dict[str, float]:
    """x, target in the [0, 1] windowed scale."""
    mse = float(np.mean((x - target) ** 2))
    out = {"mae": float(np.mean(np.abs(x - target))), "psnr": float(10 * np.log10(1.0 / max(mse, 1e-12)))}
    if ssim:
        from skimage.metrics import structural_similarity

        out["ssim"] = float(structural_similarity(x, target, data_range=1.0))
    return out


def batches(dataset, batch_size: int):
    """Yield dicts of numpy arrays (and lists for strings) over a deterministic dataset."""
    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = {}
        for key in items[0]:
            values = [item[key] for item in items]
            batch[key] = [v.numpy() if hasattr(v, "numpy") else v for v in values]
        yield batch


def evaluate_dataset(dataset, predict, batch_size: int, ssim: bool, width: float) -> list[dict]:
    """One record per item: scanner plus identity/model metrics (MAE converted to HU)."""
    records = []
    for batch in batches(dataset, batch_size):
        predictions = predict(batch) if predict is not None else None
        for i in range(len(batch["scanner"])):
            h, w = int(batch["orig_height"][i]), int(batch["orig_width"][i])
            target = (batch["target"][i][0, :h, :w] + 1) / 2
            record = {"scanner": batch["scanner"][i]}
            sets = {"identity": (batch["source"][i][0, :h, :w] + 1) / 2}
            if predictions is not None:
                sets["model"] = (predictions[i][:h, :w] + 1) / 2
            for name, image in sets.items():
                for metric, value in image_metrics(image, target, ssim).items():
                    record[f"{name}_{metric}"] = value * width if metric == "mae" else value
            records.append(record)
    return records


def mean_of(records: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in records])) if records else float("nan")


# -- model (torch) -------------------------------------------------------------------------------------------
def load_model(path: str, device):
    import torch
    from eval_pipeline.core.importing import import_module_from_path

    module = import_module_from_path(REPO / "experiment-pipeline" / "components" / "models" / "conditional_unet.py")
    state = torch.load(path, map_location=device, weights_only=True)
    state = state.get("model", state)
    model = module.unet_from_state_dict(state).to(device).eval()
    return model, any(k.startswith("slice_embedding.") for k in state)


def make_predict(model, uses_slice_pos: bool, device, force_target_dose: int | None = None):
    import torch

    def predict(batch: dict) -> np.ndarray:
        source = torch.from_numpy(np.stack(batch["source"])).to(device)
        target_domain = torch.as_tensor(batch["target_domain"], dtype=torch.long, device=device)
        if force_target_dose is not None:
            target_domain = torch.full_like(target_domain, DOSES.index(force_target_dose))
        source_domain = torch.as_tensor(batch["source_domain"], dtype=torch.long, device=device)
        slice_pos = torch.as_tensor(np.stack(batch["slice_pos"]), dtype=torch.float32, device=device) if uses_slice_pos else None
        with torch.no_grad():
            out = model(source, target_domain, source_domain, slice_pos)
        return out.float().cpu().numpy()[:, 0]

    return predict


# -- reporting -----------------------------------------------------------------------------------------------
def gain(identity: float, model: float) -> float:
    return 100.0 * (identity - model) / identity if identity else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="omit to score the identity baseline only")
    parser.add_argument("--cache_dir", default=None, help="default: CT4H_CACHE_DIR")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--split_mode", default="scanner", choices=["scanner", "repeat"])
    parser.add_argument("--directions", default="any", choices=["any", "up", "down"])
    parser.add_argument("--recons", nargs="+", default=["FBP"])
    parser.add_argument("--scanners", nargs="+", default=None, help="override the split's scanners, e.g. G2 C1")
    parser.add_argument("--repeats", nargs="+", type=int, default=None, help="override the split's repeats, e.g. 9 10")
    parser.add_argument("--force_target_dose", type=int, default=None, choices=DOSES,
                        help="give the model this target dose whatever the real target is (tests whether the "
                             "dose conditioning is used: error barely moving means it is not)")
    parser.add_argument("--hu_window", nargs=2, type=float, default=[-160.0, 240.0], metavar=("LOW", "HIGH"),
                        help="the window the model was trained with")
    parser.add_argument("--samples", type=int, default=100, help="slices per dose pair")
    parser.add_argument("--z_lo", type=int, default=11)
    parser.add_argument("--z_hi", type=int, default=304)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=5, help="same default as scripts/inspect_ct4h_cache.py")
    parser.add_argument("--domain_scheme", default="dose")
    parser.add_argument("--ssim", action="store_true", help="also report SSIM (slower)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out_dir", default=None, help="default: next to the checkpoint, else the current folder")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if not cache_dir:
        from ct4h.env import get_ct4h_cache_dir

        cache_dir = get_ct4h_cache_dir()
    if not cache_dir:
        sys.exit("set CT4H_CACHE_DIR or pass --cache_dir")

    predict, neighbours = None, 1
    if args.checkpoint:
        import torch

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, uses_slice_pos = load_model(args.checkpoint, device)
        channels = int(model.input_channels)
        if channels % 2 != 1:
            sys.exit(f"model has {channels} input channels; expected 1 + 2 * neighbours")
        neighbours = (channels - 1) // 2
        domains = int(model.target_embedding.weight.shape[0])
        if domains != NUM_DOMAINS[args.domain_scheme]:
            sys.exit(f"checkpoint has {domains} domains, domain_scheme {args.domain_scheme!r} needs "
                     f"{NUM_DOMAINS[args.domain_scheme]}")
        if args.force_target_dose is not None and args.domain_scheme != "dose":
            sys.exit("--force_target_dose needs domain_scheme dose")
        predict = make_predict(model, uses_slice_pos, device, args.force_target_dose)
        print(f"checkpoint {args.checkpoint}: {channels} input channels (neighbours={neighbours}), {domains} domains")
    else:
        uses_slice_pos = False
        print("no checkpoint: identity baseline only")

    low, high = args.hu_window
    width = high - low
    z_range = (args.z_lo, args.z_hi)
    scanners = SCANNER_SPLIT[args.split] if args.split_mode == "scanner" else "all scanners"
    if args.scanners:
        scanners = args.scanners
    repeats = f", repeats {args.repeats}" if args.repeats else ""
    forced = f", model always told target {args.force_target_dose} mGy" if args.force_target_dose else ""
    print(f"split={args.split} ({args.split_mode}: {scanners}{repeats}){forced}, window {low:.0f}..{high:.0f} HU, "
          f"{args.samples} slices per dose pair, z {z_range[0]}..{z_range[1]}")

    results: dict[tuple, list[dict]] = {}          # (source, target, mode) -> records
    for source_dose, target_dose in dose_pairs(args.directions):
        direction = "up" if target_dose > source_dose else "down"
        for mode in ("repeat", "mean"):
            dataset = CT4HPairDataset(
                cache_dir, args.split, args.split_mode, directions=direction, doses=[source_dose, target_dose],
                recons=args.recons, neighbours=neighbours, hu_window=(low, high), target_mode=mode,
                domain_scheme=args.domain_scheme, pad_multiple=16, z_range=z_range,
                include_slice_pos=uses_slice_pos, scanners=args.scanners, repeats=args.repeats, length=args.samples, deterministic=True, seed=args.seed,
            )
            results[(source_dose, target_dose, mode)] = evaluate_dataset(
                dataset, predict, args.batch_size, args.ssim, width)

    has_model = predict is not None
    metrics = ["mae", "psnr"] + (["ssim"] if args.ssim else [])
    pair_rows, summary = [], defaultdict(list)
    header = f"{'pair':>8s} {'dir':>5s} | " + " | ".join(
        f"{'vs ' + m:>9s} {'identity':>8s}" + (f" {'model':>8s} {'gain%':>7s}" if has_model else "") for m in ("repeat", "mean"))
    print("\nMAE in HU inside the window; gain% = how much of the identity error the model removes")
    print(header + "\n" + "-" * len(header))
    for source_dose, target_dose in dose_pairs(args.directions):
        direction = "up" if target_dose > source_dose else "down"
        cells, row = [], {"source_mgy": source_dose, "target_mgy": target_dose, "direction": direction}
        for mode in ("repeat", "mean"):
            records = results[(source_dose, target_dose, mode)]
            ident = mean_of(records, "identity_mae")
            row[f"{mode}_identity_mae"] = ident
            cell = f"{'':>9s} {ident:8.2f}"
            summary[(direction, mode, "identity")].append(ident)
            if has_model:
                model_mae = mean_of(records, "model_mae")
                row[f"{mode}_model_mae"] = model_mae
                row[f"{mode}_gain_pct"] = gain(ident, model_mae)
                cell += f" {model_mae:8.2f} {gain(ident, model_mae):7.1f}"
                summary[(direction, mode, "model")].append(model_mae)
            for metric in metrics[1:]:
                for name in (["identity", "model"] if has_model else ["identity"]):
                    row[f"{mode}_{name}_{metric}"] = mean_of(records, f"{name}_{metric}")
            cells.append(cell)
        pair_rows.append(row)
        print(f"{f'{source_dose}->{target_dose}':>8s} {direction:>5s} | " + " | ".join(cells))

    print("\nsummary (MAE in HU, every dose pair weighted equally)")
    for direction in ("up", "down"):
        for mode in ("repeat", "mean"):
            ident = summary.get((direction, mode, "identity"))
            if not ident:
                continue
            text = f"  {direction:>5s} vs {mode:>6s}: identity {np.mean(ident):7.2f}"
            if has_model:
                model_values = summary[(direction, mode, "model")]
                text += f"  model {np.mean(model_values):7.2f}  gain {gain(np.mean(ident), np.mean(model_values)):5.1f}%"
            print(text)

    # per scanner, against the repeat mean, split by direction
    by_scanner: dict[tuple, list[dict]] = defaultdict(list)
    for (source_dose, target_dose, mode), records in results.items():
        if mode == "mean":
            direction = "up" if target_dose > source_dose else "down"
            for record in records:
                by_scanner[(record["scanner"], direction)].append(record)
    scanner_rows = []
    print("\nper scanner, vs repeat mean (MAE in HU)")
    for (scanner, direction), records in sorted(by_scanner.items()):
        row = {"scanner": scanner, "direction": direction, "n": len(records),
               "identity_mae": mean_of(records, "identity_mae")}
        text = f"  {scanner:>3s} {direction:>5s}  identity {row['identity_mae']:7.2f}"
        if has_model:
            row["model_mae"] = mean_of(records, "model_mae")
            row["gain_pct"] = gain(row["identity_mae"], row["model_mae"])
            text += f"  model {row['model_mae']:7.2f}  gain {row['gain_pct']:5.1f}%"
        scanner_rows.append(row)
        print(text)

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.checkpoint).parent if args.checkpoint else Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_{args.split_mode}" + ("_" + "".join(args.scanners) if args.scanners else "") \
        + ("_rep" + "-".join(map(str, args.repeats)) if args.repeats else "") \
        + (f"_force{args.force_target_dose}" if args.force_target_dose else "") + (f"_{Path(args.checkpoint).stem}" if args.checkpoint else "_identity")
    for name, rows in (("pairs", pair_rows), ("scanners", scanner_rows)):
        path = out_dir / f"ct4h_eval_{name}_{tag}.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
