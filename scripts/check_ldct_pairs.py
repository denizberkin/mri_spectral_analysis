"""Sanity-check the low/full slice pairs written by preprocess_ldct.py.

Reads the npz files only (no DICOM). Per paired patient, on every ``--stride``-th slice:

    diff_mean / diff_std   low - full over soft tissue (HU). The mean should be ~0; the std
                           is the simulated noise, so it should be alike across patients.
    corr                   correlation of low and full over soft tissue.
    shift_ok               fraction of slices where the low image fits the full image best
                           with no pixel shift (tests the in-plane alignment).
    slice_ok               fraction of slices where the low image fits its own full slice
                           better than the slices on either side (tests the slice pairing).

Patients whose shift_ok or slice_ok is low, or whose diff statistics are far from the other
patients of the same anatomy, are listed at the end. The thresholds are not calibrated on
real data: read the first run as a calibration run and look at the distribution.

Usage:
    python scripts/check_ldct_pairs.py
    python scripts/check_ldct_pairs.py --stride 5 --workers 8
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldct.data.utils import ANATOMIES, read_index, slice_name
from ldct.env import get_ldct_preprocessed_dir

CLIP = (-1024.0, 1024.0)
TISSUE = (-300.0, 1000.0)  # soft tissue and bone; air and lung are noisy edges and are left out
SHIFTS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def _load(folder: Path, patient: str, idx: int) -> np.ndarray:
    with np.load(folder / f"{slice_name(patient, idx)}.npz") as npz:
        return npz["image"].astype(np.float32)


def check_patient(task: dict) -> dict:
    anatomy, patient, stride = task["anatomy"], task["patient_id"], task["stride"]
    base = Path(task["preprocessed_dir"]) / anatomy
    try:
        n = len(list((base / "full").glob(f"{patient}_s*.npz")))
        n_low = len(list((base / "low").glob(f"{patient}_s*.npz")))
        if n != n_low or n < 3:
            raise ValueError(f"{n} full and {n_low} low slices")
        stats = {"diff_mean": [], "diff_std": [], "corr": [], "shift_ok": [], "slice_ok": []}
        for i in range(1, n - 1, stride):
            low, full = _load(base / "low", patient, i), _load(base / "full", patient, i)
            # Select on the full-dose image only. Selecting on the noisy low-dose image as well
            # would cut off its noise tails and bias the difference statistics.
            mask = (full > TISSUE[0]) & (full < TISSUE[1])
            if mask.mean() < 0.02:
                continue  # a slice at the end of the scan with little tissue
            diff = (low - full)[mask]
            stats["diff_mean"].append(diff.mean())
            stats["diff_std"].append(diff.std())
            both = low[mask].std() > 0 and full[mask].std() > 0
            stats["corr"].append(np.corrcoef(low[mask], full[mask])[0, 1] if both else np.nan)

            lo, fu = np.clip(low, *CLIP), np.clip(full, *CLIP)
            h, w = lo.shape
            core = lo[1:-1, 1:-1]
            mad = {(dy, dx): np.abs(core - fu[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]).mean() for dy, dx in SHIFTS}
            stats["shift_ok"].append(min(mad, key=mad.get) == (0, 0))

            same = np.abs(lo - fu).mean()
            neighbours = [np.abs(lo - np.clip(_load(base / "full", patient, j), *CLIP)).mean() for j in (i - 1, i + 1)]
            stats["slice_ok"].append(same < min(neighbours))
        if not stats["diff_std"]:
            raise ValueError("no slice with enough tissue to compare")
        return {"anatomy": anatomy, "patient_id": patient, "n_slices": n, "n_checked": len(stats["diff_std"]),
                **{k: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for k, v in stats.items()}}
    except Exception as error:
        return {"anatomy": anatomy, "patient_id": patient, "error": f"{type(error).__name__}: {error}"}


def robust_outliers(values: np.ndarray, z: float, floor: float) -> np.ndarray:
    """Values more than z robust standard deviations from the median.

    ``floor`` is the smallest spread worth caring about, so that near-identical patients do
    not turn a negligible difference into an outlier.
    """
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values), dtype=bool)
    median = np.median(values[finite])
    spread = max(1.4826 * np.median(np.abs(values[finite] - median)), floor)
    return finite & (np.abs(values - median) > z * spread)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preprocessed_dir", default=None, help="Default: LDCT_PREPROCESSED_DIR.")
    parser.add_argument("--anatomies", nargs="+", choices=ANATOMIES, default=ANATOMIES)
    parser.add_argument("--stride", type=int, default=10, help="Check every N-th slice.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min_ok", type=float, default=0.8, help="Flag a patient below this shift_ok or slice_ok.")
    parser.add_argument("--z", type=float, default=5.0, help="Robust z-score for outliers in diff_mean / diff_std / corr.")
    args = parser.parse_args()

    out = Path(args.preprocessed_dir or get_ldct_preprocessed_dir() or "")
    if not str(out):
        sys.exit("Pass --preprocessed_dir or set LDCT_PREPROCESSED_DIR.")
    rows = read_index(out / "index.csv")
    done = {p.stem for p in (out / "_done").glob("*.json")}
    tasks = sorted(
        ({"anatomy": r["anatomy"], "patient_id": r["patient_id"], "stride": args.stride, "preprocessed_dir": str(out)}
         for r in rows if r["paired"] and r["dose"] == "full" and r["anatomy"] in args.anatomies and r["patient_id"] in done),
        key=lambda t: t["patient_id"],
    )
    print(f"checking {len(tasks)} patients (every {args.stride}th slice)")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(check_patient, tasks), total=len(tasks), desc="patients"))

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    fields = ["anatomy", "patient_id", "n_slices", "n_checked", "diff_mean", "diff_std", "corr", "shift_ok", "slice_ok"]
    with open(out / "pair_check.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in ok])

    flagged: dict[str, list[str]] = {}
    for r in errors:
        flagged.setdefault(r["patient_id"], []).append(r["error"])
    print("\nper anatomy (median over patients):")
    for anatomy in args.anatomies:
        group = [r for r in ok if r["anatomy"] == anatomy]
        if not group:
            continue
        print(f"  {anatomy}: " + ", ".join(f"{k}={np.nanmedian([r[k] for r in group]):.3f}" for k in fields[4:]))
        for key, floor in (("diff_mean", 1.0), ("diff_std", 0.05), ("corr", 0.01)):
            values = np.array([r[key] for r in group])
            if key == "diff_std":
                floor *= np.nanmedian(values)  # 5% of the typical noise level
            for r, bad in zip(group, robust_outliers(values, args.z, floor)):
                if bad:
                    flagged.setdefault(r["patient_id"], []).append(f"{key}={r[key]:.3f} is far from the other {anatomy} patients")
        for r in group:
            for key in ("shift_ok", "slice_ok"):
                if r[key] < args.min_ok:
                    flagged.setdefault(r["patient_id"], []).append(f"{key}={r[key]:.2f}")
    print(f"\nwrote {out / 'pair_check.csv'}")
    if flagged:
        print(f"\n{len(flagged)} patients to look at:")
        for patient in sorted(flagged):
            print(f"  {patient}: " + "; ".join(flagged[patient]))
        sys.exit(1)
    print("no patient flagged")


if __name__ == "__main__":
    main()
