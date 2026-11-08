"""How well do scans of the same phantom line up on one scanner?

Dose translation pairs a low-dose scan with a higher-dose scan of the same phantom. Those are
separate acquisitions, so before building pairs we need to know whether they already share a
grid or whether registration is needed. For every scanner, every selected series is compared
with that scanner's reference scan (highest selected dose, FBP, lowest selected repeat):

    hdr_z_off     offset of the first slice in the DICOM z coordinate, in slices
    img_z_off     whole-slice offset found from the image content instead, relative to the slice
                  the headers say is the match. 0 means the DICOM z coordinate can be trusted.
    z_ncc_best    correlation at that best slice, and z_ncc_next at its better neighbour. If the
                  two are almost equal the scans are probably off by a fraction of a slice (or
                  the phantom is smooth along z), so a sub-slice shift may be needed.
    shift_y/x_px  in-plane shift (pixels) that moves the series onto the reference;
                  shift_mm is its length in mm
    shift_spread  std of that shift across the checked slices (large = rotation, tilt, or noise)
    ncc           correlation with the reference after the shift (low = the numbers are unreliable)

IR/DL series are also compared with the FBP series of the SAME scan (the "twin"), which should
agree exactly: that tests the assumption that the reconstructions of one scan share a grid.

The thresholds are not calibrated on real data: read the first run as a calibration run.
Only a few slices per series are read; all headers are read (fast).

Usage:
    python scripts/check_ct4h_alignment.py --index /path/ct4h_index.csv
    python scripts/check_ct4h_alignment.py --scanners A1 G1 --doses 1 14 --repeats 1 2 3 4 5
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct4h.data.utils import DOSES, HU_RANGE, RECONS, SCANNERS, read_index, select

CLIP = HU_RANGE  # same range as the dataset authors' code
# Gaussian smoothing (px) applied to a COPY of each slice, only to estimate the shift: at 1 mGy the noise would
# otherwise drive the correlation. Nothing smoothed is saved; 0 turns it off. See --smooth_sigma.
SMOOTH_SIGMA = 1.5
COLUMNS = [
    "scanner", "dose_mgy", "recon", "repeat", "vs", "n_slices", "hdr_z_off", "img_z_off",
    "z_ncc_best", "z_ncc_next", "shift_y_px", "shift_x_px", "shift_mm", "shift_spread_px", "ncc", "slices_used", "error",
]


def read_headers(series_dir: Path) -> dict:
    """z-sorted slice paths and positions of one series (no pixel data)."""
    entries = []
    for path in series_dir.iterdir():
        if path.suffix.lower() != ".dcm":
            continue
        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["ImagePositionPatient", "PixelSpacing"])
        entries.append((float(ds.ImagePositionPatient[2]), path, float(ds.PixelSpacing[0])))
    if not entries:
        raise FileNotFoundError(f"No .dcm files in {series_dir}")
    entries.sort(key=lambda e: e[0])
    z = np.array([e[0] for e in entries])
    return {
        "z": z,
        "paths": [e[1] for e in entries],
        "spacing": float(np.median(np.diff(z))) if len(z) > 1 else float("nan"),
        "pixel_mm": entries[0][2],
    }


def load_slice(path: Path, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    ds = pydicom.dcmread(path)
    hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0)) + float(
        getattr(ds, "RescaleIntercept", 0.0)
    )
    hu = np.clip(hu, *CLIP)
    return ndimage.gaussian_filter(hu, sigma) if sigma > 0 else hu


def register(ref: np.ndarray, mov: np.ndarray) -> tuple[np.ndarray, float]:
    """(dy, dx) that moves ``mov`` onto ``ref``, and the correlation after that shift."""
    try:
        shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10, normalization=None)
    except TypeError:  # older scikit-image has no normalization argument
        shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10)
    moved = ndimage.shift(mov, shift, order=1, mode="nearest")
    return np.asarray(shift, dtype=float), float(np.corrcoef(ref.ravel(), moved.ravel())[0, 1])


def compare(ref_hdr: dict, hdr: dict, ref_cache: dict, n_check: int, window: int, sigma: float = SMOOTH_SIGMA) -> dict:
    """Header and image-based offsets of one series against a reference series."""
    n_ref, n = len(ref_hdr["z"]), len(hdr["z"])
    wanted = np.unique(np.round(np.linspace(0.3, 0.7, n_check) * (n_ref - 1)).astype(int))

    def ref_slice(i: int) -> np.ndarray:
        if i not in ref_cache:
            ref_cache[i] = load_slice(ref_hdr["paths"][i], sigma)
        return ref_cache[i]

    # the most informative reference slice: highest std (empty slices correlate with everything)
    anchor = int(max(wanted, key=lambda i: float(ref_slice(i).std())))
    nearest = lambda i: int(np.argmin(np.abs(hdr["z"] - ref_hdr["z"][i])))  # noqa: E731
    j0 = nearest(anchor)

    offsets = [o for o in range(-window, window + 1) if 0 <= j0 + o < n]
    scores = []
    for o in offsets:
        scores.append(register(ref_slice(anchor), load_slice(hdr["paths"][j0 + o], sigma))[1])
    best = int(np.argmax(scores))
    shift_offset = offsets[best]
    neighbours = [scores[k] for k in (best - 1, best + 1) if 0 <= k < len(scores)]
    z_next = max(neighbours) if neighbours else float("nan")

    shifts, nccs = [], []
    for i in wanted:
        j = nearest(i) + shift_offset
        if not 0 <= j < n:
            continue
        shift, ncc = register(ref_slice(int(i)), load_slice(hdr["paths"][j], sigma))
        shifts.append(shift)
        nccs.append(ncc)
    if not shifts:
        raise ValueError("no overlapping slices to compare")
    shifts = np.array(shifts)
    median = np.median(shifts, axis=0)
    return {
        "n_slices": n,
        "hdr_z_off": float((hdr["z"][0] - ref_hdr["z"][0]) / ref_hdr["spacing"]),
        "img_z_off": shift_offset,
        "z_ncc_best": float(scores[best]),
        "z_ncc_next": float(z_next),
        "shift_y_px": float(median[0]),
        "shift_x_px": float(median[1]),
        "shift_mm": float(np.hypot(*median) * hdr["pixel_mm"]),
        "shift_spread_px": float(np.std(np.hypot(shifts[:, 0], shifts[:, 1]))),
        "ncc": float(np.median(nccs)),
        "slices_used": len(shifts),
    }


def process_scanner(task: dict) -> list[dict]:
    scanner, data_dir = task["scanner"], Path(task["data_dir"])
    rows = task["rows"]
    by_key = {(r["dose_mgy"], r["recon"], r["repeat"]): r for r in rows}
    fbp = [r for r in rows if r["recon"] == "FBP"]
    if not fbp:
        return [{"scanner": scanner, "error": "no FBP series selected, cannot pick a reference"}]
    ref = min(fbp, key=lambda r: (-r["dose_mgy"], r["repeat"]))
    headers: dict[str, dict] = {}

    def hdr_of(row: dict) -> dict:
        key = row["series_uid"]
        if key not in headers:
            headers[key] = read_headers(data_dir / row["series_dir"])
        return headers[key]

    ref_hdr, ref_cache = hdr_of(ref), {}
    out = []
    for row in sorted(rows, key=lambda r: (r["dose_mgy"], r["recon"], r["repeat"])):
        base = {"scanner": scanner, "dose_mgy": row["dose_mgy"], "recon": row["recon"], "repeat": row["repeat"]}
        comparisons = []
        if row is not ref:
            comparisons.append(("ref", ref_hdr, ref_cache))
        if row["recon"] != "FBP":
            twin = by_key.get((row["dose_mgy"], "FBP", row["repeat"]))
            if twin is not None:
                comparisons.append(("twin", hdr_of(twin), {}))
        for label, other_hdr, cache in comparisons:
            try:
                result = compare(other_hdr, hdr_of(row), cache, task["n_check"], task["window"], task["sigma"])
                out.append({**base, "vs": label, **result, "error": ""})
            except Exception as error:  # report per series, keep going
                out.append({**base, "vs": label, "error": f"{type(error).__name__}: {error}"})
    return out


def summarize(results: list[dict]) -> None:
    ok = [r for r in results if not r.get("error") and "shift_mm" in r]
    print(f"\n{len(ok)} comparisons ok, {len(results) - len(ok)} failed")

    def table(label: str) -> None:
        print(f"\n--- vs {label}: per scanner (median / max over series)")
        print(f"{'scanner':8}{'n':>4}{'shift_mm':>16}{'|img_z_off|':>16}{'|hdr_z_off|':>16}{'ncc_min':>9}")
        for scanner in SCANNERS:
            rows = [r for r in ok if r["scanner"] == scanner and r["vs"] == label]
            if not rows:
                continue
            shift = [r["shift_mm"] for r in rows]
            iz = [abs(r["img_z_off"]) for r in rows]
            hz = [abs(r["hdr_z_off"]) for r in rows]
            print(
                f"{scanner:8}{len(rows):>4}{np.median(shift):>8.2f}/{max(shift):<7.2f}"
                f"{np.median(iz):>8.2f}/{max(iz):<7.2f}{np.median(hz):>8.2f}/{max(hz):<7.2f}"
                f"{min(r['ncc'] for r in rows):>9.3f}"
            )

    table("ref")
    table("twin")
    low_ncc = [r for r in ok if r["ncc"] < 0.5]
    if low_ncc:
        print(f"\n{len(low_ncc)} comparisons with ncc < 0.5 (shift unreliable); first 10:")
        for r in low_ncc[:10]:
            print(f"  {r['scanner']} {r['dose_mgy']}mGy {r['recon']} #{r['repeat']} vs {r['vs']}: ncc={r['ncc']:.2f}")
    for r in [r for r in results if r.get("error")][:10]:
        print(f"  FAILED {r.get('scanner')} {r.get('dose_mgy')} {r.get('recon')} #{r.get('repeat')}: {r['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="ct4h_index.csv")
    parser.add_argument("--data_dir", default=None, help="DICOM root the index paths are relative to. Default: CT4H_DIR.")
    parser.add_argument("--out", default="ct4h_alignment.csv")
    parser.add_argument("--scanners", nargs="+", choices=SCANNERS, default=SCANNERS)
    parser.add_argument("--doses", nargs="+", type=int, choices=DOSES, default=[1, 6, 14])
    parser.add_argument("--recons", nargs="+", choices=RECONS, default=["FBP", "IR"])
    parser.add_argument("--repeats", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n_check", type=int, default=5, help="slices per series for the in-plane shift.")
    parser.add_argument("--window", type=int, default=6, help="+/- slices searched for the image-based z offset (not an HU window).")
    parser.add_argument("--smooth_sigma", type=float, default=SMOOTH_SIGMA,
                        help="Gaussian sigma (px) on a copy of each slice used only for the shift estimate; 0 = none.")
    parser.add_argument("--workers", type=int, default=4, help="processes, one scanner each.")
    args = parser.parse_args()

    if args.data_dir is None:
        from ct4h.env import get_ct4h_dir

        args.data_dir = get_ct4h_dir()
    rows = select(read_index(args.index), scanners=args.scanners, doses=args.doses,
                  recons=args.recons, repeats=args.repeats)
    if not rows:
        sys.exit("No series match the selection.")
    tasks = [
        {"scanner": s, "rows": [r for r in rows if r["scanner"] == s], "data_dir": args.data_dir,
         "n_check": args.n_check, "window": args.window, "sigma": args.smooth_sigma}
        for s in args.scanners if any(r["scanner"] == s for r in rows)
    ]
    print(f"{len(rows)} series on {len(tasks)} scanners")

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for chunk in tqdm(pool.map(process_scanner, tasks), total=len(tasks), desc="scanners"):
            results.extend(chunk)

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {args.out}")
    summarize(results)


if __name__ == "__main__":
    main()
