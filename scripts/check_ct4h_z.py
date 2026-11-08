"""Sub-slice z alignment of CT4H scans from per-slice mean profiles.

check_ct4h_alignment.py finds the z offset by correlating a single anchor slice. On a phantom that
barely changes along z that is a weak signal (the best and the next slice score almost the same),
so its image-based z offset cannot be trusted at the 1-slice level. This script uses the whole
volume instead: for every slice it takes the mean HU, which averages away the noise, and slides
that 1-D profile along z until it matches the reference scan's profile.

    dz_mm        sample the series at (z + dz_mm) to land on the reference at z. The DICOM z
                 coordinates are already taken into account, so this is the *residual* phantom
                 displacement, not the table offset. 0 = the headers are right.
    hdr_start_mm first-slice z of the series minus the reference's (for information)
    rms_best     std of the profile difference at the best offset (HU). Noise level of the match.
    rms_zero     the same at dz = 0, i.e. what trusting the headers would cost.
    sharp        min(rms at dz-1mm, rms at dz+1mm) / rms_best. Near 1 = a flat minimum, the
                 estimate is poorly determined; clearly above 1 (say > 2) = well determined.

Only FBP is read by default: the twin test showed that FBP/IR/DL of one scan share a grid.

Usage:
    python scripts/check_ct4h_z.py --index /path/ct4h_index.csv
    python scripts/check_ct4h_z.py --scanners E2 F1 --repeats 1 2 3 4 5 6 7 8 9 10
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct4h.data.utils import DOSES, HU_RANGE, RECONS, SCANNERS, read_index, select

CLIP = HU_RANGE  # same range as the dataset authors' code
GRID_MM = 0.1
COLUMNS = [
    "scanner", "dose_mgy", "recon", "repeat", "n_slices", "spacing_mm", "hdr_start_mm",
    "dz_mm", "rms_best", "rms_zero", "sharp", "overlap_frac", "error",
]


def read_profile(series_dir: Path) -> dict:
    """z positions (mm, sorted) and mean HU per slice of one series."""
    entries = []
    for path in series_dir.iterdir():
        if path.suffix.lower() != ".dcm":
            continue
        ds = pydicom.dcmread(path)
        hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0)) + float(
            getattr(ds, "RescaleIntercept", 0.0)
        )
        entries.append((float(ds.ImagePositionPatient[2]), float(np.clip(hu, *CLIP).mean())))
    if len(entries) < 3:
        raise FileNotFoundError(f"fewer than 3 slices in {series_dir}")
    entries.sort()
    z = np.array([e[0] for e in entries])
    return {"z": z, "p": np.array([e[1] for e in entries]), "spacing": float(np.median(np.diff(z)))}


def rms_at(ref: dict, mov: dict, d: float) -> tuple[float, float]:
    """std of (ref(z) - mov(z + d)) over the part of ref that lands inside mov, and that overlap fraction."""
    z = ref["z"] + d
    inside = (z >= mov["z"][0]) & (z <= mov["z"][-1])
    if inside.sum() < 5:
        return float("inf"), 0.0
    diff = ref["p"][inside] - np.interp(z[inside], mov["z"], mov["p"])
    return float(np.std(diff)), float(inside.mean())


def estimate_dz(ref: dict, mov: dict, max_shift_mm: float, min_overlap: float = 0.5) -> dict:
    grid = np.arange(-max_shift_mm, max_shift_mm + GRID_MM / 2, GRID_MM)
    scores = []
    for d in grid:
        rms, overlap = rms_at(ref, mov, d)
        scores.append(rms if overlap >= min_overlap else float("inf"))
    scores = np.array(scores)
    if not np.isfinite(scores).any():
        raise ValueError("no offset with enough overlap")
    k = int(np.argmin(scores))
    best = float(grid[k])
    if 0 < k < len(grid) - 1 and np.isfinite(scores[k - 1]) and np.isfinite(scores[k + 1]):
        a, b, c = scores[k - 1], scores[k], scores[k + 1]
        denom = a - 2 * b + c
        if denom > 0:
            best += float(GRID_MM * 0.5 * (a - c) / denom)
    rms_best, overlap = rms_at(ref, mov, best)
    rms_zero, _ = rms_at(ref, mov, 0.0)
    around = [rms_at(ref, mov, best + s)[0] for s in (-1.0, 1.0)]
    return {
        "dz_mm": best, "rms_best": rms_best, "rms_zero": rms_zero,
        "sharp": float(min(around) / rms_best) if rms_best > 0 else float("inf"), "overlap_frac": overlap,
        "at_edge": bool(k in (0, len(grid) - 1)),
    }


def process_scanner(task: dict) -> list[dict]:
    scanner, data_dir = task["scanner"], Path(task["data_dir"])
    rows = [r for r in task["rows"] if r["recon"] == "FBP"] if task["fbp_only"] else task["rows"]
    fbp = [r for r in rows if r["recon"] == "FBP"]
    if not fbp:
        return [{"scanner": scanner, "error": "no FBP series selected, cannot pick a reference"}]
    ref = min(fbp, key=lambda r: (-r["dose_mgy"], r["repeat"]))
    ref_profile = read_profile(data_dir / ref["series_dir"])
    out = []
    for row in sorted(rows, key=lambda r: (r["dose_mgy"], r["recon"], r["repeat"])):
        base = {"scanner": scanner, "dose_mgy": row["dose_mgy"], "recon": row["recon"], "repeat": row["repeat"]}
        if row is ref:
            out.append({**base, "n_slices": len(ref_profile["z"]), "spacing_mm": ref_profile["spacing"],
                        "hdr_start_mm": 0.0, "dz_mm": 0.0, "error": "reference"})
            continue
        try:
            profile = read_profile(data_dir / row["series_dir"])
            result = estimate_dz(ref_profile, profile, task["max_shift_mm"])
            note = "search window edge: dz may be larger than --max_shift_mm" if result.pop("at_edge") else ""
            out.append({**base, "n_slices": len(profile["z"]), "spacing_mm": profile["spacing"],
                        "hdr_start_mm": float(profile["z"][0] - ref_profile["z"][0]), **result, "error": note})
        except Exception as error:  # report per series, keep going
            out.append({**base, "error": f"{type(error).__name__}: {error}"})
    return out


def summarize(results: list[dict]) -> None:
    ok = [r for r in results if "rms_best" in r]
    print(f"\n{len(ok)} series compared, {sum(1 for r in results if r.get('error') and r['error'] != 'reference')} with a note or error")
    print("\n--- dz_mm per scanner and dose (median / max |.| / spread over repeats); rms in HU")
    print(f"{'scanner':8}{'dose':>5}{'n':>4}{'dz_med':>9}{'|dz|max':>9}{'std':>7}{'rms_best':>10}{'rms_zero':>10}{'sharp_min':>10}")
    for scanner in sorted({r["scanner"] for r in ok}):
        for dose in sorted({r["dose_mgy"] for r in ok if r["scanner"] == scanner}):
            sel = [r for r in ok if r["scanner"] == scanner and r["dose_mgy"] == dose]
            dz = np.array([r["dz_mm"] for r in sel])
            print(f"{scanner:8}{dose:>5}{len(sel):>4}{np.median(dz):>9.2f}{np.abs(dz).max():>9.2f}{dz.std():>7.2f}"
                  f"{np.median([r['rms_best'] for r in sel]):>10.2f}{np.median([r['rms_zero'] for r in sel]):>10.2f}"
                  f"{min(r['sharp'] for r in sel):>10.1f}")
    weak = [r for r in ok if r["sharp"] < 1.5]
    if weak:
        print(f"\n{len(weak)} series with sharp < 1.5 (flat minimum, offset poorly determined); first 10:")
        for r in weak[:10]:
            print(f"  {r['scanner']} {r['dose_mgy']}mGy #{r['repeat']}: dz={r['dz_mm']:.2f} sharp={r['sharp']:.2f}")
    for r in results:
        if r.get("error") and r["error"] != "reference":
            print(f"  NOTE {r.get('scanner')} {r.get('dose_mgy')} {r.get('recon')} #{r.get('repeat')}: {r['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="ct4h_index.csv")
    parser.add_argument("--data_dir", default=None, help="DICOM root the index paths are relative to. Default: CT4H_DIR.")
    parser.add_argument("--out", default="ct4h_z.csv")
    parser.add_argument("--scanners", nargs="+", choices=SCANNERS, default=SCANNERS)
    parser.add_argument("--doses", nargs="+", type=int, choices=DOSES, default=DOSES)
    parser.add_argument("--recons", nargs="+", choices=RECONS, default=["FBP"])
    parser.add_argument("--repeats", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--max_shift_mm", type=float, default=15.0, help="search +/- this many mm.")
    parser.add_argument("--workers", type=int, default=4, help="processes, one scanner each.")
    args = parser.parse_args()

    if args.data_dir is None:
        from ct4h.env import get_ct4h_dir

        args.data_dir = get_ct4h_dir()
    rows = select(read_index(args.index), scanners=args.scanners, doses=args.doses,
                  recons=args.recons, repeats=args.repeats)
    if not rows:
        sys.exit("No series match the selection.")
    fbp_only = args.recons == ["FBP"]
    tasks = [
        {"scanner": s, "rows": [r for r in rows if r["scanner"] == s], "data_dir": args.data_dir,
         "max_shift_mm": args.max_shift_mm, "fbp_only": fbp_only}
        for s in args.scanners if any(r["scanner"] == s for r in rows)
    ]
    print(f"{len(rows)} series on {len(tasks)} scanners (every slice is read: this takes a few minutes)")

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
