"""How well aligned are the volumes written by run_authors_registration.py?

The runner puts every series on the grid of the authors' reference scan (A1 10 mGy IR). This measures, with
our own non-resampling estimators, what is left over, scan against one registered reference volume:

    shift_y_px, shift_x_px, shift_mm   in-plane residual (median over slices, phase correlation upsample 10)
    shift_spread_px                    spread of that shift across slices (std of the per-slice |shift|)
    ncc                                correlation of the slices after that shift
    dz_slices, dz_mm                   residual z displacement from the per-slice mean-HU profile (sub-slice)
    rms_best / rms_zero                profile mismatch at the best offset / at zero (HU)

0 everywhere means the registration put the volumes on top of each other. Nothing is resampled here:
a Gaussian (--smooth_sigma) is applied to throwaway copies for the shift estimate only.

Axes of the files: their runner saves flip_volume(volume.transpose(2, 1, 0), axis=1), i.e. (x, y flipped, z).
We undo the transpose to get (z, y, x) slices; the y flip only mirrors the sign of dy.

Usage:
    python scripts/check_ct4h_registered.py --reg_dir <authors_registered> --out ct4h_registered_check.csv
"""

import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_ct4h_alignment import register  # noqa: E402

NAME = re.compile(r"^(?P<scanner>[A-H]\d)_(?P<dose>\d+)mGy_(?P<recon>[A-Z]+)_rep(?P<rep>\d+)\.nii\.gz$")
COLUMNS = [
    "file", "scanner", "dose_mgy", "recon", "repeat", "shift_y_px", "shift_x_px", "shift_mm", "shift_spread_px",
    "ncc", "slices_used", "dz_slices", "dz_mm", "rms_best", "rms_zero", "error",
]
_ref_cache: dict = {}


def load_volume(path: Path) -> tuple[np.ndarray, tuple]:
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    return arr.transpose(2, 1, 0), tuple(float(v) for v in img.header.get_zooms()[:3])  # (z, y, x)


def profile_dz(ref_p: np.ndarray, mov_p: np.ndarray, max_slices: float, step: float = 0.05) -> tuple[float, float, float]:
    """Shift d (slices) such that mov(z + d) matches ref(z); the best rms, and the rms at 0."""
    z = np.arange(len(ref_p), dtype=float)

    def rms(d: float) -> float:
        inside = (z + d >= 0) & (z + d <= len(mov_p) - 1)
        if inside.sum() < 0.5 * len(z):
            return float("inf")
        return float(np.std(ref_p[inside] - np.interp(z[inside] + d, z, mov_p)))

    grid = np.arange(-max_slices, max_slices + step / 2, step)
    scores = np.array([rms(d) for d in grid])
    k = int(np.argmin(scores))
    best = float(grid[k])
    if 0 < k < len(grid) - 1 and np.isfinite(scores[[k - 1, k + 1]]).all():
        a, b, c = scores[k - 1], scores[k], scores[k + 1]
        denom = a - 2 * b + c
        if denom > 0:
            best += float(step * 0.5 * (a - c) / denom)
    return best, rms(best), rms(0.0)


def check_one(task: dict) -> dict:
    path = Path(task["path"])
    match = NAME.match(path.name)
    row = {"file": path.name, "scanner": match["scanner"], "dose_mgy": int(match["dose"]), "recon": match["recon"],
           "repeat": int(match["rep"]), "error": ""}
    try:
        if "ref" not in _ref_cache:
            _ref_cache["ref"] = load_volume(Path(task["ref"]))
        ref, _ = _ref_cache["ref"]
        mov, zooms = load_volume(path)
        if mov.shape != ref.shape:
            raise ValueError(f"shape {mov.shape} differs from the reference {ref.shape}")
        sigma = task["sigma"]
        smooth = (lambda a: ndimage.gaussian_filter(a, sigma)) if sigma > 0 else (lambda a: a)
        lo, hi = task["span"]
        wanted = np.unique(np.round(np.linspace(lo, hi, task["n_check"]) * (ref.shape[0] - 1)).astype(int))
        shifts, nccs = [], []
        for i in wanted:
            shift, ncc = register(smooth(ref[i]), smooth(mov[i]))
            shifts.append(shift)
            nccs.append(ncc)
        shifts = np.array(shifts)
        median = np.median(shifts, axis=0)
        # zooms are the NIfTI voxel sizes (x, y, z); the in-plane ones are used for the shift in mm
        row.update(
            shift_y_px=float(median[0]), shift_x_px=float(median[1]),
            shift_mm=float(np.hypot(median[0] * zooms[1], median[1] * zooms[0])),
            shift_spread_px=float(np.std(np.hypot(shifts[:, 0], shifts[:, 1]))), ncc=float(np.median(nccs)),
            slices_used=len(wanted),
        )
        dz, rms_best, rms_zero = profile_dz(ref.mean(axis=(1, 2)), mov.mean(axis=(1, 2)), task["max_z"])
        row.update(dz_slices=dz, dz_mm=dz * zooms[2], rms_best=rms_best, rms_zero=rms_zero)
    except Exception as error:  # report per file, keep going
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if not r["error"]]
    print(f"\n{len(ok)} volumes checked, {len(rows) - len(ok)} errors")
    print("\n--- residual vs the reference volume, per scanner (median / max over scans)")
    print(f"{'scanner':8}{'n':>4}{'shift_mm':>14}{'|dz_mm|':>14}{'spread_px':>11}{'ncc_min':>9}{'rms_best':>10}{'rms_zero':>10}")
    for scanner in sorted({r["scanner"] for r in ok}):
        sel = [r for r in ok if r["scanner"] == scanner]
        sh = np.array([r["shift_mm"] for r in sel])
        dz = np.abs([r["dz_mm"] for r in sel])
        print(f"{scanner:8}{len(sel):>4}{np.median(sh):>8.2f}/{sh.max():<5.2f}{np.median(dz):>8.2f}/{dz.max():<5.2f}"
              f"{max(r['shift_spread_px'] for r in sel):>11.2f}{min(r['ncc'] for r in sel):>9.3f}"
              f"{np.median([r['rms_best'] for r in sel]):>10.2f}{np.median([r['rms_zero'] for r in sel]):>10.2f}")
    print("\n--- worst 10 scans by in-plane residual (mm)")
    for r in sorted(ok, key=lambda r: -r["shift_mm"])[:10]:
        print(f"  {r['file']}: shift {r['shift_mm']:.2f} mm (dy {r['shift_y_px']:.2f}, dx {r['shift_x_px']:.2f} px), "
              f"dz {r['dz_mm']:.2f} mm, ncc {r['ncc']:.3f}")
    print("\n--- worst 10 scans by |dz| (mm)")
    for r in sorted(ok, key=lambda r: -abs(r["dz_mm"]))[:10]:
        print(f"  {r['file']}: dz {r['dz_mm']:.2f} mm, shift {r['shift_mm']:.2f} mm")
    for r in rows:
        if r["error"]:
            print(f"  ERROR {r['file']}: {r['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reg_dir", required=True, help="folder with the .nii.gz of run_authors_registration.py")
    parser.add_argument("--out", default="ct4h_registered_check.csv")
    parser.add_argument("--ref", default="A1_14mGy_FBP_rep01.nii.gz",
                        help="file name (in --reg_dir) of the volume everything is compared to. The runner's own "
                             "reference (A1 10 mGy IR rep 1) is not in the FBP output; this scan is the closest.")
    parser.add_argument("--scanners", nargs="+", default=None)
    parser.add_argument("--n_check", type=int, default=9)
    parser.add_argument("--span", nargs=2, type=float, default=[0.15, 0.85])
    parser.add_argument("--max_z", type=float, default=8.0, help="+/- slices searched for the z offset.")
    parser.add_argument("--smooth_sigma", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    reg_dir = Path(args.reg_dir)
    files = sorted(p for p in reg_dir.glob("*.nii.gz") if NAME.match(p.name) and not p.name.startswith("tmp_"))
    if args.scanners:
        files = [p for p in files if NAME.match(p.name)["scanner"] in args.scanners]
    if not (reg_dir / args.ref).is_file():
        raise SystemExit(f"reference {args.ref} not found in {reg_dir}")
    tasks = [{"path": str(p), "ref": str(reg_dir / args.ref), "sigma": args.smooth_sigma, "span": tuple(args.span),
              "n_check": args.n_check, "max_z": args.max_z} for p in files]
    print(f"{len(tasks)} volumes, reference {args.ref}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(tqdm(pool.map(check_one, tasks), total=len(tasks), desc="volumes"))
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")
    summarize(rows)


if __name__ == "__main__":
    main()
