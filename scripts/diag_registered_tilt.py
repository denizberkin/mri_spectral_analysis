"""Is the z residual of a registered volume constant across the phantom, or does it vary with y / x (a tilt)?

For each volume the per-slice mean-HU profile is computed separately in three bands along y and three along x
(thirds of the phantom's extent in the reference), and each band's profile is matched to the same band of the
reference volume (sub-slice). A constant offset gives the same dz in all bands; a small rotation about the x or y
axis gives dz that changes from one band to the next.

    dz_y_low/mid/high, dz_x_low/mid/high   residual z (mm) per band
    tilt_y_mm, tilt_x_mm                   dz high band minus dz low band
    rms_*                                  profile mismatch at the best offset, per band (HU)

Usage:
    python scripts/diag_registered_tilt.py --reg_dir <authors_registered> --ref F1_14mGy_FBP_rep01.nii.gz --scanners F1
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_ct4h_registered import NAME, load_volume, profile_dz  # noqa: E402


def bands(mask_1d: np.ndarray, n: int = 3) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask_1d)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    edges = np.linspace(lo, hi, n + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reg_dir", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--scanners", nargs="+", required=True)
    parser.add_argument("--out", default="ct4h_registered_tilt.csv")
    parser.add_argument("--max_z", type=float, default=8.0)
    args = parser.parse_args()

    reg_dir = Path(args.reg_dir)
    ref, _ = load_volume(reg_dir / args.ref)  # (z, y, x)
    body = ref.mean(axis=0) > -500  # phantom footprint in (y, x)
    yb, xb = bands(body.any(axis=1)), bands(body.any(axis=0))
    names = ("low", "mid", "high")
    rows = []
    files = sorted(p for p in reg_dir.glob("*.nii.gz") if NAME.match(p.name) and NAME.match(p.name)["scanner"] in args.scanners)
    for path in files:
        mov, zooms = load_volume(path)
        row = {"file": path.name}
        for axis, spans in (("y", yb), ("x", xb)):
            for name, (a, b) in zip(names, spans):
                sl = (slice(None), slice(a, b), slice(None)) if axis == "y" else (slice(None), slice(None), slice(a, b))
                dz, rms_best, _ = profile_dz(ref[sl].mean(axis=(1, 2)), mov[sl].mean(axis=(1, 2)), args.max_z)
                row[f"dz_{axis}_{name}"] = dz * zooms[2]
                row[f"rms_{axis}_{name}"] = rms_best
            row[f"tilt_{axis}_mm"] = row[f"dz_{axis}_high"] - row[f"dz_{axis}_low"]
        rows.append(row)
        print(f"{path.name}: dz_y {row['dz_y_low']:.2f}/{row['dz_y_mid']:.2f}/{row['dz_y_high']:.2f}  "
              f"dz_x {row['dz_x_low']:.2f}/{row['dz_x_mid']:.2f}/{row['dz_x_high']:.2f} mm", flush=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
