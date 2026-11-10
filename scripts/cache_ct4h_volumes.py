from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct4h.data.utils import HU_RANGE, RECONS, VENDORS  # noqa: E402

EXPECTED_NIB_SHAPE = (381, 275, 310)   # as nibabel loads the registered files; array order z, y, x is the transpose
FIELDS = ["scanner", "vendor", "dose_mgy", "recon", "repeat", "path", "n_slices", "height", "width"]


def to_zyx(volume: np.ndarray) -> np.ndarray:
    """nibabel (x, y, z) -> (z, y, x), the order of the authors' arrays and of our slices."""
    return np.transpose(volume, (2, 1, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reg_dir", required=True, help="folder with registration_manifest.csv and the .nii.gz files")
    parser.add_argument("--out_dir", required=True, help="cache folder (created)")
    parser.add_argument("--manifest", default=None, help="default: <reg_dir>/registration_manifest.csv")
    parser.add_argument("--recons", nargs="+", default=["FBP"], choices=RECONS)
    parser.add_argument("--limit", type=int, default=0, help="cache only the first N volumes (smoke test)")
    args = parser.parse_args()

    reg_dir, out_dir = Path(args.reg_dir), Path(args.out_dir)
    manifest = Path(args.manifest) if args.manifest else reg_dir / "registration_manifest.csv"
    if not manifest.is_file():
        sys.exit(f"{manifest} not found. Run scripts/make_ct4h_manifest.py --reg_dir {reg_dir} first.")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["recon"] in args.recons]
    usable = [r for r in rows if r["out_file"] and int(float(r.get("fallback") or 0)) == 0]
    skipped = len(rows) - len(usable)
    if args.limit:
        usable = usable[: args.limit]
    print(f"{len(usable)} volumes to cache ({skipped} manifest rows skipped: failed or fallback)")

    index_path = out_dir / "cache_index.csv"
    done: dict[str, dict] = {}
    if index_path.is_file():
        with open(index_path, newline="", encoding="utf-8") as handle:
            done = {r["path"]: r for r in csv.DictReader(handle)}

    low, high = HU_RANGE
    start = time.time()
    for n, row in enumerate(usable, 1):
        scanner, dose, recon, repeat = row["scanner"], int(float(row["dose_mgy"])), row["recon"], int(float(row["repeat"]))
        rel = f"{scanner}/{scanner}_{dose}mGy_{recon}_rep{repeat:02d}.npy"
        target = out_dir / rel
        if rel in done and target.is_file():
            continue
        source = Path(row["out_file"])
        if not source.is_absolute():
            source = reg_dir / source.name
        if not source.is_file():
            print(f"  missing {source}, skipped")
            continue
        volume = nib.load(str(source))
        if tuple(volume.shape) != EXPECTED_NIB_SHAPE:
            sys.exit(f"{source.name}: shape {tuple(volume.shape)}, expected {EXPECTED_NIB_SHAPE}")
        array = np.clip(np.rint(to_zyx(np.asarray(volume.dataobj, dtype=np.float32))), low, high).astype(np.int16)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp.npy")
        np.save(tmp, np.ascontiguousarray(array))
        tmp.replace(target)  # a crash never leaves a half-written .npy under the final name
        done[rel] = {
            "scanner": scanner, "vendor": VENDORS[scanner], "dose_mgy": dose, "recon": recon, "repeat": repeat,
            "path": rel, "n_slices": array.shape[0], "height": array.shape[1], "width": array.shape[2],
        }
        if n % 20 == 0 or n == len(usable):
            print(f"  {n}/{len(usable)}  {time.time() - start:.0f}s", flush=True)
            _write_index(index_path, done)
    _write_index(index_path, done)
    print(f"cache_index.csv: {len(done)} volumes in {out_dir}")


def _write_index(path: Path, done: dict[str, dict]) -> None:
    ordered = sorted(done.values(), key=lambda r: (r["scanner"], r["recon"], int(r["dose_mgy"]), int(r["repeat"])))
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    tmp.replace(path)


if __name__ == "__main__":
    main()
