"""
Usage:
    python scripts/preprocess_ldct.py
    python scripts/preprocess_ldct.py --anatomies liver --limit_patients 2 --workers 2
    python scripts/preprocess_ldct.py --include_unpaired        # also the GE full-dose-only patients
    
Structure:
    {output_dir}/{anatomy}/{dose}/{patient}_s{NNN}.npz
        image      (H, W) int16 Hounsfield units (not windowed; the dataset applies the window)
        slice_idx  int32, rank of the slice among the kept slices
        slice_pos  float32 in [0, 1], slice_idx / (N - 1)
        z          float32, position in mm
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldct.data.dicom import match_slices, read_series
from ldct.data.utils import ANATOMIES, read_index, slice_name
from ldct.env import get_ldct_dir, get_ldct_preprocessed_dir


def save_slice(path: Path, image: np.ndarray, slice_idx: int, n_slices: int, z: float, compress: bool) -> None:
    tmp = path.with_name(path.name + ".tmp")
    save = np.savez_compressed if compress else np.savez
    with open(tmp, "wb") as handle:
        save(
            handle,
            image=image,
            slice_idx=np.int32(slice_idx),
            slice_pos=np.float32(slice_idx / max(n_slices - 1, 1)),
            z=np.float32(z),
        )
    os.replace(tmp, path)


def process_patient(task: dict) -> dict:
    """Write every slice of one patient. Returns a summary, or {'error': ...}."""
    patient, anatomy = task["patient_id"], task["anatomy"]
    output_dir, data_dir = Path(task["output_dir"]), Path(task["data_dir"])
    marker = output_dir / "_done" / f"{patient}.json"
    try:
        if marker.exists() and not task["overwrite"]:
            return {"patient_id": patient, "skipped": True}

        volumes = {}
        for dose, series_dir in task["series"].items():
            volume, z, _ = read_series(data_dir / series_dir)
            volumes[dose] = (volume, z)

        if len(volumes) == 2:
            (low, z_low), (full, z_full) = volumes["low"], volumes["full"]
            if low.shape[1:] != full.shape[1:]:
                raise ValueError(f"low/full slice sizes differ: {low.shape[1:]} vs {full.shape[1:]}")
            i_low, i_full = match_slices(z_low, z_full, task["z_tol"])
            matched = len(i_low) / max(len(z_low), len(z_full))
            if matched < task["min_match"]:
                raise ValueError(f"only {matched:.0%} of slices match on z (low {len(z_low)}, full {len(z_full)})")
            chosen = {"low": (low[i_low], z_low[i_low]), "full": (full[i_full], z_full[i_full])}
        else:
            chosen = volumes

        n_slices = len(next(iter(chosen.values()))[1])
        for dose, (volume, z) in chosen.items():
            folder = output_dir / anatomy / dose
            folder.mkdir(parents=True, exist_ok=True)
            for idx in range(n_slices):
                save_slice(folder / f"{slice_name(patient, idx)}.npz", volume[idx], idx, n_slices, z[idx], task["compress"])

        summary = {
            "patient_id": patient,
            "anatomy": anatomy,
            "doses": sorted(chosen),
            "n_slices": int(n_slices),
            "dropped": {dose: int(len(volumes[dose][1]) - n_slices) for dose in volumes},
        }
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(summary))
        return summary
    except Exception as error:  # report per patient, keep the other workers going
        return {"patient_id": patient, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=None, help="Downloaded DICOM root. Default: LDCT_DIR.")
    parser.add_argument("--output_dir", default=None, help="Default: LDCT_PREPROCESSED_DIR.")
    parser.add_argument("--index", default=None, help="Default: {output_dir}/index.csv.")
    parser.add_argument("--anatomies", nargs="+", choices=ANATOMIES, default=ANATOMIES)
    parser.add_argument("--include_unpaired", action="store_true", help="Also write the full-dose-only (GE) patients.")
    parser.add_argument("--limit_patients", type=int, default=0, help="Debug: at most N patients per anatomy.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--z_tol", type=float, default=0.05, help="mm tolerance when matching low/full slices.")
    parser.add_argument("--min_match", type=float, default=0.95, help="Fraction of slices that must match on z.")
    parser.add_argument("--no_compress", action="store_true", help="Plain .npz: faster to read, about 2-3x larger.")
    parser.add_argument("--overwrite", action="store_true", help="Redo patients that are already done.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir or get_ldct_dir())
    output_dir = Path(args.output_dir or get_ldct_preprocessed_dir() or "")
    if not str(output_dir):
        sys.exit("Pass --output_dir or set LDCT_PREPROCESSED_DIR.")
    rows = read_index(args.index or output_dir / "index.csv")

    series: dict[str, dict] = {}
    for row in rows:
        if row["anatomy"] not in args.anatomies or (not row["paired"] and not args.include_unpaired):
            continue
        entry = series.setdefault(row["patient_id"], {"anatomy": row["anatomy"], "series": {}})
        entry["series"][row["dose"]] = row["series_dir"]
    patients = sorted(series)
    if args.limit_patients:
        patients = [p for anatomy in args.anatomies
                    for p in [q for q in patients if series[q]["anatomy"] == anatomy][: args.limit_patients]]

    tasks = [
        {
            "patient_id": p, "anatomy": series[p]["anatomy"], "series": series[p]["series"],
            "data_dir": str(data_dir), "output_dir": str(output_dir), "overwrite": args.overwrite,
            "z_tol": args.z_tol, "min_match": args.min_match, "compress": not args.no_compress,
        }
        for p in patients
    ]
    print(f"{len(tasks)} patients -> {output_dir}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_patient, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="patients"):
            results.append(future.result())

    done = [r for r in results if "n_slices" in r]
    skipped = [r for r in results if r.get("skipped")]
    failed = [r for r in results if "error" in r]
    print(f"\nwritten {len(done)} patients ({sum(r['n_slices'] for r in done)} slice positions), "
          f"already done {len(skipped)}, failed {len(failed)}")
    for r in done:
        if any(r["dropped"].values()):
            print(f"  {r['patient_id']}: dropped unmatched slices {r['dropped']}")
    for r in sorted(failed, key=lambda r: r["patient_id"]):
        print(f"  FAILED {r['patient_id']}: {r['error']}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
