"""Run the dataset authors' own preprocessing on CT4H series, and log what it does.

This is the loop body of Harmonization-Dataset/registered_dataset.py (create_registered_dataset), step
for step, calling THEIR functions from THEIR repo (read_dicom, rolled_ssim, find_shifts, ImageRegistrator,
flip_volume, array_to_tensor). Nothing of the algorithm is changed. What is added:

  * our paths: series come from ct4h_index.csv instead of their NAS layout;
  * a log row per series (the original prints to the console and swallows every exception);
  * the SSIM numbers behind their silent fallback ("Registration failed. Using the original image."),
    plus a like-for-like version of it (see below);
  * resumable (existing outputs are skipped), sharding for several processes, --limit for a benchmark.

Their steps per series
  1. read_dicom: sort by InstanceNumber (z positions are ignored), z-zoom to 2 mm if SliceThickness != 2 (linear),
     stretch/trim to 343 slices (linear), clip to [-1000, 2000] HU.
  2. decide whether the z axis is flipped (rolled SSIM of the volume and of its flipped copy vs the reference).
  3. find_shifts: circular roll along z in steps of 4 slices, lowest RMSE against the reference.
  4. crop [13:323, 120:395, 64:445] and rigid elastix registration to the reference (cubic B-spline resampling).
  5. fallback: their rule saves the unregistered crop if SSIM(after, cropped) < SSIM(before, WHOLE volume).
     The two SSIMs are computed on different regions (the whole volume includes easy air), so a good registration
     can be thrown away (seen on F1: crop SSIM 0.39 -> 0.50, yet fallback). --fallback_rule picks the rule applied:
     authors (theirs), like_for_like (cropped before vs cropped after; default here) or never.
     Both decisions are always logged (fallback_authors, fallback_like_for_like).

  --force_roll SCANNER=SLICES replaces step 3 for that scanner by a fixed roll. Added because step 3 only tries multiples
     of 4 slices and, on noisy low-dose F1 scans, flips between 4 and 8 (15/50 F1 scans got 8, leaving a 1.28 mm
     z offset; the raw headers say all F1 scans start at the same z). Logged in roll_forced.

  --roll_from <registration_manifest.csv> gives every series the roll its FBP twin got (same scan, same grid), looked
     up by scanner, dose and repeat; series without a twin in the manifest keep their own search. Use it for IR and DL
     so the three reconstructions of one acquisition are rolled identically. Logged as roll_forced = 2.

Output per series: <out_dir>/<scanner>_<dose>mGy_<recon>_rep<NN>.nii.gz (float32; theirs is float64) and a
row in <out_dir>/registration_log.csv.

Run it in the env that has itk-elastix, antspyx, dcmstack, monai, nibabel and a CUDA torch:
    python scripts/run_authors_registration.py --index ct4h_index.csv --data_dir <DICOM root> \\
        --authors_repo ~/Documents/deniz/mri_fields/Harmonization-Dataset --out_dir <dir> --limit 3
"""

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct4h.data.utils import DOSES, RECONS, SCANNERS, read_index, select

# constants of registered_dataset.py
CROP = [13, 323, 120, 395, 64, 445]
SSIM_DATA_RANGE = 2000
COLUMNS = [
    "scanner", "dose_mgy", "recon", "repeat", "series_uid", "out_file", "flipped", "ssim_raw", "ssim_roll_a",
    "ssim_roll_b", "roll_slices", "roll_forced", "ssim_before_full", "ssim_before_crop", "ssim_after", "fallback",
    "fallback_authors", "fallback_like_for_like", "seconds", "error",
]


def load_authors(repo: Path) -> SimpleNamespace:
    """Import their modules (they import each other by bare name) and build the objects their script uses."""
    sys.path.insert(0, str(repo))
    import nibabel as nib
    import torch
    from ImageRegistration import ImageRegistrator
    from monai import metrics
    from utils import array_to_tensor, find_shifts, flip_volume, read_dicom, rolled_ssim

    return SimpleNamespace(
        nib=nib, torch=torch, ImageRegistrator=ImageRegistrator, read_dicom=read_dicom, rolled_ssim=rolled_ssim,
        find_shifts=find_shifts, array_to_tensor=array_to_tensor, flip_volume=flip_volume,
        ssim=metrics.SSIMMetric(spatial_dims=3, data_range=SSIM_DATA_RANGE),
    )


def register_one(folder: Path, out_file: Path, ref: SimpleNamespace, registrator, t: SimpleNamespace,
                 fallback_rule: str = "like_for_like", forced_roll: int | None = None, roll_source: int = 1) -> dict:
    """Their loop body. Returns the log values."""
    log = {}
    volumes = t.read_dicom(str(folder), numpy_format=True, crop_region=None)
    nifti_image = volumes[-1]

    log["ssim_raw"] = t.ssim(ref.volumes[0], volumes[0]).item()
    log["ssim_roll_a"] = t.rolled_ssim(ref.volumes[0], volumes[0]).item()
    log["ssim_roll_b"] = t.rolled_ssim(ref.volumes[0], volumes[2]).item()
    log["flipped"] = int(log["ssim_roll_b"] > log["ssim_roll_a"])
    if log["flipped"]:
        volumes[0] = volumes[2]
        volumes[1] = volumes[3]

    if forced_roll is None:
        volumes[0], shift, _ = t.find_shifts(volumes[0], ref.volumes[0], axis=-1)
    else:  # their roll search only tries multiples of 4 slices and noise can flip it between neighbours
        shift = int(forced_roll)
        volumes[0] = t.torch.roll(volumes[0], shifts=shift, dims=-1)
    volumes[1] = np.roll(volumes[1], shift, axis=0)
    log["roll_slices"] = int(shift)
    log["roll_forced"] = 0 if forced_roll is None else roll_source
    log["ssim_before_full"] = t.ssim(ref.volumes[0], volumes[0]).item()

    c = CROP
    volume = volumes[1][c[0]:c[1], c[2]:c[3], c[4]:c[5]]
    ref_volume = ref.volumes[1][c[0]:c[1], c[2]:c[3], c[4]:c[5]]

    registered = registrator.register_image(volume, ref_volume, downsample_factor=1)[0]

    registered_t = t.array_to_tensor(registered.transpose(1, 2, 0))
    ref_t = t.array_to_tensor(ref_volume.transpose(1, 2, 0))
    log["ssim_after"] = t.ssim(ref_t, registered_t).item()
    log["ssim_before_crop"] = t.ssim(ref_t, t.array_to_tensor(volume.transpose(1, 2, 0))).item()

    log["fallback_authors"] = int(log["ssim_after"] < log["ssim_before_full"])  # their rule
    log["fallback_like_for_like"] = int(log["ssim_after"] < log["ssim_before_crop"])
    log["fallback"] = {"authors": log["fallback_authors"], "like_for_like": log["fallback_like_for_like"], "never": 0}[fallback_rule]
    if log["fallback"]:
        registered = volume

    nifti = registered.transpose(2, 1, 0)
    nifti = t.flip_volume(nifti, axis=1)
    tmp = out_file.with_name("tmp_" + out_file.name)
    t.nib.save(t.nib.Nifti1Image(nifti.astype(np.float32), nifti_image.affine), str(tmp))
    os.replace(tmp, out_file)  # an interrupted run never leaves a half-written file behind
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="ct4h_index.csv")
    parser.add_argument("--data_dir", required=True, help="DICOM root the index paths are relative to.")
    parser.add_argument("--authors_repo", required=True, help="clone of medgift/Harmonization-Dataset")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scanners", nargs="+", choices=SCANNERS, default=SCANNERS)
    parser.add_argument("--doses", nargs="+", type=int, choices=DOSES, default=DOSES)
    parser.add_argument("--recons", nargs="+", choices=RECONS, default=["FBP"],
                        help="reconstructions to register (the authors use all of them). Default FBP; run IR DL afterwards with --roll_from.")
    parser.add_argument("--repeats", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--ref_scanner", default="A1")
    parser.add_argument("--ref_dose", type=int, default=10)
    parser.add_argument("--ref_recon", default="IR")
    parser.add_argument("--ref_repeat", type=int, default=1,
                        help="their reference folder name does not say which of the 10 repeats it was "
                             "(the StudyTime 174008.691 is shared by all of them); the repeats agree to ~0.1 mm.")
    parser.add_argument("--fallback_rule", choices=["authors", "like_for_like", "never"], default="like_for_like",
                        help="when the unregistered crop is saved instead of the registered volume (see the docstring).")
    parser.add_argument("--force_roll", nargs="+", default=[], metavar="SCANNER=SLICES",
                        help="use this circular z roll for the scanner instead of their search (e.g. F1=4).")
    parser.add_argument("--roll_from", default=None, metavar="MANIFEST.csv",
                        help="take each series' roll from its FBP twin in this manifest (for --recons IR DL).")
    parser.add_argument("--log_name", default=None, help="log file name (default registration_log_<K>of<N>.csv).")
    parser.add_argument("--shard", default="0/1", help="K/N: this process takes every N-th series (start N processes).")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many series (benchmark).")
    args = parser.parse_args()

    shard, n_shards = (int(v) for v in args.shard.split("/"))
    forced = {k: int(v) for k, v in (item.split("=") for item in args.force_roll)}
    twin_roll = {}
    if args.roll_from:
        manifest = csv.DictReader(open(args.roll_from, newline="", encoding="utf-8"))
        twin_roll = {(r["scanner"], int(r["dose_mgy"]), int(r["repeat"])): int(float(r["roll_slices"]))
                     for r in manifest if r["recon"] == "FBP"}
        print(f"--roll_from: {len(twin_roll)} FBP rolls read from {args.roll_from}")
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = read_index(args.index)
    ref_rows = select(index, scanners=[args.ref_scanner], doses=[args.ref_dose], recons=[args.ref_recon],
                      repeats=[args.ref_repeat])
    if len(ref_rows) != 1:
        raise SystemExit(f"reference selection matched {len(ref_rows)} series, expected 1")
    rows = select(index, scanners=args.scanners, doses=args.doses, recons=args.recons, repeats=args.repeats)
    rows = sorted(rows, key=lambda r: (r["scanner"], r["recon"], r["dose_mgy"], r["repeat"]))
    rows = [r for i, r in enumerate(rows) if i % n_shards == shard]

    t = load_authors(Path(args.authors_repo).expanduser())
    ref_dir = data_dir / ref_rows[0]["series_dir"]
    print(f"reference: {args.ref_scanner} {args.ref_dose} mGy {args.ref_recon} repeat {args.ref_repeat}: {ref_dir.name}")
    ref = SimpleNamespace(volumes=t.read_dicom(str(ref_dir), numpy_format=True, crop_region=None))
    registrator = t.ImageRegistrator("elastic", ref.volumes[1])  # 'elastic' is their label for a rigid elastix map

    log_path = out_dir / (args.log_name or f"registration_log_{shard}of{n_shards}.csv")
    new_log = not log_path.exists()
    done = 0
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore", restval="")
        if new_log:
            writer.writeheader()
        for row in rows:
            name = f"{row['scanner']}_{row['dose_mgy']}mGy_{row['recon']}_rep{row['repeat']:02d}.nii.gz"
            out_file = out_dir / name
            if out_file.exists():
                continue
            base = {"scanner": row["scanner"], "dose_mgy": row["dose_mgy"], "recon": row["recon"],
                    "repeat": row["repeat"], "series_uid": row["series_uid"], "out_file": name}
            start = time.time()
            try:
                twin = twin_roll.get((row["scanner"], row["dose_mgy"], row["repeat"]))
                if twin is not None:
                    roll, source = twin, 2
                else:
                    roll, source = forced.get(row["scanner"]), 1
                result = register_one(data_dir / row["series_dir"], out_file, ref, registrator, t, args.fallback_rule,
                                      roll, source)
                error = ""
            except Exception as exc:  # logged per series; the run goes on
                result, error = {}, f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            record = {**base, **result, "seconds": round(time.time() - start, 1), "error": error}
            writer.writerow(record)
            handle.flush()
            t.torch.cuda.empty_cache()
            print(f"{name}: {record['seconds']}s fallback={record.get('fallback', '-')} "
                  f"ssim {record.get('ssim_before_full', float('nan')):.4f} -> {record.get('ssim_after', float('nan')):.4f}"
                  f"{' ERROR ' + error if error else ''}", flush=True)
            done += 1
            if args.limit is not None and done >= args.limit:
                break


if __name__ == "__main__":
    main()
