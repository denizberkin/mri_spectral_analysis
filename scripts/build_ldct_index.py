import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldct.data.utils import (
    FULL_PATTERN, LOW_PATTERN, PATIENT_RE, UNPAIRED,
    anatomy_of, assign_splits, classify_dose, write_index,
)
from ldct.env import get_ldct_dir, get_ldct_preprocessed_dir


def find_series_dirs(patient_dir: Path) -> list[Path]:
    """Every directory under a patient that directly holds .dcm files."""
    found = []
    for root, _dirs, files in os.walk(patient_dir):
        if any(name.lower().endswith(".dcm") for name in files):
            found.append(Path(root))
    return sorted(found)


def describe_series(series_dir: Path, data_dir: Path, patient_id: str, low_pattern: str, full_pattern: str) -> dict:
    files = sorted(p for p in series_dir.iterdir() if p.suffix.lower() == ".dcm")
    header = pydicom.dcmread(files[0], stop_before_pixels=True)
    description = str(getattr(header, "SeriesDescription", ""))
    dose = classify_dose(description, low_pattern, full_pattern)
    spacing = getattr(header, "PixelSpacing", None)
    return {
        "anatomy": anatomy_of(patient_id),
        "patient_id": patient_id,
        "dose": dose,
        "split": "",
        "paired": False,
        "series_uid": str(header.SeriesInstanceUID),
        "series_dir": series_dir.relative_to(data_dir).as_posix(),
        "n_slices": len(files),
        "manufacturer": str(getattr(header, "Manufacturer", "")),
        "model": str(getattr(header, "ManufacturerModelName", "")),
        "kernel": str(getattr(header, "ConvolutionKernel", "")),
        "series_description": description,
        "slice_thickness": float(header.SliceThickness) if "SliceThickness" in header else "",
        "pixel_spacing": float(spacing[0]) if spacing is not None else "",
        "rows": int(header.Rows),
        "columns": int(header.Columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=None, help="Folder with the patient folders. Default: LDCT_DIR.")
    parser.add_argument("--output", default=None, help="index.csv path. Default: $LDCT_PREPROCESSED_DIR/index.csv.")
    parser.add_argument("--val_per_anatomy", type=int, default=5, help="Paired patients per anatomy for validation.")
    parser.add_argument("--test_per_anatomy", type=int, default=10, help="Paired patients per anatomy for testing.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--low_pattern", default=LOW_PATTERN, help="Regex on the series description for low dose.")
    parser.add_argument("--full_pattern", default=FULL_PATTERN, help="Regex on the series description for full dose.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir or get_ldct_dir())
    if args.output:
        output = Path(args.output)
    else:
        preprocessed = get_ldct_preprocessed_dir()
        if not preprocessed:
            sys.exit("Pass --output or set LDCT_PREPROCESSED_DIR.")
        output = Path(preprocessed) / "index.csv"
    if not data_dir.is_dir():
        sys.exit(f"Data directory not found: {data_dir}")

    patient_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and PATIENT_RE.match(p.name))
    if not patient_dirs:
        sys.exit(f"No patient folders (C001 ... L100) found in {data_dir}. Is --data_dir one level too high or too low?")

    rows, problems = [], []
    for patient_dir in patient_dirs:
        series_dirs = find_series_dirs(patient_dir)
        if not series_dirs:
            problems.append(f"{patient_dir.name}: no .dcm files")
        for series_dir in series_dirs:
            row = describe_series(series_dir, data_dir, patient_dir.name, args.low_pattern, args.full_pattern)
            if row["dose"] is None:
                problems.append(f"{patient_dir.name}: cannot tell the dose from SeriesDescription {row['series_description']!r}")
            rows.append(row)

    # exactly one series per patient and dose
    seen = Counter((r["patient_id"], r["dose"]) for r in rows)
    problems += [f"{p}: {n} {d}-dose series, expected 1" for (p, d), n in seen.items() if n > 1 and d is not None]
    if problems:
        print("Problems found, index not written:")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)

    doses = defaultdict(set)
    for r in rows:
        doses[r["patient_id"]].add(r["dose"])
    paired = {p for p, d in doses.items() if d == {"low", "full"}}
    paired_by_anatomy = defaultdict(list)
    for p in sorted(paired):
        paired_by_anatomy[anatomy_of(p)].append(p)
    splits = assign_splits(paired_by_anatomy, args.val_per_anatomy, args.test_per_anatomy, args.seed)
    for r in rows:
        r["paired"] = r["patient_id"] in paired
        r["split"] = splits.get(r["patient_id"], UNPAIRED)

    # low and full of one patient come from the same scan: the slice counts should match
    by_patient = defaultdict(dict)
    for r in rows:
        by_patient[r["patient_id"]][r["dose"]] = r["n_slices"]
    for patient in sorted(paired):
        counts = by_patient[patient]
        if counts["low"] != counts["full"]:
            print(f"warning: {patient} has {counts['low']} low-dose and {counts['full']} full-dose slices; "
                  "preprocessing will match them by z position")

    write_index(sorted(rows, key=lambda r: (r["anatomy"], r["patient_id"], r["dose"])), output)

    print(f"Wrote {output}: {len(rows)} series, {len(doses)} patients, {len(paired)} paired\n")
    summary = Counter((r["anatomy"], r["manufacturer"], r["dose"], r["split"]) for r in rows)
    print(f"{'anatomy':8} {'manufacturer':14} {'dose':5} {'split':9} series")
    for (anatomy, manufacturer, dose, split), n in sorted(summary.items()):
        print(f"{anatomy:8} {manufacturer[:14]:14} {dose:5} {split:9} {n}")


if __name__ == "__main__":
    main()
