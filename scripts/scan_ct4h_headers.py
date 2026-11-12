"""Scan CT4Harmonization-Multicentric DICOM headers into one CSV row per series.
Reads headers only (no pixel data) and interprets nothing.

    python scripts/scan_ct4h_headers.py <DICOM root> <ct4h_series_index.csv>

<DICOM root> is the folder the series paths are taken relative to (the same folder later passed as --data_dir to
run_authors_registration.py), so series_dir in the output reads CT4Harmonization-Multicentric/<study>/<series>.
The output feeds scripts/build_ct4h_index.py.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import pydicom

FIELDS = [
    "Modality", "Manufacturer", "ManufacturerModelName", "StationName", "DeviceSerialNumber",
    "InstitutionName", "StudyDescription", "SeriesDescription", "ProtocolName", "SeriesNumber",
    "ConvolutionKernel", "SliceThickness", "SpacingBetweenSlices", "PixelSpacing", "Rows", "Columns",
    "KVP", "XRayTubeCurrent", "ExposureTime", "Exposure", "CTDIvol", "SingleCollimationWidth",
    "TotalCollimationWidth", "SpiralPitchFactor", "ReconstructionDiameter", "AcquisitionDate",
    "StudyInstanceUID", "SeriesInstanceUID", "PatientID",
]


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)) or v.__class__.__name__ == "MultiValue":
        return "\\".join(str(x) for x in v)
    return str(v).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out_csv")
    args = ap.parse_args()

    rows = []
    for dirpath, _dirs, files in os.walk(args.root):
        dcm = sorted(f for f in files if f.lower().endswith(".dcm"))
        if not dcm:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(dirpath, dcm[0]), stop_before_pixels=True)
        except Exception as e:  # report, keep going
            print(f"FAILED {dirpath}: {e}", file=sys.stderr)
            continue
        row = {k: fmt(getattr(ds, k, None)) for k in FIELDS}
        row["n_files"] = len(dcm)
        row["series_dir"] = os.path.relpath(dirpath, args.root)
        rows.append(row)

    if not rows:
        sys.exit(f"No .dcm files found under {args.root}")

    cols = FIELDS + ["n_files", "series_dir"]
    rows.sort(key=lambda r: (r["StudyDescription"], r["SeriesDescription"], r["series_dir"]))
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    total_files = sum(r["n_files"] for r in rows)
    print(f"{len(rows)} series, {total_files} .dcm files -> {args.out_csv}\n")
    uniq = defaultdict(Counter)
    for r in rows:
        for k in FIELDS:
            uniq[k][r[k]] += 1
    for k in FIELDS:
        n = len(uniq[k])
        if n <= 40 and k not in ("StudyInstanceUID", "SeriesInstanceUID", "PatientID"):
            print(f"{k} ({n}): " + ", ".join(f"{v or '<empty>'}={c}" for v, c in uniq[k].most_common()))
        else:
            print(f"{k}: {n} distinct values")


if __name__ == "__main__":
    main()
