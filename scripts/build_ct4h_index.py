#!/usr/bin/env python3
"""Turn the raw header scan (ct4h_series_index.csv) into a clean CT4Harmonization index.

One row per CT series, with the acquisition factors parsed from SeriesDescription
(e.g. "A1_Harmonized_10mGy_FBP - #3") and the matching liver SEG series attached.

Input (made by scripts/scan_ct4h_headers.py): one row per DICOM series (CT and SEG) of the TCIA download, with the columns Modality, SeriesDescription,
StudyDescription, StudyInstanceUID, SeriesInstanceUID, Manufacturer, ManufacturerModelName, InstitutionName,
ConvolutionKernel, SliceThickness, PixelSpacing, CTDIvol, Exposure, AcquisitionDate, n_files and series_dir
(the series folder relative to the DICOM root, e.g. CT4Harmonization-Multicentric/<study>/<series>).

Usage:
    python scripts/build_ct4h_index.py <ct4h_series_index.csv> <ct4h_index.csv>
"""
from __future__ import annotations

import re
import sys

import pandas as pd

DESC_RE = re.compile(r"^(?P<scanner>[A-H]\d)_Harmonized_(?P<dose>\d+)mGy_(?P<recon>FBP|IR|DL) - #(?P<repeat>\d+)$")
VENDOR = {"SIEMENS": "Siemens", "Siemens Healthineers": "Siemens", "GE MEDICAL SYSTEMS": "GE",
          "Philips": "Philips", "TOSHIBA": "Toshiba"}


def main(src: str, dst: str) -> None:
    raw = pd.read_csv(src, dtype=str, keep_default_na=False)
    ct = raw[raw.Modality == "CT"].copy()
    seg = raw[raw.Modality == "SEG"].set_index("StudyInstanceUID")

    parsed = ct.SeriesDescription.str.extract(DESC_RE)
    bad = parsed.scanner.isna()
    if bad.any():
        print("UNPARSED SeriesDescription (first 10):")
        print(ct.loc[bad, "SeriesDescription"].head(10).to_string())
        sys.exit(f"{bad.sum()} CT series could not be parsed; nothing written.")

    df = pd.concat([ct.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    if (df.scanner != df.StudyDescription).any():
        sys.exit("scanner tag in SeriesDescription disagrees with StudyDescription for some series")

    unknown = set(df.Manufacturer) - set(VENDOR)
    if unknown:
        sys.exit(f"unmapped manufacturer(s): {unknown}")

    missing_seg = ~df.StudyInstanceUID.isin(seg.index)
    if missing_seg.any():
        print(f"warning: {missing_seg.sum()} CT series have no SEG partner")

    out = pd.DataFrame({
        "series_uid": df.SeriesInstanceUID,
        "study_uid": df.StudyInstanceUID,
        "scanner": df.scanner,                       # 13 scanners: A1, A2, B1, ... H2
        "institution": df.InstitutionName,           # 8 institutions: A..H
        "vendor": df.Manufacturer.map(VENDOR),
        "model": df.ManufacturerModelName,
        "dose_mgy": df.dose.astype(int),             # nominal, from the series description
        "recon": df.recon,                           # FBP | IR | DL
        "repeat": df.repeat.astype(int),             # 1..10
        "kernel": df.ConvolutionKernel,
        "slice_thickness_mm": pd.to_numeric(df.SliceThickness),
        "pixel_spacing_mm": pd.to_numeric(df.PixelSpacing.str.split("\\").str[0]),
        "n_slices": df.n_files.astype(int),
        "ctdivol_header": pd.to_numeric(df.CTDIvol, errors="coerce"),   # empty for GE scanners
        "mas_header": pd.to_numeric(df.Exposure, errors="coerce"),
        "acq_date": df.AcquisitionDate,
        "series_dir": df.series_dir,
        "seg_dir": df.StudyInstanceUID.map(seg.series_dir) if len(seg) else "",
    })
    out = out.sort_values(["scanner", "dose_mgy", "recon", "repeat"]).reset_index(drop=True)
    out.to_csv(dst, index=False)

    print(f"{len(out)} CT series -> {dst}")
    print(f"scanners={out.scanner.nunique()} institutions={out.institution.nunique()} "
          f"vendors={out.vendor.nunique()} doses={[int(d) for d in sorted(out.dose_mgy.unique())]} recons={sorted(out.recon.unique())}")
    print("\nseries per scanner x recon:")
    print(out.groupby(["scanner", "recon"]).size().unstack(fill_value=0).to_string())
    print("\nseries per dose x recon:")
    print(out.groupby(["dose_mgy", "recon"]).size().unstack(fill_value=0).to_string())
    short = out[out.n_slices < out.groupby("scanner").n_slices.transform("max")]
    if len(short):
        print(f"\nseries with fewer slices than the scanner's usual count: {len(short)}")
        print(short[["scanner", "dose_mgy", "recon", "repeat", "n_slices"]].head(10).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
