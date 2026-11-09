from __future__ import annotations

import csv
from pathlib import Path

# HU range kept by the dataset authors' code (Harmonization-Dataset utils.read_dicom: level 500, window 3000),
# and the range their PSNR (max 2000) and SSIM (data_range 3000) are defined on. Use it everywhere.
HU_RANGE = (-1000.0, 2000.0)

DOSES = [1, 3, 6, 10, 14]          # nominal mGy, from the series description
RECONS = ["FBP", "IR", "DL"]       # DL exists only on scanners D1 and E2
SCANNERS = ["A1", "A2", "B1", "B2", "C1", "D1", "E1", "E2", "F1", "G1", "G2", "H1", "H2"]
VENDORS = {
    "A1": "Siemens", "A2": "Siemens", "B1": "Siemens", "B2": "Siemens", "G1": "Siemens", "G2": "Siemens",
    "C1": "Philips", "H2": "Philips",
    "D1": "GE", "E2": "GE", "F1": "GE",
    "E1": "Toshiba", "H1": "Toshiba",
}

INDEX_NUMERIC = {
    "dose_mgy": int, "repeat": int, "n_slices": int,
    "slice_thickness_mm": float, "pixel_spacing_mm": float,
}

# Domain schemes for the conditional UNet's source/target embeddings.
#   dose:          one domain per dose level (5). The scanner is not told to the model.
#   scanner_dose:  one domain per (scanner, dose) (65); the model knows which scanner it is looking at.
DOMAIN_SCHEMES = ("dose", "scanner_dose")
NUM_DOMAINS = {"dose": len(DOSES), "scanner_dose": len(SCANNERS) * len(DOSES)}


def get_domain(scanner: str, dose: int, scheme: str = "dose") -> int:
    """Map (scanner, dose) to a domain index in [0, NUM_DOMAINS[scheme])."""
    if scheme not in NUM_DOMAINS:
        raise ValueError(f"domain scheme must be one of {sorted(NUM_DOMAINS)}, got {scheme!r}")
    if dose not in DOSES:
        raise ValueError(f"unknown dose {dose!r}; expected one of {DOSES}")
    if scanner not in SCANNERS:
        raise ValueError(f"unknown scanner {scanner!r}; expected one of {SCANNERS}")
    if scheme == "dose":
        return DOSES.index(dose)
    return SCANNERS.index(scanner) * len(DOSES) + DOSES.index(dose)


def read_index(path: Path | str) -> list[dict]:
    """Read ct4h_index.csv (one row per series) with the numeric columns converted."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. It lists every downloaded series (see ct4h/README.md).")
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for column, kind in INDEX_NUMERIC.items():
                row[column] = kind(row[column])
            rows.append(row)
    return rows


def select(rows: list[dict], *, scanners=None, doses=None, recons=None, repeats=None) -> list[dict]:
    """Rows matching every given filter (None keeps everything)."""
    def keep(row: dict) -> bool:
        return (
            (scanners is None or row["scanner"] in scanners)
            and (doses is None or row["dose_mgy"] in doses)
            and (recons is None or row["recon"] in recons)
            and (repeats is None or row["repeat"] in repeats)
        )
    return [row for row in rows if keep(row)]
