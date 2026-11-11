from __future__ import annotations

import csv
from pathlib import Path

# HU range kept by the dataset authors' code (Harmonization-Dataset utils.read_dicom: level 500, window 3000),
# and the range their PSNR (max 2000) and SSIM (data_range 3000) are defined on. The cache stores this range.
HU_RANGE = (-1000.0, 2000.0)

# Soft-tissue window: what the models are trained on and what every reported metric is computed in.
# ~88% of all voxels fall inside it, and the main HU peaks agree across scanners, while the bright tail
# (above ~800 HU) differs by hundreds of HU per scanner. Keep it in step with data.params.hu_window.
SOFT_WINDOW = (-160.0, 240.0)

CACHE_INDEX = "cache_index.csv"      # written by scripts/cache_ct4h_volumes.py
CACHE_INDEX_NUMERIC = ("dose_mgy", "repeat", "n_slices", "height", "width")

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


def read_cache_index(cache_dir: Path | str) -> list[dict]:
    """Read cache_index.csv (one row per cached volume) with the numeric columns converted.

    The single reader of the cache index: CT4HPairDataset, the inspection script and the figure script all
    use it, so a column added by scripts/cache_ct4h_volumes.py only has to be handled here.
    """
    path = Path(cache_dir) / CACHE_INDEX
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run scripts/cache_ct4h_volumes.py first.")
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for column in CACHE_INDEX_NUMERIC:
                row[column] = int(row[column])
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


# Splits. One phantom means every scan shows the same anatomy, so a random split would put near-identical
# images (same scanner, different noise realisation) on both sides. Two protocols instead:
#   scanner: whole scanners are held out (generalisation to an unseen scanner).
#   repeat:  every scanner is in every split, the 10 repeats are divided (a seen scanner, an unseen noise draw).
# Proposed defaults, not final. Every vendor stays in train; E2 and F1 (known z-alignment caveats) stay out of test.
SPLIT_MODES = ("scanner", "repeat")
SCANNER_SPLIT = {
    "train": ["A1", "A2", "B2", "G1", "D1", "E2", "F1", "H2", "H1"],
    "val": ["B1", "E1"],
    "test": ["G2", "C1"],
}
REPEAT_SPLIT = {"train": list(range(1, 8)), "val": [8], "test": [9, 10]}
DIRECTIONS = ("any", "up", "down")


def split_selection(mode: str, split: str) -> tuple[list[str], list[int]]:
    """(scanners, repeats) of one split. ``scanner`` mode keeps every repeat, ``repeat`` mode every scanner."""
    if mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {mode!r}")
    if split not in SCANNER_SPLIT:
        raise ValueError(f"split must be one of {sorted(SCANNER_SPLIT)}, got {split!r}")
    if mode == "scanner":
        return list(SCANNER_SPLIT[split]), list(range(1, 11))
    return list(SCANNERS), list(REPEAT_SPLIT[split])


def dose_pairs(directions: str = "any", doses: list[int] | None = None) -> list[tuple[int, int]]:
    """(source_dose, target_dose) pairs. ``up`` = towards a higher dose (denoising), ``down`` = towards a lower
    one (the model has to add noise, which a deterministic regressor cannot do), ``any`` = both."""
    if directions not in DIRECTIONS:
        raise ValueError(f"directions must be one of {DIRECTIONS}, got {directions!r}")
    doses = sorted(doses or DOSES)
    pairs = [(s, t) for s in doses for t in doses if s != t]
    if directions == "up":
        return [(s, t) for s, t in pairs if t > s]
    if directions == "down":
        return [(s, t) for s, t in pairs if t < s]
    return pairs
