from __future__ import annotations

import csv
import random
import re
from pathlib import Path

ANATOMIES = ["chest", "liver"]

PATIENT_PREFIX = {"C": "chest", "L": "liver"}
PATIENT_RE = re.compile(r"^[CL]\d{3}$")

UNPAIRED = "unpaired"  # full-dose only (GE): no low-dose partner, never used for paired training

DOMAIN_SCHEMES = {
    "dose": {("chest", "low"): 0, ("liver", "low"): 1, ("chest", "full"): 2, ("liver", "full"): 2},
    "anatomy_dose": {("chest", "low"): 0, ("chest", "full"): 1, ("liver", "low"): 2, ("liver", "full"): 3},
}
NUM_DOMAINS = {name: len(set(table.values())) for name, table in DOMAIN_SCHEMES.items()}

INDEX_COLUMNS = [
    "anatomy", "patient_id", "dose", "split", "paired", "series_uid", "series_dir",
    "n_slices", "manufacturer", "model", "kernel", "series_description",
    "slice_thickness", "pixel_spacing", "rows", "columns",
]

# Series descriptions in the release name the dose ("Low Dose Images", "Full Dose Images");
# the patterns are arguments of the index builder so a different wording needs no code change.
LOW_PATTERN = r"\b(low|quarter|reduced|simulated)\b"
FULL_PATTERN = r"\b(full|routine|normal)\b"


def anatomy_of(patient_id: str) -> str:
    """'C001' -> 'chest', 'L017' -> 'liver'."""
    if not PATIENT_RE.match(patient_id):
        raise ValueError(f"Not a chest/liver patient id: {patient_id!r}")
    return PATIENT_PREFIX[patient_id[0]]


def get_domain(anatomy: str, dose: str, scheme: str = "dose") -> int:
    """Map (anatomy, dose) to a domain index in [0, NUM_DOMAINS[scheme])."""
    try:
        return DOMAIN_SCHEMES[scheme][(anatomy, dose)]
    except KeyError:
        raise ValueError(f"Unknown domain scheme/anatomy/dose: {scheme!r}, {anatomy!r}, {dose!r}") from None


def classify_dose(
    series_description: str, low_pattern: str = LOW_PATTERN, full_pattern: str = FULL_PATTERN
) -> str | None:
    """'low' / 'full' from a series description, or None when it names neither."""
    text = series_description.lower()
    if re.search(low_pattern, text):
        return "low"
    if re.search(full_pattern, text):
        return "full"
    return None


def slice_name(patient_id: str, slice_idx: int) -> str:
    """File stem of one preprocessed slice; identical for the low- and full-dose partner."""
    return f"{patient_id}_s{slice_idx:03d}"


def parse_slice_name(path: Path | str) -> tuple[str, int]:
    """Inverse of slice_name: path -> (patient_id, slice_idx)."""
    match = re.fullmatch(r"([CL]\d{3})_s(\d+)", Path(path).stem)
    if match is None:
        raise ValueError(f"Not a preprocessed LDCT slice name: {Path(path).name}")
    return match.group(1), int(match.group(2))


def assign_splits(
    paired_patients: dict[str, list[str]], val_per_anatomy: int, test_per_anatomy: int, seed: int
) -> dict[str, str]:
    """Patient-level split, drawn separately inside each anatomy.

    ``paired_patients`` maps anatomy -> patients that have both doses. Slices of one patient
    are strongly correlated, so a random slice split would leak; every slice of a patient
    goes to the same split.
    """
    splits: dict[str, str] = {}
    for anatomy in sorted(paired_patients):
        patients = sorted(paired_patients[anatomy])
        if val_per_anatomy + test_per_anatomy >= len(patients):
            raise ValueError(
                f"{anatomy}: {len(patients)} paired patients cannot give {val_per_anatomy} val + "
                f"{test_per_anatomy} test and still leave training data"
            )
        random.Random(f"{seed}-{anatomy}").shuffle(patients)
        test = patients[:test_per_anatomy]
        val = patients[test_per_anatomy : test_per_anatomy + val_per_anatomy]
        for patient in patients:
            splits[patient] = "test" if patient in test else "val" if patient in val else "train"
    return splits


def write_index(rows: list[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_index(path: Path | str) -> list[dict]:
    """Read index.csv back with ``paired`` as bool and the numeric columns converted."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/build_ldct_index.py")
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["paired"] = row["paired"] == "True"
            row["n_slices"] = int(row["n_slices"])
            row["rows"] = int(row["rows"])
            row["columns"] = int(row["columns"])
            row["slice_thickness"] = float(row["slice_thickness"]) if row["slice_thickness"] else float("nan")
            rows.append(row)
    return rows


def patients_in(rows: list[dict], anatomy: str, split: str) -> list[str]:
    """Paired patients of one anatomy in one split, sorted."""
    return sorted({r["patient_id"] for r in rows if r["anatomy"] == anatomy and r["split"] == split})
