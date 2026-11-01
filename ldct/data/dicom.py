from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom

HU_DTYPE = np.int16


def _slice_z(dataset: pydicom.Dataset) -> float:
    if "ImagePositionPatient" in dataset:
        return float(dataset.ImagePositionPatient[2])
    if "SliceLocation" in dataset:
        return float(dataset.SliceLocation)
    raise ValueError("DICOM slice has neither ImagePositionPatient nor SliceLocation")


def read_series(series_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a series as (volume, z, info).

    volume: (N, H, W) int16 Hounsfield units, ordered by ascending z.
    z:      (N,) float64 slice positions in mm.
    info:   series_uid, slice_thickness, pixel_spacing.

    Raises ValueError when the folder mixes series, repeats a z position, or has slices of
    different sizes: any of those means the folder is not the single series it should be.
    """
    series_dir = Path(series_dir)
    files = sorted(p for p in series_dir.iterdir() if p.suffix.lower() == ".dcm")
    if not files:
        raise FileNotFoundError(f"No .dcm files in {series_dir}")

    slices = []
    for path in files:
        dataset = pydicom.dcmread(path)
        slices.append((_slice_z(dataset), dataset))
    slices.sort(key=lambda item: item[0])

    uids = {str(ds.SeriesInstanceUID) for _, ds in slices}
    if len(uids) != 1:
        raise ValueError(f"{series_dir} holds {len(uids)} different series")
    z = np.array([item[0] for item in slices], dtype=np.float64)
    if len(np.unique(np.round(z, 3))) != len(z):
        raise ValueError(f"{series_dir} has repeated slice positions")

    volume = np.empty((len(slices), int(slices[0][1].Rows), int(slices[0][1].Columns)), dtype=HU_DTYPE)
    for index, (_, dataset) in enumerate(slices):
        if (int(dataset.Rows), int(dataset.Columns)) != volume.shape[1:]:
            raise ValueError(f"{series_dir} has slices of different sizes")
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        hu = dataset.pixel_array.astype(np.float32) * slope + intercept
        volume[index] = np.clip(np.rint(hu), np.iinfo(HU_DTYPE).min, np.iinfo(HU_DTYPE).max)

    first = slices[0][1]
    info = {
        "series_uid": next(iter(uids)),
        "slice_thickness": float(getattr(first, "SliceThickness", np.nan)),
        "pixel_spacing": float(first.PixelSpacing[0]) if "PixelSpacing" in first else float("nan"),
    }
    return volume, z, info


def match_slices(z_a: np.ndarray, z_b: np.ndarray, tol: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one match of two ascending z arrays: indices (ia, ib) with |z_a[ia] - z_b[ib]| <= tol.

    Both doses of a Siemens patient are reconstructed from one scan on the same grid, so the
    positions normally agree to the DICOM rounding (~0.001 mm). Slices without a partner
    are dropped; the caller decides how many drops are acceptable.
    """
    if len(z_a) == 0 or len(z_b) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    position = np.searchsorted(z_b, z_a)
    candidates = np.stack([np.clip(position - 1, 0, len(z_b) - 1), np.clip(position, 0, len(z_b) - 1)])
    distance = np.abs(z_b[candidates] - z_a)
    nearest = candidates[np.argmin(distance, axis=0), np.arange(len(z_a))]
    keep = np.nonzero(distance.min(axis=0) <= tol)[0]
    _, first = np.unique(nearest[keep], return_index=True)
    keep = keep[first]
    return keep, nearest[keep]
