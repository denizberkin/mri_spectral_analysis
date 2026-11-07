from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import parse_slice_name

# Soft-tissue-oriented windows, one per anatomy (HU). The chest window keeps lung and
# mediastinum; the liver window is the usual abdominal one. They decide where the noise is
# visible, so the metrics must be computed in the same window.

# WHICH ONE!!! CHECK THIS LATER
DEFAULT_HU_WINDOWS = {"chest": (-1000.0, 400.0), "liver": (-160.0, 240.0)}
#DEFAULT_HU_WINDOWS = {"chest": (-160.0, 240.0), "liver": (-160.0, 240.0)}

def window_to_unit(hu: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip HU to [low, high] and rescale to [0, 1] (float32)."""
    if not high > low:
        raise ValueError(f"HU window must have high > low, got ({low}, {high})")
    clipped = np.clip(hu.astype(np.float32), low, high)
    return (clipped - low) / (high - low)


def unit_to_hu(unit: np.ndarray, low: float, high: float) -> np.ndarray:
    """Inverse of window_to_unit for a [0, 1] array (values outside the window are lost)."""
    return unit * (high - low) + low


class CachedLDCTDataset(Dataset):
    """Matched low-dose -> full-dose 2D slices of one anatomy, for a set of patients.

    Source and target of an item are the same slice of the same patient (matched on z when
    preprocessing), so no registration is involved.

    Args:
        preprocessed_dir: LDCT_PREPROCESSED_DIR.
        anatomy: 'chest' or 'liver'.
        patients: patient ids to include (take them from the index.csv split column).
        source_dose / target_dose: folder names, 'low' and 'full' by default.
        hu_window: (low, high) in HU; defaults to DEFAULT_HU_WINDOWS[anatomy].
        include_slice_pos: also return the normalised axial position of the slice.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        anatomy: str,
        patients: Sequence[str],
        source_dose: str = "low",
        target_dose: str = "full",
        hu_window: Sequence[float] | None = None,
        include_slice_pos: bool = False,
    ):
        self.anatomy = anatomy
        self.hu_window = tuple(float(v) for v in (hu_window or DEFAULT_HU_WINDOWS[anatomy]))
        window_to_unit(np.zeros(1), *self.hu_window)  # validates the window
        self.include_slice_pos = include_slice_pos

        base = Path(preprocessed_dir) / anatomy
        self.pairs: list[tuple[Path, Path]] = []
        for patient in sorted(patients):
            sources = sorted((base / source_dose).glob(f"{patient}_s*.npz"))
            if not sources:
                raise FileNotFoundError(
                    f"No {anatomy}/{source_dose} slices for {patient} in {preprocessed_dir}. "
                    "Run: python scripts/preprocess_ldct.py"
                )
            for source in sources:
                target = base / target_dose / source.name
                if not target.is_file():
                    raise FileNotFoundError(f"{source.name} has no {target_dose}-dose partner at {target}")
                self.pairs.append((source, target))

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, path: Path) -> tuple[torch.Tensor, float]:
        with np.load(path) as npz:
            unit = window_to_unit(npz["image"], *self.hu_window)
            slice_pos = float(npz["slice_pos"])
        return torch.from_numpy(unit).unsqueeze(0) * 2.0 - 1.0, slice_pos

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int]:
        source_path, target_path = self.pairs[index]
        source, slice_pos = self._load(source_path)
        target, _ = self._load(target_path)
        patient, slice_idx = parse_slice_name(source_path)
        item: Dict[str, torch.Tensor | str | int] = {
            "source": source,
            "target": target,
            "patient_id": patient,
            "slice_idx": slice_idx,
        }
        if self.include_slice_pos:
            item["slice_pos"] = torch.tensor(slice_pos, dtype=torch.float32)
        return item
