from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import HU_RANGE, dose_pairs, get_domain, split_selection

CACHE_INDEX = "cache_index.csv"


def window_to_unit(hu: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip HU to [low, high] and rescale to [0, 1] (float32)."""
    if not high > low:
        raise ValueError(f"HU window must have high > low, got ({low}, {high})")
    clipped = np.clip(hu.astype(np.float32), low, high)
    return (clipped - low) / (high - low)


def unit_to_hu(unit: np.ndarray, low: float, high: float) -> np.ndarray:
    """Inverse of window_to_unit for a [0, 1] array (values outside the window are lost)."""
    return unit * (high - low) + low


def neighbour_offsets(neighbours: int) -> list[int]:
    """Slice offsets of a 2.5D stack, centre first: 1 -> [0, -1, 1], 2 -> [0, -1, 1, -2, 2].
    The centre slice is channel 0, so a residual head that adds channel 0 to its output works unchanged."""
    offsets = [0]
    for step in range(1, neighbours + 1):
        offsets += [-step, step]
    return offsets


class CT4HPairDataset(Dataset):
    """Dose-to-dose slice pairs from the registered, cached CT4Harmonization volumes.

    One item = one scanner, one reconstruction, a (source dose, target dose) pair, one repeat for each
    dose, one axial slice. Source and target are different scans of the same phantom, registered to the
    same reference (so they are aligned, but they are not the same acquisition and the noise differs).
    This is a noise-to-noise setup: with L1/L2 the expected output is the mean over the target noise.

    Items are drawn at random: ``length`` of them per epoch, uniformly over scanners, then dose pairs, then
    repeats, then slices. ``deterministic=True`` fixes the draw of item ``i`` (seeded by ``seed`` and ``i``),
    which is what validation and test need. Without it the draws come from a per-worker generator that keeps
    advancing, so every epoch sees new draws (``index`` is then ignored).

    Args:
        cache_dir: folder written by scripts/cache_ct4h_volumes.py (int16 .npy volumes + cache_index.csv).
        split: 'train', 'val' or 'test'.
        split_mode: 'scanner' (whole scanners held out) or 'repeat' (repeats held out, all scanners seen).
        directions: 'any' (all 20 dose pairs), 'up' (towards higher dose) or 'down'.
        doses: restrict the doses taking part (default all five).
        recons: reconstructions to draw from; source and target always share one.
        neighbours: slices on each side stacked as extra input channels (0 = plain 2D).
        hu_window: (low, high) HU mapped to [-1, 1]. Default is the authors' range (-1000, 2000).
        target_mode: 'repeat' (one noisy target repeat) or 'mean' (average of the split's target-dose repeats,
            the low-noise reference to evaluate against).
        pad_multiple: pad height and width up to a multiple of this with air (-1) so a UNet with ``levels``
            down-samplings accepts whole slices. ``orig_height`` / ``orig_width`` give the crop back.
        z_range: (first, last+1) slice indices to draw the centre slice from; default the whole volume.
        scanners / repeats: override the split's scanners / repeats (e.g. to score a seen-scanner and an unseen-scanner
            model on exactly the same scans). The split still decides everything else.
        include_slice_pos: also return the normalised axial position (only for a model with slice_conditioning).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        split_mode: str = "scanner",
        directions: str = "any",
        doses: Sequence[int] | None = None,
        recons: Sequence[str] = ("FBP",),
        neighbours: int = 1,
        hu_window: Sequence[float] = HU_RANGE,
        target_mode: str = "repeat",
        domain_scheme: str = "dose",
        pad_multiple: int = 16,
        z_range: Sequence[int] | None = None,
        scanners: Sequence[str] | None = None,
        repeats: Sequence[int] | None = None,
        include_slice_pos: bool = False,
        length: int = 10000,
        deterministic: bool = False,
        seed: int = 0,
        max_open_volumes: int = 48,
    ):
        if target_mode not in ("repeat", "mean"):
            raise ValueError("target_mode must be 'repeat' or 'mean'")
        if neighbours < 0:
            raise ValueError("neighbours must be >= 0")
        self.cache_dir = Path(cache_dir)
        self.hu_window = tuple(float(v) for v in hu_window)
        window_to_unit(np.zeros(1), *self.hu_window)  # validates the window
        self.offsets = neighbour_offsets(neighbours)
        self.target_mode = target_mode
        self.domain_scheme = domain_scheme
        self.pad_multiple = int(pad_multiple)
        self.include_slice_pos = include_slice_pos
        self.length = int(length)
        self.deterministic = deterministic
        self.seed = int(seed)
        self.max_open = int(max_open_volumes)
        self._open: OrderedDict[str, np.ndarray] = OrderedDict()
        self._generator: np.random.Generator | None = None
        self._generator_pid = -1

        split_scanners, split_repeats = split_selection(split_mode, split)
        scanners = list(scanners) if scanners else split_scanners
        repeats = list(repeats) if repeats else split_repeats
        rows = self._read_cache_index(self.cache_dir)
        wanted_pairs = dose_pairs(directions, list(doses) if doses else None)
        wanted_doses = {d for pair in wanted_pairs for d in pair}

        # (scanner, recon, dose) -> {repeat: relative path}
        self.volumes: dict[tuple[str, str, int], dict[int, str]] = {}
        shapes = set()
        for row in rows:
            if row["scanner"] in scanners and row["recon"] in recons and row["repeat"] in repeats \
                    and row["dose_mgy"] in wanted_doses:
                key = (row["scanner"], row["recon"], row["dose_mgy"])
                self.volumes.setdefault(key, {})[row["repeat"]] = row["path"]
                shapes.add((row["n_slices"], row["height"], row["width"]))
        if not self.volumes:
            raise FileNotFoundError(
                f"No cached volumes for split={split!r} ({split_mode}), recons={list(recons)} in {self.cache_dir}"
            )
        if len(shapes) != 1:
            raise ValueError(f"cached volumes differ in shape: {sorted(shapes)}")
        self.n_slices, self.height, self.width = next(iter(shapes))
        self.z_lo, self.z_hi = (0, self.n_slices) if z_range is None else (int(z_range[0]), int(z_range[1]))
        if not 0 <= self.z_lo < self.z_hi <= self.n_slices:
            raise ValueError(f"z_range {z_range} outside 0..{self.n_slices}")

        # one entry per (scanner, recon) that has at least one usable dose pair
        self.groups: list[tuple[str, str, list[tuple[int, int]]]] = []
        for scanner in scanners:
            for recon in recons:
                have = {d for (s, r, d) in self.volumes if s == scanner and r == recon}
                usable = [(s, t) for s, t in wanted_pairs if s in have and t in have]
                if usable:
                    self.groups.append((scanner, recon, usable))
        if not self.groups:
            raise FileNotFoundError("no scanner has two of the requested doses cached")
        self.scanners = sorted({g[0] for g in self.groups})

    # -- cache index ---------------------------------------------------------------------------------
    @staticmethod
    def _read_cache_index(cache_dir: Path) -> list[dict]:
        import csv

        path = cache_dir / CACHE_INDEX
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run scripts/cache_ct4h_volumes.py first.")
        rows = []
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for column in ("dose_mgy", "repeat", "n_slices", "height", "width"):
                    row[column] = int(row[column])
                rows.append(row)
        return rows

    # -- volumes -------------------------------------------------------------------------------------
    def _volume(self, rel_path: str) -> np.ndarray:
        volume = self._open.get(rel_path)
        if volume is None:
            volume = np.load(self.cache_dir / rel_path, mmap_mode="r")
            self._open[rel_path] = volume
            while len(self._open) > self.max_open:
                self._open.popitem(last=False)
        else:
            self._open.move_to_end(rel_path)
        return volume

    def _unit(self, hu: np.ndarray) -> np.ndarray:
        return window_to_unit(hu, *self.hu_window) * 2.0 - 1.0

    def _pad(self, image: np.ndarray) -> np.ndarray:
        multiple = self.pad_multiple
        if multiple <= 1:
            return image
        pad_h = (-image.shape[-2]) % multiple
        pad_w = (-image.shape[-1]) % multiple
        if not (pad_h or pad_w):
            return image
        return np.pad(image, [(0, 0)] * (image.ndim - 2) + [(0, pad_h), (0, pad_w)], constant_values=-1.0)

    # -- items ---------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return self.length

    def _rng(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng([self.seed, index])
        # One generator per process whose state advances with every item. It must not be re-seeded per index:
        # DataLoader(persistent_workers=True) keeps each worker's torch.initial_seed() for the whole run and
        # hands a given index to the same worker every epoch, so an index-seeded draw would repeat exactly.
        # The pid is checked so a generator created before the workers forked is not shared by all of them.
        pid = os.getpid()
        if self._generator is None or self._generator_pid != pid:
            self._generator = np.random.default_rng([self.seed, torch.initial_seed() % (2**32), pid])
            self._generator_pid = pid
        return self._generator

    def draw(self, index: int) -> dict:
        """The random choices of one item, without reading any data (useful for tests and sampling stats)."""
        rng = self._rng(index)
        scanner, recon, pairs = self.groups[int(rng.integers(len(self.groups)))]
        source_dose, target_dose = pairs[int(rng.integers(len(pairs)))]
        source_repeats = sorted(self.volumes[(scanner, recon, source_dose)])
        target_repeats = sorted(self.volumes[(scanner, recon, target_dose)])
        return {
            "scanner": scanner,
            "recon": recon,
            "source_dose": source_dose,
            "target_dose": target_dose,
            "source_repeat": int(rng.choice(source_repeats)),
            "target_repeat": int(rng.choice(target_repeats)),
            "z": int(rng.integers(self.z_lo, self.z_hi)),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int]:
        choice = self.draw(index)
        scanner, recon, z = choice["scanner"], choice["recon"], choice["z"]

        source_volume = self._volume(self.volumes[(scanner, recon, choice["source_dose"])][choice["source_repeat"]])
        z_stack = np.clip(z + np.asarray(self.offsets), 0, self.n_slices - 1)  # edge slices repeat
        source = self._unit(np.stack([source_volume[k] for k in z_stack]))

        target_files = self.volumes[(scanner, recon, choice["target_dose"])]
        if self.target_mode == "mean":
            target_hu = np.mean([self._volume(path)[z].astype(np.float32) for path in target_files.values()], axis=0)
        else:
            target_hu = self._volume(target_files[choice["target_repeat"]])[z]
        target = self._unit(target_hu)[None]

        item: Dict[str, torch.Tensor | str | int] = {
            "source": torch.from_numpy(np.ascontiguousarray(self._pad(source))),
            "target": torch.from_numpy(np.ascontiguousarray(self._pad(target))),
            "source_domain": get_domain(scanner, choice["source_dose"], self.domain_scheme),
            "target_domain": get_domain(scanner, choice["target_dose"], self.domain_scheme),
            "scanner": scanner,
            "recon": recon,
            "source_dose": choice["source_dose"],
            "target_dose": choice["target_dose"],
            "source_repeat": choice["source_repeat"],
            "target_repeat": choice["target_repeat"] if self.target_mode == "repeat" else 0,
            "slice_idx": z,
            "orig_height": self.height,
            "orig_width": self.width,
        }
        if self.include_slice_pos:
            item["slice_pos"] = torch.tensor(z / max(self.n_slices - 1, 1), dtype=torch.float32)
        return item
