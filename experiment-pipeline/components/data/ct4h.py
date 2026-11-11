from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ct4h.data.cached_dataset import CT4HPairDataset
from ct4h.data.utils import HU_RANGE, NUM_DOMAINS, split_selection
from ct4h.env import get_ct4h_cache_dir
from eval_pipeline.components.data.base import DataModule
from eval_pipeline.registry import register_component


class _Augment(Dataset):
    """Paired augmentation: one crop window and one flip for the source stack and the target together."""

    def __init__(self, dataset: Dataset, horizontal_flip: float, patch_size: int) -> None:
        self.dataset = dataset
        self.horizontal_flip = horizontal_flip
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        source, target = sample["source"], sample["target"]
        if self.patch_size:
            height, width = source.shape[-2:]
            top = int(torch.randint(0, max(height - self.patch_size, 0) + 1, ()))
            left = int(torch.randint(0, max(width - self.patch_size, 0) + 1, ()))
            source = source[..., top : top + self.patch_size, left : left + self.patch_size]
            target = target[..., top : top + self.patch_size, left : left + self.patch_size]
        if torch.rand(()) < self.horizontal_flip:
            source, target = source.flip(-1), target.flip(-1)
        sample["source"], sample["target"] = source, target
        return sample


@register_component("ct4h_paired", category="data")
class CT4HDataModule(DataModule[dict[str, DataLoader]]):
    """Dose-conditioned any-to-any CT loaders (CT4Harmonization-Multicentric, registered volumes).

    ``split_mode`` decides what is held out: 'scanner' (unseen scanners, see ct4h.data.utils.SCANNER_SPLIT) or
    'repeat' (unseen noise realisations on seen scanners, REPEAT_SPLIT). ``directions`` picks the dose pairs:
    'any' (20), 'up' (10, denoising) or 'down' (10). The domain is the dose, and must agree with the model's
    ``num_domains`` (5 for 'dose').

    Training draws ``train_samples`` random items per epoch; validation draws a fixed ``validation_samples``.
    Both use one noisy target repeat; evaluate against the repeat mean (target_mode = 'mean') separately.
    """

    def setup(self) -> dict[str, DataLoader]:
        p = self.params
        scheme = str(p.get("domain_scheme", "dose"))
        if scheme not in NUM_DOMAINS:
            raise ValueError(f"domain_scheme must be one of {sorted(NUM_DOMAINS)}")
        split_mode = str(p.get("split_mode", "scanner"))
        split_selection(split_mode, "train")  # validates the mode
        batch_size = int(p.get("batch_size", 16))
        num_workers = int(p.get("num_workers", 8))
        seed = int(p.get("seed", 0))
        patch_size = int(p.get("patch_size", 0))
        horizontal_flip = float(p.get("horizontal_flip", 0.0))
        if not 0.0 <= horizontal_flip <= 1.0:
            raise ValueError("horizontal_flip must be between 0 and 1")

        cache_dir = p.get("cache_dir") or get_ct4h_cache_dir()
        if not cache_dir or not Path(cache_dir).is_dir():
            raise FileNotFoundError(
                f"CT4H cache not found: {cache_dir!r}. Set CT4H_CACHE_DIR or data.params.cache_dir, "
                "after running scripts/cache_ct4h_volumes.py"
            )
        common = dict(
            cache_dir=cache_dir,
            split_mode=split_mode,
            directions=str(p.get("directions", "any")),
            doses=p.get("doses") or None,
            recons=list(p.get("recons", ["FBP"])),
            neighbours=int(p.get("neighbours", 1)),
            hu_window=tuple(p.get("hu_window", HU_RANGE)),
            domain_scheme=scheme,
            pad_multiple=int(p.get("pad_multiple", 16)),
            z_range=p.get("z_range") or None,
            include_slice_pos=bool(p.get("include_slice_pos", False)),
            seed=seed,
        )
        train = CT4HPairDataset(
            split="train", length=int(p.get("train_samples", 20000)), deterministic=False, target_mode="repeat", **common
        )
        data = {
            "train": self._loader(
                _Augment(train, horizontal_flip, patch_size),
                batch_size=batch_size, num_workers=num_workers, shuffle=False, seed=seed,
            )
        }
        # No augmentation, whole (padded) slices, a fixed draw: the checkpoint is chosen on a comparable signal.
        validation = CT4HPairDataset(
            split="val", length=int(p.get("validation_samples", 400)), deterministic=True,
            target_mode=str(p.get("validation_target", "repeat")), **common,
        )
        data["validation"] = self._loader(
            validation, batch_size=batch_size, num_workers=num_workers, shuffle=False, seed=seed + 1
        )
        return data

    @staticmethod
    def _loader(dataset: Dataset, *, batch_size: int, num_workers: int, shuffle: bool, seed: int) -> DataLoader:
        # train items are already random draws (index only seeds them), so shuffle stays False
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(seed),
        )
