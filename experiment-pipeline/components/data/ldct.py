from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from eval_pipeline.components.data.base import DataModule
from eval_pipeline.registry import register_component
from ldct.data.cached_dataset import DEFAULT_HU_WINDOWS, CachedLDCTDataset
from ldct.data.utils import ANATOMIES, NUM_DOMAINS, get_domain, patients_in, read_index
from ldct.env import get_ldct_preprocessed_dir


class _DoseDomainDataset(Dataset):
    """Adds the dose domains and the paired augmentations the trainer expects."""

    def __init__(
        self,
        dataset: Dataset,
        source_domain: int,
        target_domain: int,
        horizontal_flip: float,
        patch_size: int,
    ) -> None:
        self.dataset = dataset
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.horizontal_flip = horizontal_flip
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        source, target = sample["source"], sample["target"]
        if self.patch_size:
            # one window for both images: they are the same slice, so the crop must match
            height, width = source.shape[-2:]
            top = int(torch.randint(0, max(height - self.patch_size, 0) + 1, ()))
            left = int(torch.randint(0, max(width - self.patch_size, 0) + 1, ()))
            source = source[..., top : top + self.patch_size, left : left + self.patch_size]
            target = target[..., top : top + self.patch_size, left : left + self.patch_size]
        if torch.rand(()) < self.horizontal_flip:
            source, target = source.flip(-1), target.flip(-1)
        sample["source"], sample["target"] = source, target
        sample["source_domain"] = self.source_domain
        sample["target_domain"] = self.target_domain
        return sample


@register_component("ldct_paired", category="data")
class LDCTDataModule(DataModule[dict[str, DataLoader]]):
    """Low-dose -> full-dose loaders, split by patient.

    The patient split is the ``split`` column of index.csv (scripts/build_ldct_index.py),
    so every run and every model sees the same train/val/test patients. Only the paired
    (Siemens) patients take part.

    The domain is the dose: source_domain / target_domain come from ``domain_scheme``
    (see ldct.data.utils.DOMAIN_SCHEMES) and must agree with the model's ``num_domains``.
    """

    def setup(self) -> dict[str, DataLoader]:
        anatomies = list(self.params.get("anatomies", ANATOMIES))
        unknown = set(anatomies) - set(ANATOMIES)
        if unknown:
            raise ValueError(f"Unknown anatomies: {sorted(unknown)}")
        scheme = str(self.params.get("domain_scheme", "dose"))
        if scheme not in NUM_DOMAINS:
            raise ValueError(f"domain_scheme must be one of {sorted(NUM_DOMAINS)}")
        source_dose = str(self.params.get("source_dose", "low"))
        target_dose = str(self.params.get("target_dose", "full"))
        batch_size = int(self.params.get("batch_size", 8))
        num_workers = int(self.params.get("num_workers", 4))
        seed = int(self.params.get("seed", 0))
        patch_size = int(self.params.get("patch_size", 0))
        horizontal_flip = float(self.params.get("horizontal_flip", 0.0))
        include_slice_pos = bool(self.params.get("include_slice_pos", False))
        if not 0.0 <= horizontal_flip <= 1.0:
            raise ValueError("horizontal_flip must be between 0 and 1")

        preprocessed_dir = self.params.get("preprocessed_dir") or get_ldct_preprocessed_dir()
        if not preprocessed_dir:
            raise ValueError("ldct_paired requires LDCT_PREPROCESSED_DIR or data.params.preprocessed_dir")
        if not Path(preprocessed_dir).is_dir():
            raise FileNotFoundError(
                f"Preprocessed data not found: {preprocessed_dir}. Run scripts/build_ldct_index.py, "
                "then scripts/preprocess_ldct.py"
            )
        rows = read_index(Path(preprocessed_dir) / "index.csv")

        windows = {a: tuple(w) for a, w in dict(self.params.get("hu_windows", {})).items()}
        windows = {a: windows.get(a, DEFAULT_HU_WINDOWS[a]) for a in anatomies}

        def datasets(split: str, flip: float, crop: int) -> list[Dataset]:
            parts: list[Dataset] = []
            for anatomy in anatomies:
                patients = patients_in(rows, anatomy, split)
                if not patients:
                    continue
                inner = CachedLDCTDataset(
                    preprocessed_dir, anatomy, patients, source_dose, target_dose,
                    hu_window=windows[anatomy], include_slice_pos=include_slice_pos,
                )
                parts.append(
                    _DoseDomainDataset(
                        inner,
                        get_domain(anatomy, source_dose, scheme),
                        get_domain(anatomy, target_dose, scheme),
                        flip,
                        crop,
                    )
                )
            return parts

        train = datasets("train", horizontal_flip, patch_size)
        if not train:
            raise FileNotFoundError(f"No training patients for {anatomies} in {preprocessed_dir}/index.csv")
        data = {
            "train": self._loader(
                ConcatDataset(train), batch_size=batch_size, num_workers=num_workers, shuffle=True, seed=seed
            )
        }

        # Validation patients are never trained on, so this is the local overfitting signal.
        # No augmentation and no crop: the checkpoint is chosen on whole slices.
        validation = datasets("val", 0.0, 0)
        if validation:
            data["validation"] = self._loader(
                self._subsample(ConcatDataset(validation), int(self.params.get("validation_samples", 0))),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,  # a fixed order keeps the loss comparable across epochs
                seed=seed + 1,
            )
        return data

    @staticmethod
    def _subsample(dataset: Dataset, samples: int) -> Dataset:
        """Evenly thin the validation fold to at most ``samples`` slices; 0 keeps it whole."""
        if samples <= 0 or len(dataset) <= samples:
            return dataset
        return Subset(dataset, range(0, len(dataset), -(-len(dataset) // samples)))

    @staticmethod
    def _loader(dataset: Dataset, *, batch_size: int, num_workers: int, shuffle: bool, seed: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(seed),
        )
