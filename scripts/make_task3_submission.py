#!/usr/bin/env python3
"""
Build a Task 3 validation-phase submission ZIP from a ConditionalUNet checkpoint.

Usage (the submitted model: mean of stage-2 epochs 6-8, 4-flip TTA):
    python scripts/make_task3_submission.py \
        --checkpoint experiment-pipeline/runs/task3_mc_ssim_slice/artifacts/avg_e6_e8.pt \
        --tta --name task3_mc_ssim_slice_avg_tta

    python scripts/make_task3_submission.py --checkpoint ... --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiment-pipeline")]

from mrixfields.data.transforms import CenterCropOrPad  # noqa: E402
from mrixfields.data.utils import get_joint_domain, load_nifti  # noqa: E402
from mrixfields.zclip_constants import Z_CLIP_RANGE  # noqa: E402
from mrixfields.data.cached_dataset import SLICE_INDEX_RANGE  # noqa: E402

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")

#: Validation subject ids per source field. They are disjoint across fields by design, so the ids in
#: an output filename always come from the source folder even though the name carries the target.
SOURCE_IDS = {
    "0.1T": ("0001", "0002", "0003"),
    "1.5T": ("0004", "0005", "0008"),
    "3T": ("0010", "0011", "0012"),
    "5T": ("0013", "0014", "0015"),
    "7T": ("0016", "0017", "0018"),
}

CROP_SIZE = (368, 448)
BACKGROUND_THRESHOLD = 1e-3
EXPECTED_SHAPE = (364, 436, Z_CLIP_RANGE[1] - Z_CLIP_RANGE[0])


def find_split(data_dir: Path) -> Path:
    """Validation split directory, tolerating the dataset's inconsistent casing."""
    # "validation_prospective" is how the split is named on the team's Drive copy.
    for name in ("Validating_prospective", "validating_prospective", "validation_prospective"):
        candidate = data_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No validation split under {data_dir}")


def derive_shape(state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Read (base_channels, max_channels, levels) back out of a state dict.

    Hardcoding these invites a silent mismatch whenever a config is edited
    between training and packaging, so infer them instead:
      base_channels  = width of the first encoder conv
      levels         = number of encoder blocks
      max_channels   = bottleneck width, which reproduces the channel schedule
                       whether or not the cap actually binds
    """
    base_channels = state["encoders.0.0.weight"].shape[0]
    levels = len({key.split(".")[1] for key in state if key.startswith("encoders.")})
    max_channels = state["bottleneck.0.weight"].shape[0]
    return int(base_channels), int(max_channels), int(levels)


def derive_unet_flags(state: dict[str, torch.Tensor]) -> tuple[bool, bool, bool]:
    """Read ConditionalUNet's three architecture flags back out of a state dict.

    Both are visible in the tensors, so neither is taken on trust: a residual
    checkpoint carries residual_head.weight where a Tanh one carries
    output.0.weight, and per-scale FiLM adds film_projections.*. A wrong guess
    would fail in load_state_dict rather than pass silently -- both flags move
    key names -- but there is no reason to guess in the first place.
    """
    residual_output = "residual_head.weight" in state
    film_conditioning = any(key.startswith("film_projections.") for key in state)
    # 39: axial-position conditioning adds slice_embedding.*. Same reasoning as the other
    # two -- it is visible in the tensors, so read it rather than take it on trust. Without
    # this the keys would be unexpected at load and the position input silently ignored.
    slice_conditioning = any(key.startswith("slice_embedding.") for key in state)
    return residual_output, film_conditioning, slice_conditioning


def build_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Construct the ConditionalUNet the checkpoint was trained as and load it strictly.

    Every architecture choice is read off the tensors rather than taken from a config,
    so a checkpoint can never be loaded into a differently-shaped model.
    """
    from components.models.conditional_unet import ConditionalUNet

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    weights = state.get("model", state)
    base_channels, max_channels, levels = derive_shape(weights)
    residual_output, film_conditioning, slice_conditioning = derive_unet_flags(weights)
    # Input width is visible in the first encoder conv, so a 4-channel multi-contrast
    # checkpoint builds the model it was trained as instead of failing in
    # load_state_dict against the 1-channel default.
    input_channels = int(weights["encoders.0.0.weight"].shape[1])
    print(
        f"derived from checkpoint: base_channels={base_channels} "
        f"max_channels={max_channels} levels={levels} residual_output={residual_output} "
        f"film_conditioning={film_conditioning} input_channels={input_channels} "
        f"slice_conditioning={slice_conditioning}",
        flush=True,
    )
    model = ConditionalUNet(
        input_channels=input_channels,
        base_channels=base_channels,
        max_channels=max_channels,
        levels=levels,
        residual_output=residual_output,
        film_conditioning=film_conditioning,
        slice_conditioning=slice_conditioning,
    )
    model.load_state_dict(weights)
    return model.to(device).eval()


def predict_slab(
    model: torch.nn.Module,
    volume: np.ndarray,
    source_domain: int,
    target_domain: int,
    device: torch.device,
    batch_size: int,
    tta: bool = False,
    auxiliary: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Predict the submission slab only. Copied from
    experiment-pipeline/scripts/inference_task3_conditional_unet.py so the two
    paths cannot drift in intensity handling or cropping.

    ``tta`` averages the four in-plane flips, matching scripts/eval_holdout.py so a
    gain measured locally is the same computation that ships. Flips only, not the
    eight-way dihedral: the data modules augment with flips, never rotations, so a
    rotated brain is out of distribution.

    ``auxiliary`` carries the other contrasts of the same subject at the same field, in
    the fixed (T1W, T2W, T2FLAIR) order of CachedMultiContrastDataset. They become
    channels 1..3 behind ``volume`` itself on channel 0, which is the layout the
    multi-contrast model was trained on -- channel 0 duplicates the contrast being
    predicted, and the duplication is what let the single-channel weights transfer."""
    start, stop = Z_CLIP_RANGE
    crop = CenterCropOrPad(CROP_SIZE)
    uncrop = CenterCropOrPad(volume.shape[:2])
    output = np.zeros_like(volume, dtype=np.float32)

    for batch_start in range(start, stop, batch_size):
        indices = range(batch_start, min(batch_start + batch_size, stop))
        channels = [volume, *auxiliary] if auxiliary else [volume]
        slices = np.stack([[crop(channel[:, :, index]) for channel in channels]
                           for index in indices])
        images = torch.from_numpy(slices).to(device=device, dtype=torch.float32)
        images = images.mul(2).sub(1)
        source = torch.full((len(slices),), source_domain, device=device, dtype=torch.long)
        target = torch.full((len(slices),), target_domain, device=device, dtype=torch.long)
        # 39: the z index of each slice, normalised on the same 72..291 range the training
        # files carry in their _s<idx> suffix. `indices` are volume z indices, which is the
        # very quantity slice_position() parses out of a filename, so the two agree by
        # construction. None for a model without slice conditioning, which ignores it.
        positions = None
        if getattr(model, "slice_embedding", None) is not None:
            low, high = SLICE_INDEX_RANGE
            positions = torch.tensor(
                [min(max((index - low) / (high - low), 0.0), 1.0) for index in indices],
                device=device, dtype=torch.float32,
            )
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            if tta:
                accumulated = None
                for dims in ((), (-1,), (-2,), (-2, -1)):
                    view = torch.flip(images, dims) if dims else images
                    result = model(view, target, source, positions)
                    result = torch.flip(result, dims) if dims else result
                    accumulated = result if accumulated is None else accumulated + result
                predictions = accumulated / 4
            else:
                predictions = model(images, target, source, positions)
        predictions = predictions.float().cpu().numpy()[:, 0]
        for index, prediction in zip(indices, predictions, strict=True):
            output[:, :, index] = uncrop(np.clip(prediction, -1, 1) * 0.5 + 0.5)
    return output


def predict_to_slab_image(
    model: torch.nn.Module,
    source_path: Path,
    modality: str,
    source: str,
    target: str,
    device: torch.device,
    batch_size: int,
    tta: bool = False,
    auxiliary_paths: list[Path] | None = None,
) -> nib.Nifti1Image:
    """Run the model on one volume and return the clipped submission slab.

    The volume is predicted in canonical orientation, then mapped back to the
    file's own orientation before clipping, exactly as the inference script
    does. The z axis is unaffected by that transform for this dataset, so the
    Z_CLIP_RANGE indices stay aligned with the ground-truth pack.
    """
    original = nib.load(str(source_path))
    original_orientation = nib.io_orientation(original.affine)
    volume, canonical_affine = load_nifti(source_path)

    source_domain = get_joint_domain(modality, source)
    target_domain = get_joint_domain(modality, target)
    # Loaded through the same load_nifti as the primary, so every channel is in the
    # same canonical orientation before any of them reaches the model.
    auxiliary = ([load_nifti(path)[0] for path in auxiliary_paths]
                 if auxiliary_paths else None)
    prediction = predict_slab(
        model, volume, source_domain, target_domain, device, batch_size, tta=tta,
        auxiliary=auxiliary,
    )

    # Zero the background using the source volume, matching the inference script.
    # The threshold is 1e-3, not 1e-6: P_T2W_1.5T_0006 and _0009 carry a flat
    # non-zero pedestal in air (1.2e-4 and 1.1e-5, uniform), so at 1e-6 their
    # foreground mask covers the whole volume and nothing is zeroed -- those two
    # sources would ship predicted texture where the target is flat air. At 1e-3
    # every volume masks consistently (a healthy one moves 0.228 -> 0.226, i.e.
    # 0.2% of voxels at the brain edge, where both sides are already near zero).
    # Rejected an adaptive per-volume pedestal estimate: more precise, but it
    # makes the mask depend on statistics we cannot check against the evaluator.
    start, stop = Z_CLIP_RANGE
    prediction[:, :, start:stop] *= volume[:, :, start:stop] > BACKGROUND_THRESHOLD

    canonical_orientation = nib.io_orientation(canonical_affine)
    transform = nib.orientations.ornt_transform(canonical_orientation, original_orientation)
    prediction = nib.orientations.apply_orientation(prediction, transform)

    image = nib.Nifti1Image(prediction.astype(np.float32), original.affine, original.header)
    image.set_data_dtype(np.float32)
    return image.slicer[:, :, start:stop]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", default=None, help="subdirectory under SUBMISSION_DIR")
    parser.add_argument("--data-dir", type=Path, default=None, help="defaults to DATA_DIR from .env")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="defaults to SUBMISSION_DIR/<name> from .env")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tta", action="store_true",
                        help="average the four in-plane flips")
    parser.add_argument("--dry-run", action="store_true", help="list the work without writing")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    from mrixfields.env import get_data_dir, get_submission_dir

    name = args.name or args.checkpoint.stem
    data_dir = args.data_dir or Path(get_data_dir())
    out_dir = args.out_dir or (Path(get_submission_dir()) / name)
    split = find_split(data_dir)

    pairs = [(source, target) for source in FIELDS for target in FIELDS if source != target]
    total = len(MODALITIES) * len(pairs) * 3
    print(f"checkpoint  : {args.checkpoint}", flush=True)
    print(f"source      : {split}", flush=True)
    print(f"output      : {out_dir}", flush=True)
    print(
        f"{len(pairs)} field pairs x {len(MODALITIES)} modalities x 3 subjects = {total} files",
        flush=True,
    )

    if args.dry_run:
        for modality in MODALITIES[:1]:
            for source, target in pairs[:2]:
                for subject in SOURCE_IDS[source]:
                    print(
                        f"  {split.name}/{modality}/{source}/P_{modality}_{source}_{subject}.nii.gz"
                        f"  ->  task3/{modality}/{source}_to_{target}/pred/"
                        f"P_{modality}_{target}_{subject}.nii.gz",
                        flush=True,
                    )
        print("... (nothing written)", flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {device}")
    model = build_model(args.checkpoint, device)
    # The model itself decides whether the other contrasts are needed; nothing here is
    # a flag the caller could get wrong.
    first_conv = next((p for n, p in model.named_parameters()
                       if n == "encoders.0.0.weight"), None)
    multicontrast = first_conv is not None and first_conv.shape[1] > 1
    if multicontrast:
        print(f"multi-contrast: channel 0 duplicates the predicted contrast, then "
              f"{', '.join(MODALITIES)} at the source field", flush=True)

    task_dir = out_dir / "task3"
    if task_dir.exists():
        shutil.rmtree(task_dir)

    written: list[Path] = []
    for modality in MODALITIES:
        for source, target in pairs:
            for subject in SOURCE_IDS[source]:
                origin = split / modality / source / f"P_{modality}_{source}_{subject}.nii.gz"
                if not origin.is_file():
                    raise FileNotFoundError(origin)
                destination = (
                    task_dir / modality / f"{source}_to_{target}" / "pred"
                    / f"P_{modality}_{target}_{subject}.nii.gz"
                )
                auxiliary_paths = None
                if multicontrast:
                    auxiliary_paths = [
                        split / other / source / f"P_{other}_{source}_{subject}.nii.gz"
                        for other in MODALITIES
                    ]
                    for path in auxiliary_paths:
                        # Every validation subject has all three contrasts at its one
                        # source field; stop rather than zero-fill if that ever changes,
                        # because a zero channel is a legitimate intensity here.
                        if not path.is_file():
                            raise FileNotFoundError(path)
                clipped = predict_to_slab_image(
                    model, origin, modality, source, target, device, args.batch_size,
                    tta=args.tta, auxiliary_paths=auxiliary_paths,
                )
                if clipped.shape != EXPECTED_SHAPE:
                    raise RuntimeError(
                        f"{destination} has shape {clipped.shape}, expected {EXPECTED_SHAPE}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                nib.save(clipped, str(destination))
                written.append(destination)
        print(f"{modality}: {len(written)} files so far", flush=True)

    if len(written) != total:
        raise RuntimeError(f"wrote {len(written)} files, expected {total}")

    zip_path = out_dir / "task3.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(written):
            archive.write(path, path.relative_to(task_dir.parent))
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"corrupt archive member: {bad}")
    if len(members) != total:
        raise RuntimeError(f"archive has {len(members)} members, expected {total}")

    print(f"wrote {len(written)} files", flush=True)
    print(f"archive: {zip_path} ({zip_path.stat().st_size / 2**20:.0f} MiB, {len(members)} members)",
          flush=True)


if __name__ == "__main__":
    main()
