#!/usr/bin/env python3
"""Weight-average two fine-tunes that started from the same seed (a "model soup").

    ~/anaconda3/envs/mri/bin/python scripts/make_soup.py \
        --member runs/task3_mc_ssim_slice/artifacts/avg_e6_e8.pt:0.5 \
        --member runs/task3_mc_ssim_fine/artifacts/task3_unet_finetune_8.pt:0.5 \
        --out runs/task3_mc_ssim_slice/artifacts/soup_50_50.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Only the tensors that are a function of the conditioning path, not of the image path:
# these are the ones the control implicitly holds at zero.
ZEROED_IN_CONTROL = ("slice_embedding.projection.2.weight", "slice_embedding.projection.2.bias")


def weights_of(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        raise SystemExit(f"{path} is a bare tensor, not a checkpoint")
    state = payload.get("model_ema") or payload.get("model") or payload
    return {k: v for k, v in state.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--member", action="append", required=True,
                        help="path:weight, repeatable. Weights are normalised to sum to 1.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    members = []
    for entry in args.member:
        path, _, weight = entry.rpartition(":")
        members.append((Path(path), float(weight)))
    total = sum(w for _, w in members)
    members = [(p, w / total) for p, w in members]

    states = [(weights_of(p), w, p) for p, w in members]
    shared = set.intersection(*(set(s) for s, _, _ in states))
    union = set().union(*(set(s) for s, _, _ in states))
    extra = union - shared
    unexpected = extra - set().union(*(
        {k for k in s if k.startswith("slice_embedding.")} for s, _, _ in states))
    if unexpected:
        raise SystemExit(f"members disagree on keys beyond slice conditioning: {sorted(unexpected)}")

    for key in sorted(shared):
        shapes = {tuple(s[key].shape) for s, _, _ in states}
        if len(shapes) > 1:
            raise SystemExit(f"{key}: incompatible shapes {shapes}")

    souped: dict[str, torch.Tensor] = {}
    for key in sorted(shared):
        reference = states[0][0][key]
        if not reference.is_floating_point():
            # Integer buffers (num_batches_tracked and friends) are counters, not
            # parameters; a fractional mean of two counters is meaningless.
            values = {int(s[key].reshape(-1)[0]) for s, _, _ in states} if reference.numel() else {0}
            if len(values) > 1:
                raise SystemExit(f"{key}: integer buffers differ {values}, refusing to average")
            souped[key] = reference.clone()
            continue
        acc = torch.zeros_like(reference, dtype=torch.float64)
        for state, weight, _ in states:
            acc += state[key].to(torch.float64) * weight
        souped[key] = acc.to(reference.dtype)

    for key in sorted(extra):
        owners = [(s, w, p) for s, w, p in states if key in s]
        if len(owners) != 1:
            raise SystemExit(f"{key}: expected exactly one owner, got {len(owners)}")
        state, weight, path = owners[0]
        scale = weight if key in ZEROED_IN_CONTROL else 1.0
        souped[key] = (state[key].to(torch.float64) * scale).to(state[key].dtype)
        print(f"  {key}: from {path.name} x{scale:g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": souped}, args.out)
    print(f"{len(souped)} tensors ({len(shared)} averaged, {len(extra)} carried) -> {args.out}")
    for path, weight in members:
        print(f"  {weight:.3f}  {path}")


if __name__ == "__main__":
    main()
