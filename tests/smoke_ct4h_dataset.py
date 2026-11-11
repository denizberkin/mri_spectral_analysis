"""Smoke test of CT4HPairDataset on small fake volumes (no real data needed):
    python tests/smoke_ct4h_dataset.py
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct4h.data.cached_dataset import CT4HPairDataset, neighbour_offsets  # noqa: E402
from ct4h.data.utils import DOSES, SCANNER_SPLIT, VENDORS  # noqa: E402

N_Z, H, W = 12, 40, 50


def make_cache(root: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for scanner in VENDORS:
        for dose in DOSES:
            for repeat in range(1, 11):
                if (scanner, dose, repeat) == ("B2", 14, 5):
                    continue  # the known registration failure
                rel = f"{scanner}/{scanner}_{dose}mGy_FBP_rep{repeat:02d}.npy"
                (root / scanner).mkdir(exist_ok=True)
                # value encodes the slice index so neighbour order and slice choice can be checked
                vol = np.broadcast_to(np.arange(N_Z)[:, None, None] * 10 - 500, (N_Z, H, W)).astype(np.int16).copy()
                vol += rng.integers(-3, 4, vol.shape, dtype=np.int16)
                np.save(root / rel, vol)
                rows.append(dict(scanner=scanner, vendor=VENDORS[scanner], dose_mgy=dose, recon="FBP", repeat=repeat,
                                 path=rel, n_slices=N_Z, height=H, width=W))
    with open(root / "cache_index.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    assert neighbour_offsets(1) == [0, -1, 1] and neighbour_offsets(2) == [0, -1, 1, -2, 2]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_cache(root)

        # scanner split: no scanner of one split appears in another
        seen = {}
        for split in ("train", "val", "test"):
            ds = CT4HPairDataset(root, split, "scanner", length=200, deterministic=True, pad_multiple=16)
            seen[split] = {ds.draw(i)["scanner"] for i in range(200)}
            assert seen[split] <= set(SCANNER_SPLIT[split]), (split, seen[split])
        assert not (seen["train"] & seen["val"] or seen["train"] & seen["test"] or seen["val"] & seen["test"])

        # repeat split: every scanner, disjoint repeats
        for split, repeats in (("train", set(range(1, 8))), ("val", {8}), ("test", {9, 10})):
            ds = CT4HPairDataset(root, split, "repeat", length=300, deterministic=True)
            draws = [ds.draw(i) for i in range(300)]
            assert {d["source_repeat"] for d in draws} <= repeats and {d["target_repeat"] for d in draws} <= repeats
            assert len({d["scanner"] for d in draws}) > 8

        # item content
        ds = CT4HPairDataset(root, "train", "scanner", directions="up", neighbours=1, length=50, deterministic=True)
        for i in range(50):
            item = ds[i]
            d = ds.draw(i)
            assert d["target_dose"] > d["source_dose"]
            assert item["source"].shape == (3, 48, 64) and item["target"].shape == (1, 48, 64), item["source"].shape
            assert (item["orig_height"], item["orig_width"]) == (H, W)
            z = item["slice_idx"]
            centre = item["source"][0, :H, :W].mean().item()
            expected = ((z * 10 - 500) + 1000) / 3000 * 2 - 1
            assert abs(centre - expected) < 0.01, (z, centre, expected)
            lower = item["source"][1, :H, :W].mean().item()
            assert abs(lower - (((max(z - 1, 0) * 10 - 500) + 1000) / 3000 * 2 - 1)) < 0.01  # channel 1 is z-1
            assert float(item["source"][0, H:, :].min()) == -1.0  # padding is air
            assert item["source_domain"] == DOSES.index(item["source_dose"])

        # determinism, and variety when not deterministic
        a, b = ds[7], ds[7]
        assert np.array_equal(a["source"].numpy(), b["source"].numpy())

        # non-deterministic draws must keep changing for the same index (persistent workers re-use one seed)
        free = CT4HPairDataset(root, "train", "scanner", length=50, deterministic=False)
        first_epoch = [free.draw(i) for i in range(50)]
        second_epoch = [free.draw(i) for i in range(50)]
        assert first_epoch != second_epoch, "training draws repeat between epochs"
        assert len({tuple(sorted(d.items())) for d in first_epoch + second_epoch}) > 90

        # target_mode "mean" averages the target-dose repeats (B2 14 mGy has only 9: rep 5 failed to register)
        mean_ds = CT4HPairDataset(root, "train", "scanner", target_mode="mean", length=20, deterministic=True)
        assert mean_ds[0]["target"].shape == (1, 48, 64)

        # directions
        assert all(CT4HPairDataset(root, "test", "scanner", directions="down", length=30, deterministic=True)
                   .draw(i)["target_dose"] < CT4HPairDataset(root, "test", "scanner", directions="down",
                   length=30, deterministic=True).draw(i)["source_dose"] for i in range(30))
    print("smoke test passed")


if __name__ == "__main__":
    main()
