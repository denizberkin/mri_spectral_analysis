"""Save a picture of one low/full slice pair: images, difference, in a wide and a soft-tissue window.

Usage:
    python scripts/preview_ldct_pair.py --patient C005
    python scripts/preview_ldct_pair.py --patient L012 --slice 80 --output /tmp/l012.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldct.data.utils import anatomy_of, slice_name
from ldct.env import get_ldct_preprocessed_dir

WINDOWS = {"wide (-1000, 400)": (-1000, 400), "soft tissue (-160, 240)": (-160, 240)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient", required=True, help="e.g. C005 or L012")
    parser.add_argument("--slice", type=int, default=None, help="Slice index. Default: the middle slice.")
    parser.add_argument("--preprocessed_dir", default=None, help="Default: LDCT_PREPROCESSED_DIR.")
    parser.add_argument("--output", default=None, help="PNG path. Default: {preprocessed_dir}/preview/{patient}_sNNN.png.")
    args = parser.parse_args()

    out = Path(args.preprocessed_dir or get_ldct_preprocessed_dir() or "")
    base = out / anatomy_of(args.patient)
    n = len(list((base / "full").glob(f"{args.patient}_s*.npz")))
    if n == 0:
        sys.exit(f"No slices for {args.patient} in {base}")
    idx = n // 2 if args.slice is None else args.slice
    with np.load(base / "low" / f"{slice_name(args.patient, idx)}.npz") as npz:
        low = npz["image"].astype(np.float32)
    with np.load(base / "full" / f"{slice_name(args.patient, idx)}.npz") as npz:
        full = npz["image"].astype(np.float32)

    fig, axes = plt.subplots(len(WINDOWS), 3, figsize=(13, 4.4 * len(WINDOWS)))
    for row, (name, (lo, hi)) in zip(axes, WINDOWS.items()):
        for ax, image, title in ((row[0], low, "low dose"), (row[1], full, "full dose")):
            ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(f"{title}, {name}")
        row[2].imshow(low - full, cmap="coolwarm", vmin=-(hi - lo) / 2, vmax=(hi - lo) / 2)
        row[2].set_title(f"low - full (+/-{(hi - lo) / 2:.0f} HU)")
        for ax in row:
            ax.axis("off")
    fig.suptitle(f"{args.patient} slice {idx}/{n - 1}")
    fig.tight_layout()
    output = Path(args.output) if args.output else out / "preview" / f"{slice_name(args.patient, idx)}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=90)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
