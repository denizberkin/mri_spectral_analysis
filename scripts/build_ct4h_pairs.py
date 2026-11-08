"""Build the low-dose -> high-dose training pairs of CT4H from the index and the alignment checks.

A pair is two scans of the same phantom on the same scanner, with the same reconstruction. Each
scan's position relative to its scanner's reference scan comes from the checks:

    check_ct4h_alignment.py  -> in-plane shift (dy, dx) in pixels, correlation, header z offset
    check_ct4h_z.py          -> residual z displacement dz_mm  (optional, but needed for E2)

Both are measured against the same reference scan, so the shift between two scans is the
difference of their shifts. The checks showed that FBP, IR and DL of one scan share a grid, so
each scan's numbers are taken from its FBP series.

Per pair this writes (shifts are what to apply to the SOURCE to land on the TARGET):

    dy_px, dx_px       in-plane shift in pixels;  dy_int, dx_int  the same, rounded (no interpolation)
    dz_mm              sample the source at (target z + dz_mm)
    needs_shift        |shift| > --shift_tol_px   (otherwise the pair can be used as is in-plane)
    needs_z            |dz| > --z_tol_mm          (only if a z table was given)
    excluded_reason    a scan that sits somewhere else entirely (outlier), or has not been checked
    usable             not excluded, not unchecked

--pairing matched pairs repeat k with repeat k; all pairs every source repeat with every target repeat.

Usage:
    python scripts/build_ct4h_pairs.py --index ct4h_index.csv --alignment ct4h_alignment.csv --z ct4h_z.csv
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct4h.data.utils import DOSES, RECONS, SCANNERS, read_index, select

COLUMNS = [
    "scanner", "recon", "src_dose", "tgt_dose", "src_repeat", "tgt_repeat", "src_series_uid", "tgt_series_uid",
    "src_dir", "tgt_dir", "dy_px", "dx_px", "dy_int", "dx_int", "shift_px", "dz_mm",
    "needs_shift", "needs_z", "excluded_reason", "usable",
]


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_alignment(path: Path) -> dict:
    """(scanner, dose, repeat) -> in-plane numbers of that scan against the scanner reference (FBP row preferred)."""
    best: dict = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("vs") != "ref" or row.get("error"):
                continue
            key = (row["scanner"], int(row["dose_mgy"]), int(row["repeat"]))
            if key not in best or row["recon"] == "FBP":
                best[key] = {
                    "dy": _float(row["shift_y_px"]), "dx": _float(row["shift_x_px"]), "ncc": _float(row["ncc"]),
                    "shift_mm": _float(row["shift_mm"]), "hdr_z_off": _float(row["hdr_z_off"]),
                }
    return best


def load_z(path: Path) -> tuple[dict, set]:
    """(scanner, dose, repeat) -> dz_mm, and the set of reference scans."""
    dz, refs = {}, set()
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scanner"], int(row["dose_mgy"]), int(row["repeat"]))
            if row.get("error") == "reference":
                refs.add(key)
                dz[key] = {"dz": 0.0, "reliable": True}
            elif row.get("dz_mm") not in (None, ""):
                dz[key] = {"dz": _float(row["dz_mm"]), "reliable": not row.get("error")}
    return dz, refs


def scan_state(key, align, zinfo, refs, args) -> dict:
    """Position of one scan (relative to its scanner reference) and why it may be unusable."""
    state = {"dy": 0.0, "dx": 0.0, "dz": float("nan"), "reason": ""}
    is_ref = key in refs
    if is_ref:
        state["dz"] = 0.0
        return state
    info = align.get(key)
    if info is None:
        state["reason"] = "unchecked_inplane"
    else:
        state.update(dy=info["dy"], dx=info["dx"])
        if info["ncc"] < args.min_ncc or info["shift_mm"] > args.max_shift_mm or abs(info["hdr_z_off"]) > args.max_hdr_z_slices:
            state["reason"] = "outlier_position"
    if zinfo is not None:
        z = zinfo.get(key)
        if z is None or not z["reliable"] or not np.isfinite(z["dz"]):
            state["reason"] = state["reason"] or "unchecked_z"
        else:
            state["dz"] = z["dz"]
    return state


def build(args) -> list[dict]:
    index = select(read_index(args.index), scanners=args.scanners, recons=args.recons)
    align = load_alignment(Path(args.alignment))
    zinfo, z_refs = load_z(Path(args.z)) if args.z else (None, set())

    # the alignment check's reference: highest dose, lowest repeat (the z check marks it explicitly)
    refs = set(z_refs)
    if not refs:
        for scanner in {r["scanner"] for r in index}:
            doses = [r["dose_mgy"] for r in index if r["scanner"] == scanner and r["recon"] == "FBP"]
            reps = [r["repeat"] for r in index if r["scanner"] == scanner and r["recon"] == "FBP" and r["dose_mgy"] == max(doses)]
            refs.add((scanner, max(doses), min(reps)))

    by_scan = defaultdict(dict)  # (scanner, recon, dose) -> repeat -> index row
    for row in index:
        by_scan[(row["scanner"], row["recon"], row["dose_mgy"])][row["repeat"]] = row

    pairs = []
    for scanner in args.scanners:
        for recon in args.recons:
            target_rows = by_scan.get((scanner, recon, args.target_dose), {})
            for dose in sorted(d for d in {k[2] for k in by_scan if k[0] == scanner and k[1] == recon}
                               if d < args.target_dose and (args.source_doses is None or d in args.source_doses)):
                source_rows = by_scan[(scanner, recon, dose)]
                if args.pairing == "matched":
                    combos = [(r, r) for r in sorted(source_rows) if r in target_rows]
                else:
                    combos = [(rs, rt) for rs in sorted(source_rows) for rt in sorted(target_rows)]
                for rs, rt in combos:
                    s = scan_state((scanner, dose, rs), align, zinfo, refs, args)
                    t = scan_state((scanner, args.target_dose, rt), align, zinfo, refs, args)
                    dy, dx = s["dy"] - t["dy"], s["dx"] - t["dx"]
                    dz = s["dz"] - t["dz"]
                    shift = float(np.hypot(dy, dx))
                    reason = s["reason"] or t["reason"]
                    pairs.append({
                        "scanner": scanner, "recon": recon, "src_dose": dose, "tgt_dose": args.target_dose,
                        "src_repeat": rs, "tgt_repeat": rt,
                        "src_series_uid": source_rows[rs]["series_uid"], "tgt_series_uid": target_rows[rt]["series_uid"],
                        "src_dir": source_rows[rs]["series_dir"], "tgt_dir": target_rows[rt]["series_dir"],
                        "dy_px": round(dy, 3), "dx_px": round(dx, 3), "dy_int": int(round(dy)), "dx_int": int(round(dx)),
                        "shift_px": round(shift, 3), "dz_mm": "" if not np.isfinite(dz) else round(dz, 3),
                        "needs_shift": int(shift > args.shift_tol_px),
                        "needs_z": int(np.isfinite(dz) and abs(dz) > args.z_tol_mm),
                        "excluded_reason": reason, "usable": int(not reason),
                    })
    return pairs


def summarize(pairs: list[dict]) -> None:
    print(f"\n{len(pairs)} pairs, {sum(p['usable'] for p in pairs)} usable")
    print(f"{'scanner':8}{'recon':6}{'pairs':>7}{'usable':>8}{'shift':>7}{'z':>5}{'outlier':>9}{'unchecked':>11}")
    keys = sorted({(p["scanner"], p["recon"]) for p in pairs})
    for scanner, recon in keys:
        sel = [p for p in pairs if p["scanner"] == scanner and p["recon"] == recon]
        why = Counter(p["excluded_reason"] for p in sel)
        usable = [p for p in sel if p["usable"]]
        print(f"{scanner:8}{recon:6}{len(sel):>7}{len(usable):>8}{sum(p['needs_shift'] for p in usable):>7}"
              f"{sum(p['needs_z'] for p in usable):>5}{why['outlier_position']:>9}"
              f"{why['unchecked_inplane'] + why['unchecked_z']:>11}")
    print("\nshift = usable pairs needing an in-plane shift; z = usable pairs needing a z correction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True)
    parser.add_argument("--alignment", required=True, help="output of check_ct4h_alignment.py")
    parser.add_argument("--z", default=None, help="output of check_ct4h_z.py (optional)")
    parser.add_argument("--out", default="ct4h_pairs.csv")
    parser.add_argument("--scanners", nargs="+", choices=SCANNERS, default=SCANNERS)
    parser.add_argument("--recons", nargs="+", choices=RECONS, default=["FBP"])
    parser.add_argument("--target_dose", type=int, choices=DOSES, default=14)
    parser.add_argument("--source_doses", nargs="+", type=int, choices=DOSES, default=None, help="default: every dose below the target.")
    parser.add_argument("--pairing", choices=["matched", "all"], default="matched")
    parser.add_argument("--shift_tol_px", type=float, default=0.5)
    parser.add_argument("--z_tol_mm", type=float, default=0.5)
    parser.add_argument("--min_ncc", type=float, default=0.9, help="below this a scan counts as an outlier.")
    parser.add_argument("--max_shift_mm", type=float, default=10.0, help="in-plane shift above this = outlier.")
    parser.add_argument("--max_hdr_z_slices", type=float, default=50.0, help="header z offset above this = outlier.")
    args = parser.parse_args()

    pairs = build(args)
    if not pairs:
        sys.exit("No pairs: check --scanners/--recons/--target_dose.")
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(pairs)
    print(f"wrote {args.out}")
    summarize(pairs)


if __name__ == "__main__":
    main()
