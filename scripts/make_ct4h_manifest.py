"""One manifest for the registered CT4H volumes, built from the registration logs.

run_authors_registration.py writes one log per shard and appends on every rerun, so the logs hold
duplicate rows (A1 was rerun with --fallback_rule never; 15 F1 scans were rerun with --force_roll F1=4).
This keeps the newest row per series, only if its NIfTI exists, and writes <reg_dir>/registration_manifest.csv.
That file is what the dataset code should read, instead of globbing the folder.

    python scripts/make_ct4h_manifest.py --reg_dir <authors_registered>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY = ["scanner", "dose_mgy", "recon", "repeat"]
KEEP = ["scanner", "dose_mgy", "recon", "repeat", "out_file", "series_uid", "flipped", "roll_slices", "roll_forced",
        "ssim_raw", "ssim_after", "fallback", "seconds", "note"]
NOTE_E2 = "E2: on some scans y_high is ~1 mm off in z vs A1 (matched pairs <= ~0.75 mm); accepted as a known limitation"
NOTE_FORCED = "roll forced (authors' roll search flipped between 4 and 8 on noisy scans)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reg_dir", required=True)
    parser.add_argument("--out", default=None, help="default: <reg_dir>/registration_manifest.csv")
    args = parser.parse_args()
    reg = Path(args.reg_dir)

    logs = sorted(reg.glob("registration_log_*.csv"), key=lambda p: p.stat().st_mtime)  # newest log last
    if not logs:
        raise SystemExit(f"no registration_log_*.csv in {reg}")
    frames = [pd.read_csv(path) for path in logs]
    for frame in frames:
        if "roll_forced" not in frame:
            frame["roll_forced"] = 0
    rows = pd.concat(frames, ignore_index=True)
    rows["error"] = rows["error"].fillna("")

    failed = rows[rows["error"] != ""].drop_duplicates(KEY, keep="last")
    ok = rows[rows["error"] == ""]
    n_all = len(ok)
    ok = ok.drop_duplicates(KEY, keep="last")  # later rows (reruns) replace earlier ones
    print(f"{len(logs)} logs, {n_all} successful rows, {n_all - len(ok)} superseded by a rerun")

    exists = ok["out_file"].map(lambda name: (reg / name).is_file())
    if (~exists).any():
        print("in the logs but no file on disk (dropped):", ok.loc[~exists, "out_file"].tolist())
    ok = ok[exists].copy()
    done = set(map(tuple, ok[KEY].values))
    failed = failed[[tuple(r) not in done for r in failed[KEY].values]]

    ok["roll_forced"] = ok["roll_forced"].fillna(0).astype(int)
    ok["note"] = ""
    ok.loc[ok["scanner"] == "E2", "note"] = NOTE_E2
    ok.loc[ok["roll_forced"] == 1, "note"] = NOTE_FORCED
    ok = ok.sort_values(KEY)[KEEP]
    out = Path(args.out) if args.out else reg / "registration_manifest.csv"
    ok.to_csv(out, index=False)

    print(f"wrote {out}: {len(ok)} volumes")
    print(ok.groupby("scanner").agg(n=("out_file", "size"), fallback=("fallback", "sum"),
                                    roll=("roll_slices", lambda s: sorted(set(s.astype(int)))),
                                    forced=("roll_forced", "sum")).to_string())
    if len(failed):
        print("no registered volume (error in the log):")
        print(failed[KEY + ["error"]].to_string(index=False))


if __name__ == "__main__":
    main()
