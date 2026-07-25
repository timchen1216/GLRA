#!/usr/bin/env python3
"""
rank_cases.py — Rank GLRA recovery cases by how well they demonstrate the
method's advantage, for picking defense-slide qualitative examples.

Score rewards the thesis claim: GPR gives a more reliable position at
occlusion onset than the KF constant-velocity extrapolation (noise robustness,
near-zero true displacement) — NOT long-range extrapolation.

Reads the glra_cases/*.json produced by the GLRA pipeline. Each case has:
  seq, frame, track_id, lost_duration, end_frame,
  obs=[[f,cx,cy,w,h]...], gpr_pred{cx,cy,w,h,sigma_px}, kf_pred{cx,cy,w,h},
  det{cx,cy,w,h,score}, cost

Usage
-----
python rank_cases.py --cases glra_cases --top 20
python rank_cases.py --cases glra_cases --top 5 --csv ranked.csv
"""

import argparse
import glob
import json
import math
import os


def dist(a, b, ax="cx", ay="cy"):
    return math.hypot(a[ax] - b[ax], a[ay] - b[ay])


def analyse(d):
    gpr, kf, det = d["gpr_pred"], d["kf_pred"], d["det"]
    gpr_dist = dist(gpr, det)  # GPR prediction -> matched detection
    kf_dist = dist(kf, det)  # KF prediction  -> matched detection
    margin = kf_dist - gpr_dist  # how much GPR beats KF (px)

    # true displacement over the lost span (last obs -> detection at recover)
    obs = d["obs"]
    last = obs[-1]  # [f, cx, cy, w, h]
    disp = math.hypot(det["cx"] - last[1], det["cy"] - last[2])
    diag = math.hypot(det["w"], det["h"]) or 1.0
    disp_norm = disp / diag  # normalised by box diagonal

    return {
        "seq": d["seq"],
        "track_id": d["track_id"],
        "frame": d["frame"],
        "end_frame": d.get("end_frame"),
        "lost": d["lost_duration"],
        "gpr_dist": gpr_dist,
        "kf_dist": kf_dist,
        "margin": margin,
        "disp_norm": disp_norm,
        "det_score": det.get("score", 0.0),
        "sigma": gpr.get("sigma_px", float("nan")),
        "n_obs": len(obs),
    }


def score(a, lost_lo, lost_hi):
    """Higher = better defense example. Tunable, but grounded in the thesis."""
    # 1) GPR must beat KF by a visible margin (the whole point)
    s_margin = a["margin"]  # px, can be negative
    # 2) GPR should actually be close to the detection (clean recovery)
    s_gpr_close = max(0.0, 20.0 - a["gpr_dist"])  # reward gpr_dist < 20px
    # 3) lost_duration in a sweet spot: long enough to be a real gap,
    #    not so long it looks like implausible extrapolation
    ld = a["lost"]
    if ld < lost_lo:
        s_lost = -8.0 * (lost_lo - ld)  # penalise too-short (threshold jitter)
    elif ld > lost_hi:
        s_lost = -1.5 * (ld - lost_hi)  # mild penalty for very long
    else:
        s_lost = 6.0
    # 4) near-zero true displacement supports "noise robustness, not extrapolation"
    s_disp = max(0.0, 6.0 - 30.0 * a["disp_norm"])
    # 5) longer observed history -> GPR better conditioned, nicer trajectory plot
    s_obs = min(4.0, a["n_obs"] / 15.0)

    total = 2.0 * s_margin + s_gpr_close + s_lost + s_disp + s_obs
    return total


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cases", required=True, help="folder of glra_cases/*.json")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seq", default=None, help="filter to one sequence")
    ap.add_argument(
        "--lost-lo", type=int, default=6, help="min 'good' lost_duration (default 6)"
    )
    ap.add_argument(
        "--lost-hi", type=int, default=20, help="max 'good' lost_duration (default 20)"
    )
    ap.add_argument("--csv", default=None, help="write full ranking to CSV")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.cases, "*.json"))
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if args.seq and d.get("seq") != args.seq:
            continue
        a = analyse(d)
        a["score"] = score(a, args.lost_lo, args.lost_hi)
        a["file"] = os.path.basename(f)
        rows.append(a)

    rows.sort(key=lambda r: r["score"], reverse=True)

    hdr = (
        f"{'#':>3} {'score':>7} {'seq':<16} {'tid':>5} {'recF':>5} "
        f"{'lost':>4} {'gpr_d':>6} {'kf_d':>6} {'margin':>6} "
        f"{'dispN':>6} {'σpx':>5} {'nobs':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows[: args.top]):
        print(
            f"{i:>3} {r['score']:>7.1f} {r['seq']:<16} {r['track_id']:>5} "
            f"{r['frame']:>5} {r['lost']:>4} {r['gpr_dist']:>6.1f} "
            f"{r['kf_dist']:>6.1f} {r['margin']:>6.1f} {r['disp_norm']:>6.3f} "
            f"{r['sigma']:>5.1f} {r['n_obs']:>4}"
        )

    if args.csv:
        import csv

        keys = [
            "score",
            "seq",
            "track_id",
            "frame",
            "end_frame",
            "lost",
            "gpr_dist",
            "kf_dist",
            "margin",
            "disp_norm",
            "det_score",
            "sigma",
            "n_obs",
            "file",
        ]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nFull ranking ({len(rows)} cases) -> {args.csv}")

    # quick guidance
    print(
        "\nColumns: gpr_d/kf_d = GPR/KF prediction distance to matched det (px); "
        "margin = kf_d - gpr_d (bigger = GPR wins more); "
        "dispN = true displacement / box diagonal (small = noise-robust case)."
    )


if __name__ == "__main__":
    main()
