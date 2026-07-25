#!/usr/bin/env python3
"""
pick_and_render.py — One-shot: rank GLRA recovery cases, then render the top-N
as zoom comparison figures/videos, for defense-slide selection.

Chains rank_cases.py (scoring) with glra_compare.py (rendering). Run on the
machine that has the tracking txt + img folders.

Directory assumptions (edit --sparse-dir / --ours-dir / --data-root to match):
  <sparse_dir>/<SEQ>.txt          e.g. .../track_results_sparsetrack/MOT20-05.txt
  <ours_dir>/<SEQ>.txt            e.g. .../track_results_glra/MOT20-05.txt
  <data_root>/<train_or_test>/<SEQ>/gt/gt.txt
  <data_root>/<train_or_test>/<SEQ>/img1/

For MOT17 the seq in JSON is e.g. 'MOT17-04-FRCNN' — used verbatim as <SEQ>.

Example
-------
python pick_and_render.py \
    --cases   glra_cases \
    --sparse-dir /home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results_sparsetrack \
    --ours-dir   /home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results_glra \
    --data-root  /home/caig/data/MOT20 \
    --val-json   /home/caig/data/MOT20/annotations/val_half.json \
    --out out/best --top 5
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- scoring (kept in sync with rank_cases.py) ---------------------------- #
def dist(a, b):
    return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])


def analyse(d):
    gpr, kf, det = d["gpr_pred"], d["kf_pred"], d["det"]
    gpr_dist, kf_dist = dist(gpr, det), dist(kf, det)
    last = d["obs"][-1]
    disp = math.hypot(det["cx"] - last[1], det["cy"] - last[2])
    diag = math.hypot(det["w"], det["h"]) or 1.0
    return {
        "seq": d["seq"],
        "track_id": d["track_id"],
        "frame": d["frame"],
        "lost": d["lost_duration"],
        "gpr_dist": gpr_dist,
        "kf_dist": kf_dist,
        "margin": kf_dist - gpr_dist,
        "disp_norm": disp / diag,
        "n_obs": len(d["obs"]),
    }


def score(a, lo, hi):
    s_margin = a["margin"]
    s_gpr_close = max(0.0, 20.0 - a["gpr_dist"])
    ld = a["lost"]
    s_lost = -8.0 * (lo - ld) if ld < lo else -1.5 * (ld - hi) if ld > hi else 6.0
    s_disp = max(0.0, 6.0 - 30.0 * a["disp_norm"])
    s_obs = min(4.0, a["n_obs"] / 15.0)
    return 2.0 * s_margin + s_gpr_close + s_lost + s_disp + s_obs


# ---- path resolution ------------------------------------------------------ #
def find_seq_paths(seq, sparse_dir, ours_dir, data_root):
    sp = os.path.join(sparse_dir, f"{seq}.txt")
    ou = os.path.join(ours_dir, f"{seq}.txt")
    gt = img = None
    for split in ("train", "test", ""):
        base = (
            os.path.join(data_root, split, seq)
            if split
            else os.path.join(data_root, seq)
        )
        g = os.path.join(base, "gt", "gt.txt")
        im = os.path.join(base, "img1")
        if os.path.isfile(g) and os.path.isdir(im):
            gt, img = g, im
            break
    return sp, ou, gt, img


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cases", required=True)
    ap.add_argument("--sparse-dir", required=True)
    ap.add_argument("--ours-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-json", default=None)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seq", default=None, help="restrict to one sequence")
    ap.add_argument("--lost-lo", type=int, default=1)
    ap.add_argument("--lost-hi", type=int, default=20)
    ap.add_argument("--pad", type=int, default=20)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--zoom-w", type=int, default=320)
    ap.add_argument("--zoom-h", type=int, default=480)
    ap.add_argument("--compare-script", default=os.path.join(HERE, "glra_compare.py"))
    args = ap.parse_args()

    # rank
    rows = []
    for f in glob.glob(os.path.join(args.cases, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if args.seq and d.get("seq") != args.seq:
            continue
        a = analyse(d)
        a["score"] = score(a, args.lost_lo, args.lost_hi)
        rows.append(a)
    rows.sort(key=lambda r: r["score"], reverse=True)
    top = rows[: args.top]

    print(f"Top {len(top)} cases:")
    for i, r in enumerate(top):
        print(
            f"  {i}: {r['seq']} t{r['track_id']} f{r['frame']}  "
            f"score={r['score']:.1f} margin={r['margin']:.1f}px "
            f"lost={r['lost']} gpr={r['gpr_dist']:.1f} kf={r['kf_dist']:.1f}"
        )

    os.makedirs(args.out, exist_ok=True)

    # render each: min-gap set to just below this case's lost so it surfaces,
    # rendered into its own subfolder tagged by seq/track/frame.
    for i, r in enumerate(top):
        seq = r["seq"]
        sp, ou, gt, img = find_seq_paths(
            seq, args.sparse_dir, args.ours_dir, args.data_root
        )
        missing = [
            n
            for n, p in [("sparse", sp), ("ours", ou), ("gt", gt), ("img", img)]
            if not p or not os.path.exists(p)
        ]
        if missing:
            print(f"[skip {i}] {seq}: missing {missing}")
            continue
        sub = os.path.join(
            args.out, f"rank{i:02d}_{seq}_t{r['track_id']}_f{r['frame']}"
        )
        cmd = [
            sys.executable,
            args.compare_script,
            "--sparse",
            sp,
            "--ours",
            ou,
            "--gt",
            gt,
            "--img",
            img,
            "--out",
            sub,
            "--min-gap",
            str(max(1, r["lost"] - 1)),
            "--pad",
            str(args.pad),
            "--fps",
            str(args.fps),
            "--zoom-w",
            str(args.zoom_w),
            "--zoom-h",
            str(args.zoom_h),
            "--max-cases",
            "40",
        ]
        if args.val_json:
            cmd += ["--val-json", args.val_json, "--seq", seq]
        print(f"\n[render {i}] {seq} t{r['track_id']} f{r['frame']} -> {sub}")
        subprocess.run(cmd)

    print(f"\nDone. Browse {args.out}/rank*/keyframes/*_zoom.png to pick.")


if __name__ == "__main__":
    main()

"""
python pick_and_render.py \
    --cases glra_cases \
    --sparse-dir /home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results_sparsetrack \
    --ours-dir   /home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results_glra \
    --data-root  /home/caig/data/MOT20 \
    --val-json   /home/caig/data/MOT20/annotations/val_half.json \
    --out out/best --top 5
"""
