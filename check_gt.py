#!/usr/bin/env python3
"""
Check MOT17 ground-truth around a GLRA case: does any GT box at the recover
frame sit near the dumped GPR/det position? This tells us whether the dumped
coords are in the original image pixel frame or shifted by GMC.

Usage:
  python check_gt.py \
    --gt /home/caig/data/MOT17/train/MOT17-11-FRCNN/gt/gt.txt \
    --case glra_cases/MOT17-11-FRCNN_t350_f221.json
"""

import json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True)
ap.add_argument("--case", required=True)
a = ap.parse_args()

rec = json.load(open(a.case))
f = rec["frame"]
d = rec["det"]
g = rec["gpr_pred"]
tgt_cx, tgt_cy = d["cx"], d["cy"]

# MOT gt.txt: frame,id,bb_left,bb_top,bb_w,bb_h,conf,class,vis
rows = []
for line in open(a.gt):
    p = line.strip().split(",")
    if len(p) < 6:
        continue
    fr = int(p[0])
    if fr != f:
        continue
    x, y, w, h = map(float, p[2:6])
    cx, cy = x + w / 2, y + h / 2
    rows.append((int(p[1]), cx, cy, w, h))

print(
    f"recover frame f{f}: det=({tgt_cx:.0f},{tgt_cy:.0f}) GPR=({g['cx']:.0f},{g['cy']:.0f})"
)
print(f"GT boxes at f{f}: {len(rows)} total")
print("nearest GT boxes to the dumped det center:")
rows.sort(key=lambda r: (r[1] - tgt_cx) ** 2 + (r[2] - tgt_cy) ** 2)
for gid, cx, cy, w, h in rows[:5]:
    dist = ((cx - tgt_cx) ** 2 + (cy - tgt_cy) ** 2) ** 0.5
    print(
        f"  GT id {gid:3d}  center=({cx:.0f},{cy:.0f})  size=({w:.0f}x{h:.0f})  dist={dist:.0f}px"
    )
