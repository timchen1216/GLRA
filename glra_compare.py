#!/usr/bin/env python3
"""
glra_compare.py — Highlight where the proposed method (GLRA) beats SparseTrack.

Per recovered GT identity, colours are assigned PER FRAME (GT used as anchor):

  Right panel (Ours / GLRA), for the target GT id:
    - covered by ours, at/after the recovery frame ...... YELLOW (recovered & held)
    - covered by ours, before recovery ................... grey
    - not covered ........................................ no box
  Left panel (SparseTrack), for the target GT id:
    - covered, matched track id == baseline id ........... grey
    - covered, track id != baseline id .................. RED (id switch)
    - not covered ........................................ no box

All other (non-target) tracks are drawn as thin neutral-grey context boxes.
For EACH recovery case: full-frame video + keyframe, and a ZOOM-IN video + keyframe.

MOT txt: frame,id,x,y,w,h,conf,-1,-1,-1  (1-indexed)
"""

import argparse
import os
from collections import defaultdict, Counter

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_mot(path, gt=False):
    data = defaultdict(list)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        f = line.split(",")
        fr, tid = int(float(f[0])), int(float(f[1]))
        x, y, w, h = map(float, f[2:6])
        if gt:
            conf = float(f[6]) if len(f) > 6 else 1.0
            cls = float(f[7]) if len(f) > 7 else 1.0
            vis = float(f[8]) if len(f) > 8 else 1.0
            if conf == 0 or int(cls) != 1 or vis <= 0:
                continue
        data[fr].append((tid, x, y, w, h))
    return data


def frames_by_id(data):
    out = defaultdict(dict)
    for fr, dets in data.items():
        for tid, x, y, w, h in dets:
            out[tid][fr] = (x, y, w, h)
    return out


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def best_match(pred_frame_dets, gt_box, iou_thr):
    """Return (tid, box) of the pred det with highest IoU >= thr, else None."""
    best, best_iou = None, iou_thr
    for tid, x, y, w, h in pred_frame_dets:
        v = iou((x, y, w, h), gt_box)
        if v >= best_iou:
            best_iou, best = v, (tid, (x, y, w, h))
    return best


def covered(pred_frame_dets, gt_box, iou_thr):
    return best_match(pred_frame_dets, gt_box, iou_thr) is not None


# --------------------------------------------------------------------------- #
# Core: find recoveries
# --------------------------------------------------------------------------- #
def find_recoveries(gt, sparse, ours, min_gap, iou_thr):
    gt_by_id = frames_by_id(gt)
    events = []
    for gid, track in gt_by_id.items():
        frs = sorted(track)
        sparse_ok, ours_ok = {}, {}
        for fr in frs:
            box = track[fr]
            sparse_ok[fr] = covered(sparse.get(fr, []), box, iou_thr)
            ours_ok[fr] = covered(ours.get(fr, []), box, iou_thr)
        i = 0
        while i < len(frs):
            fr = frs[i]
            if not sparse_ok[fr] and ours_ok[fr]:
                j = i
                while j < len(frs) and not sparse_ok[frs[j]] and ours_ok[frs[j]]:
                    j += 1
                run = frs[i:j]
                had_before = i > 0 and sparse_ok[frs[i - 1]]
                if len(run) >= min_gap and had_before:
                    events.append(
                        {
                            "gt_id": gid,
                            "start": run[0],
                            "end": run[-1],
                            "length": len(run),
                            "mid": run[len(run) // 2],
                        }
                    )
                i = j
            else:
                i += 1
    events.sort(key=lambda e: e["length"], reverse=True)
    return events


def baseline_sparse_id(gt_track, sparse, iou_thr, before_frame):
    """The sparse track id that most often matched this GT before the gap."""
    votes = Counter()
    for fr in sorted(gt_track):
        if fr >= before_frame:
            break
        m = best_match(sparse.get(fr, []), gt_track[fr], iou_thr)
        if m is not None:
            votes[m[0]] += 1
    return votes.most_common(1)[0][0] if votes else None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
NEUTRAL = (150, 150, 150)  # other/context tracks & normal target (grey, BGR)
YELLOW = (0, 255, 255)  # ours: recovered & held
RED = (0, 0, 255)  # sparse: id switch


def draw_context(img, dets, skip_box=None):
    """Thin grey boxes for all non-target tracks."""
    for tid, x, y, w, h in dets:
        if skip_box is not None and (x, y, w, h) == skip_box:
            continue
        p1 = (int(x), int(y))
        cv2.rectangle(img, p1, (int(x + w), int(y + h)), NEUTRAL, 1)
    return img


def draw_target(img, box, color, tid=None):
    x, y, w, h = box
    p1 = (int(x), int(y))
    cv2.rectangle(img, p1, (int(x + w), int(y + h)), color, 3)
    if tid is not None:
        cv2.putText(
            img,
            f"id{tid}",
            (p1[0], max(0, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return img


def label_bar(img, text, color=(30, 30, 30)):
    bar = np.full((34, img.shape[1], 3), color, np.uint8)
    cv2.putText(
        bar,
        text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([bar, img])


def read_frame(img_dir, fr, fallback_shape, offset=0):
    real = fr + offset
    for w in (6, 5, 4, 8):
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(img_dir, f"{real:0{w}d}{ext}")
            if os.path.isfile(p):
                im = cv2.imread(p)
                if im is not None:
                    return im
    return np.zeros(fallback_shape, np.uint8)


def resolve_offset(val_json, seq):
    import json as _json
    import re

    vj = _json.load(open(val_json))
    for im in vj["images"]:
        fn = im["file_name"]
        if fn.split("/")[0] == seq and im.get("frame_id") == 1:
            m = re.search(r"(\d+)\.(?:jpg|png|jpeg)", fn)
            if m:
                return int(m.group(1)) - 1
    raise ValueError(f"could not derive offset for seq '{seq}' from {val_json}")


def crop_box(shape, boxes, zoom_pad, out_size):
    H, W = shape[:2]
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[0] + b[2] for b in boxes]
    ys2 = [b[1] + b[3] for b in boxes]
    cx = (min(xs1) + max(xs2)) / 2
    cy = (min(ys1) + max(ys2)) / 2
    bw = (max(xs2) - min(xs1)) * (1 + zoom_pad)
    bh = (max(ys2) - min(ys1)) * (1 + zoom_pad)
    ar = out_size[0] / out_size[1]
    if bw / bh < ar:
        bw = bh * ar
    else:
        bh = bw / ar
    x1 = int(max(0, cx - bw / 2))
    y1 = int(max(0, cy - bh / 2))
    x2 = int(min(W, cx + bw / 2))
    y2 = int(min(H, cy + bh / 2))
    if x2 <= x1:
        x2 = min(W, x1 + 2)
    if y2 <= y1:
        y2 = min(H, y1 + 2)
    return x1, y1, x2, y2


def target_state(fr, e, gt_track, sparse, ours, iou_thr, baseline_id):
    """Return (left_draw, right_draw) where each is (box, color, tid) or None."""
    gbox = gt_track.get(fr)
    left = right = None
    if gbox is None:
        return None, None

    # LEFT: SparseTrack
    sm = best_match(sparse.get(fr, []), gbox, iou_thr)
    if sm is not None:
        stid, sbox = sm
        if baseline_id is not None and stid != baseline_id:
            left = (sbox, RED, stid)  # id switch
        else:
            left = (sbox, NEUTRAL, stid)  # normal
    # else: not covered -> no box

    # RIGHT: Ours
    om = best_match(ours.get(fr, []), gbox, iou_thr)
    if om is not None:
        otid, obox = om
        if fr >= e["end"]:  # recovered & held
            right = (obox, YELLOW, otid)
        else:
            right = (obox, NEUTRAL, otid)
    return left, right


def compose(
    base, fr, sparse_dets, ours_dets, left_t, right_t, crop=None, out_size=None
):
    left, right = base.copy(), base.copy()
    # context (skip the target box so it isn't double-drawn thin)
    draw_context(left, sparse_dets, skip_box=left_t[0] if left_t else None)
    draw_context(right, ours_dets, skip_box=right_t[0] if right_t else None)
    # target on top
    if left_t:
        draw_target(left, *left_t)
    if right_t:
        draw_target(right, *right_t)
    if crop is not None:
        x1, y1, x2, y2 = crop
        left, right = left[y1:y2, x1:x2], right[y1:y2, x1:x2]
        if out_size is not None:
            left = cv2.resize(left, out_size, interpolation=cv2.INTER_LINEAR)
            right = cv2.resize(right, out_size, interpolation=cv2.INTER_LINEAR)
    left = label_bar(left, f"SparseTrack   frame {fr}")
    right = label_bar(right, f"Ours (GLRA)   frame {fr}")
    gap = np.full((left.shape[0], 20, 3), 255, np.uint8)
    return np.hstack([left, gap, right])


def probe_size(sparse, ours, img_dir, offset):
    for fr in sorted(set(sparse) | set(ours)):
        s = read_frame(img_dir, fr, (0, 0, 0), offset)
        if s.size:
            return s.shape[:2]
    raise RuntimeError("No readable frames found in --img")


def render_case(
    idx,
    e,
    sparse,
    ours,
    gt_by_id,
    img_dir,
    out_dir,
    fps,
    pad,
    offset,
    fallback,
    iou_thr,
    zoom_pad,
    zoom_size,
):
    tag = f"case{idx:02d}_id{e['gt_id']}_f{e['start']}-{e['end']}_gap{e['length']}"
    kf_dir = os.path.join(out_dir, "keyframes")
    gt_track = gt_by_id.get(e["gt_id"], {})
    baseline = baseline_sparse_id(gt_track, sparse, iou_thr, e["start"])

    H, W = fallback[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_full = cv2.VideoWriter(
        os.path.join(out_dir, f"{tag}.mp4"), fourcc, fps, (W * 2 + 20, H + 34)
    )
    zw, zh = zoom_size
    vw_zoom = cv2.VideoWriter(
        os.path.join(out_dir, f"{tag}_zoom.mp4"), fourcc, fps, (zw * 2 + 20, zh + 34)
    )

    # zoom crop: union of target GT boxes across the rendered window
    win = [f for f in range(e["start"] - pad, e["end"] + pad + 1) if f in gt_track]
    gap_boxes = [gt_track[f] for f in win] or [(0, 0, W, H)]
    crop = crop_box(fallback, gap_boxes, zoom_pad, zoom_size)

    n = 0
    for fr in range(e["start"] - pad, e["end"] + pad + 1):
        if fr < 1:
            continue
        base = read_frame(img_dir, fr, fallback, offset)
        lt, rt = target_state(fr, e, gt_track, sparse, ours, iou_thr, baseline)
        sd, od = sparse.get(fr, []), ours.get(fr, [])
        vw_full.write(compose(base, fr, sd, od, lt, rt))
        vw_zoom.write(compose(base, fr, sd, od, lt, rt, crop=crop, out_size=zoom_size))
        n += 1
    vw_full.release()
    vw_zoom.release()

    # keyframe: first recovered frame (gap end) shows the yellow box
    fr = e["end"]
    base = read_frame(img_dir, fr, fallback, offset)
    lt, rt = target_state(fr, e, gt_track, sparse, ours, iou_thr, baseline)
    sd, od = sparse.get(fr, []), ours.get(fr, [])
    cv2.imwrite(os.path.join(kf_dir, f"{tag}.png"), compose(base, fr, sd, od, lt, rt))
    cv2.imwrite(
        os.path.join(kf_dir, f"{tag}_zoom.png"),
        compose(base, fr, sd, od, lt, rt, crop=crop, out_size=zoom_size),
    )
    return tag, n, baseline


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-gap", type=int, default=5)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--pad", type=int, default=15)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-cases", type=int, default=20)
    ap.add_argument("--zoom-pad", type=float, default=1.0)
    ap.add_argument("--zoom-w", type=int, default=320)
    ap.add_argument("--zoom-h", type=int, default=480)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--val-json", default=None)
    ap.add_argument("--seq", default=None)
    args = ap.parse_args()

    offset = args.offset
    if args.val_json:
        seq = args.seq
        if seq is None:
            parts = os.path.normpath(args.img).split(os.sep)
            seq = parts[-2] if parts[-1] in ("img1", "img") else parts[-1]
        offset = resolve_offset(args.val_json, seq)
        print(f"[offset] {seq}: rel f1 -> {offset + 1:06d}  (offset={offset})")

    print("Loading...")
    gt = load_mot(args.gt, gt=True)
    sparse = load_mot(args.sparse)
    ours = load_mot(args.ours)

    print("Finding recoveries...")
    events = find_recoveries(gt, sparse, ours, args.min_gap, args.iou_thr)

    print(
        f"\n{len(events)} recovery cases "
        f"(SparseTrack drops >= {args.min_gap} frames that Ours covers):"
    )
    for e in events[: args.max_cases]:
        print(
            f"  GT id {e['gt_id']:>4}  frames {e['start']}-{e['end']}  "
            f"({e['length']} frames)"
        )
    if len(events) > args.max_cases:
        print(f"  ... and {len(events) - args.max_cases} more (not rendered)")
    if not events:
        print("\nNothing to render. Try lowering --min-gap or --iou-thr.")
        return

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "keyframes"), exist_ok=True)
    H, W = probe_size(sparse, ours, args.img, offset)
    fallback = (H, W, 3)
    gt_by_id = frames_by_id(gt)
    zoom_size = (args.zoom_w, args.zoom_h)

    print(f"\nRendering {min(len(events), args.max_cases)} cases (full + zoom)...")
    for idx, e in enumerate(events[: args.max_cases]):
        tag, n, base_id = render_case(
            idx,
            e,
            sparse,
            ours,
            gt_by_id,
            args.img,
            args.out,
            args.fps,
            args.pad,
            offset,
            fallback,
            args.iou_thr,
            args.zoom_pad,
            zoom_size,
        )
        print(f"  [{idx:02d}] {tag}  ({n} frames)  baseline sparse id={base_id}")

    print(f"\nDone.")
    print(f"  full videos : {args.out}/case*.mp4")
    print(f"  zoom videos : {args.out}/case*_zoom.mp4")
    print(f"  keyframes   : {os.path.join(args.out, 'keyframes')}/")


if __name__ == "__main__":
    main()

"""
python glra_compare.py \
    --sparse /home/caig/repo/SparseTrack/yolox_mix17_ablation/yolox_mix17_ablation_det/track_results_sparsetrack/MOT17-13-FRCNN.txt \
    --ours   /home/caig/repo/SparseTrack/yolox_mix17_ablation/yolox_mix17_ablation_det/track_results_glra/MOT17-13-FRCNN.txt \
    --gt     /home/caig/data/MOT17/train/MOT17-13-FRCNN/gt/gt.txt \
    --img    /home/caig/data/MOT17/train/MOT17-13-FRCNN/img1 \
    --out    /home/caig/repo/SparseTrack/out/MOT17-13-FRCNN \
    --val-json /home/caig/data/MOT17/annotations/val_half.json

python glra_compare.py \
    --sparse /home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results_sparsetrack/MOT20-03.txt \
    --ours   /home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results_glra/MOT20-03.txt \
    --gt     /home/caig/data/MOT20/train/MOT20-03/gt/gt.txt \
    --img    /home/caig/data/MOT20/train/MOT20-03/img1 \
    --out    /home/caig/repo/SparseTrack/out/MOT20-03 \
    --val-json /home/caig/data/MOT20/annotations/val_half.json
"""
