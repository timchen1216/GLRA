#!/usr/bin/env python3
"""
glra_compare.py — Highlight where the proposed method (GLRA) beats SparseTrack.

Focus: recovered tracks — GT identities that SparseTrack loses (a coverage gap
in the middle of an existing track) but the proposed method keeps covered.
Produces a side-by-side comparison video (left: SparseTrack, right: Ours) and
PNG snapshots of the key frames where a recovery happens.

MOT txt format (per line):  frame,id,x,y,w,h,conf,-1,-1,-1   (1-indexed frames)

Example
-------
python glra_compare.py \
    --sparse results/MOT20-05_sparsetrack.txt \
    --ours   results/MOT20-05_glra.txt \
    --gt     MOT20/train/MOT20-05/gt/gt.txt \
    --img    MOT20/train/MOT20-05/img1 \
    --out    out/MOT20-05 \
    --min-gap 5 --iou-thr 0.5
"""

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_mot(path, gt=False):
    """Return {frame: [(id, x, y, w, h), ...]} keeping only valid rows."""
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
            # MOTChallenge gt: col7=consider flag, col8=class, col9=visibility
            conf = float(f[6]) if len(f) > 6 else 1.0
            cls = float(f[7]) if len(f) > 7 else 1.0
            vis = float(f[8]) if len(f) > 8 else 1.0
            if (
                conf == 0 or int(cls) != 1 or vis <= 0
            ):  # keep only evaluated pedestrians
                continue
        data[fr].append((tid, x, y, w, h))
    return data


def frames_by_id(data):
    """{id: {frame: (x,y,w,h)}} from a {frame:[...]} dict."""
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


def covered(pred_frame_dets, gt_box, iou_thr):
    """Is gt_box matched by any predicted box in this frame?"""
    for _, x, y, w, h in pred_frame_dets:
        if iou((x, y, w, h), gt_box) >= iou_thr:
            return True
    return False


# --------------------------------------------------------------------------- #
# Core: find recoveries
# --------------------------------------------------------------------------- #
def find_recoveries(gt, sparse, ours, min_gap, iou_thr):
    """
    A 'recovery event' = a maximal run of consecutive GT frames for one GT id
    where SparseTrack fails to cover the GT box but Ours does, with the run
    bracketed by frames both methods cover (i.e. a genuine mid-track gap that
    SparseTrack drops and Ours fills). Returns list of dicts.
    """
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
            # start of a gap: sparse fails, ours succeeds
            if not sparse_ok[fr] and ours_ok[fr]:
                j = i
                while j < len(frs) and not sparse_ok[frs[j]] and ours_ok[frs[j]]:
                    j += 1
                run = frs[i:j]
                # require sparse to have HAD this id just before the gap
                # (mid-track drop, not a track that never started)
                had_before = i > 0 and sparse_ok[frs[i - 1]]
                if len(run) >= min_gap and had_before:
                    events.append(
                        {
                            "gt_id": gid,
                            "start": run[0],
                            "end": run[-1],
                            "length": len(run),
                            "box": track[run[len(run) // 2]],  # mid-gap box
                            "mid": run[len(run) // 2],
                        }
                    )
                i = j
            else:
                i += 1

    events.sort(key=lambda e: e["length"], reverse=True)
    return events


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_PALETTE = [
    (66, 133, 244),
    (219, 68, 55),
    (244, 180, 0),
    (15, 157, 88),
    (171, 71, 188),
    (0, 172, 193),
    (255, 112, 67),
    (158, 157, 36),
    (94, 53, 177),
    (0, 121, 107),
    (233, 30, 99),
    (121, 85, 72),
]


def color_for(tid):
    return _PALETTE[tid % len(_PALETTE)]


def draw_dets(img, dets, highlight_box=None, iou_thr=0.5):
    for tid, x, y, w, h in dets:
        c = color_for(tid)
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(img, p1, p2, c, 2)
        cv2.putText(
            img,
            str(tid),
            (p1[0], max(0, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            c,
            1,
            cv2.LINE_AA,
        )
    if highlight_box is not None:
        x, y, w, h = highlight_box
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 3)
    return img


def label_bar(img, text, color=(30, 30, 30)):
    h_bar = 34
    bar = np.full((h_bar, img.shape[1], 3), color, np.uint8)
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
    """Look up an image frame. `offset` maps a val-half (relative) frame id
    to the absolute image number: image_number = fr + offset."""
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
    """Derive image_number offset from a val_half COCO json for `seq`:
    finds the absolute image number of that sequence's relative frame 1."""
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


def render(sparse, ours, gt, events, img_dir, out_dir, fps, iou_thr, pad, offset=0):
    os.makedirs(out_dir, exist_ok=True)
    png_dir = os.path.join(out_dir, "keyframes")
    os.makedirs(png_dir, exist_ok=True)

    # probe a frame for size
    sample = None
    for fr in sorted(set(sparse) | set(ours)):
        s = read_frame(img_dir, fr, (0, 0, 0), offset)
        if s.size:
            sample = s
            break
    if sample is None:
        raise RuntimeError("No readable frames found in --img")
    H, W = sample.shape[:2]
    fallback = (H, W, 3)

    # frames to render = union of event windows (± pad)
    render_frames = set()
    for e in events:
        for fr in range(e["start"] - pad, e["end"] + pad + 1):
            if fr >= 1:
                render_frames.add(fr)
    render_frames = sorted(render_frames)
    if not render_frames:
        print("No recovery events found — nothing to render.")
        return

    gt_by_id = frames_by_id(gt)

    out_w, out_h = W * 2 + 20, H + 34
    vw = cv2.VideoWriter(
        os.path.join(out_dir, "comparison.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    # map each frame -> active events (for the yellow highlight box)
    active = defaultdict(list)
    for e in events:
        for fr in range(e["start"] - pad, e["end"] + pad + 1):
            active[fr].append(e)

    for fr in render_frames:
        base = read_frame(img_dir, fr, fallback, offset)
        left, right = base.copy(), base.copy()

        hl = None
        for e in active.get(fr, []):
            gbox = gt_by_id.get(e["gt_id"], {}).get(fr)
            if gbox is not None:
                hl = gbox  # highlight the GT box being recovered
        draw_dets(left, sparse.get(fr, []))
        draw_dets(right, ours.get(fr, []), highlight_box=hl, iou_thr=iou_thr)

        left = label_bar(left, f"SparseTrack   frame {fr}")
        right = label_bar(right, f"Ours (GLRA)   frame {fr}")
        gap = np.full((left.shape[0], 20, 3), 255, np.uint8)
        vw.write(np.hstack([left, gap, right]))

    vw.release()

    # key-frame PNGs: mid-gap frame of each event (top 12 by length)
    for k, e in enumerate(events[:12]):
        fr = e["mid"]
        base = read_frame(img_dir, fr, fallback, offset)
        left, right = base.copy(), base.copy()
        gbox = gt_by_id.get(e["gt_id"], {}).get(fr)
        draw_dets(left, sparse.get(fr, []))
        draw_dets(right, ours.get(fr, []), highlight_box=gbox, iou_thr=iou_thr)
        left = label_bar(left, f"SparseTrack  f{fr}  (GT id {e['gt_id']} dropped)")
        right = label_bar(right, f"Ours  f{fr}  (recovered, {e['length']} frm gap)")
        gap = np.full((left.shape[0], 20, 3), 255, np.uint8)
        png = np.hstack([left, gap, right])
        name = f"recovery_{k:02d}_id{e['gt_id']}_f{e['start']}-{e['end']}.png"
        cv2.imwrite(os.path.join(png_dir, name), png)

    return len(render_frames)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sparse", required=True, help="SparseTrack result txt")
    ap.add_argument("--ours", required=True, help="Your method result txt")
    ap.add_argument("--gt", required=True, help="Ground-truth gt.txt")
    ap.add_argument("--img", required=True, help="img1 frame directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument(
        "--min-gap",
        type=int,
        default=5,
        help="min consecutive recovered frames to count (default 5)",
    )
    ap.add_argument(
        "--iou-thr",
        type=float,
        default=0.5,
        help="IoU threshold for GT coverage (default 0.5)",
    )
    ap.add_argument(
        "--pad",
        type=int,
        default=15,
        help="frames of context before/after each event (default 15)",
    )
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="val_frame_id + offset = absolute image number",
    )
    ap.add_argument(
        "--val-json",
        default=None,
        help="val_half COCO json; auto-derive offset for --seq",
    )
    ap.add_argument(
        "--seq",
        default=None,
        help="sequence name for --val-json lookup "
        "(e.g. MOT17-11-FRCNN); inferred from --img if omitted",
    )
    args = ap.parse_args()

    offset = args.offset
    if args.val_json:
        seq = args.seq
        if seq is None:
            # infer from img dir: .../<SEQ>/img1
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
        f"\n{len(events)} recovery events "
        f"(SparseTrack drops >= {args.min_gap} frames that Ours covers):"
    )
    tot = 0
    for e in events[:20]:
        tot += e["length"]
        print(
            f"  GT id {e['gt_id']:>4}  frames {e['start']}-{e['end']}  "
            f"({e['length']} frames)"
        )
    if len(events) > 20:
        print(f"  ... and {len(events) - 20} more")
    print(f"Total recovered frame-instances: " f"{sum(e['length'] for e in events)}")

    if not events:
        print("\nNothing to render. Try lowering --min-gap or --iou-thr.")
        return

    print("\nRendering video + keyframes...")
    nf = render(
        sparse,
        ours,
        gt,
        events,
        args.img,
        args.out,
        args.fps,
        args.iou_thr,
        args.pad,
        offset,
    )
    print(f"\nDone.")
    print(f"  video    : {os.path.join(args.out, 'comparison.mp4')} ({nf} frames)")
    print(f"  keyframes: {os.path.join(args.out, 'keyframes')}/")


if __name__ == "__main__":
    main()
"""
# t268, MOT17-09 固定相機, offset 263
python overlay_glra_on_frame.py \
  --case glra_cases/MOT17-09-FRCNN_t268_f86.json \
  --img_dir /home/caig/data/MOT17/train/MOT17-09-FRCNN/img1 \
  --val_json /home/caig/data/MOT17/annotations/val_half.json \
  --out overlay_t268.jpg
"""
