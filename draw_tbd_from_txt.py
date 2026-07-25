"""
draw_tbd_from_txt.py

Make three *illustrative* tracking-by-detection stage images from a standard
MOT results txt (frame,id,x,y,w,h,conf,...). All three use the SAME boxes of a
chosen frame; only the coloring differs:

    1_detect.png   all boxes filled YELLOW  (pretend: detections, no IDs)
    2_predict.png  all boxes filled RED     (pretend: KF predictions)
    4_update.png   per-track-id colors      (real: updated tracks with IDs)

This is for slide illustration only. The txt holds final tracks, so 1 and 2 are
stand-ins, not true intermediate states.

Usage:
    python draw_tbd_from_txt.py \
        --txt  MOT20-05.txt \
        --img  datasets/MOT20/train/MOT20-05/img1/000173.jpg \
        --frame 173 \
        --out  ./tbd_figs \
        [--fill-alpha 0.35] [--show-ids]
"""

import argparse
import os

import cv2
import numpy as np


def _hue_to_bgr(h_deg, s=0.65, v=0.90):
    """HSV (hue in degrees) -> BGR uint8 tuple."""
    hsv = np.uint8(
        [[[int(h_deg / 2) % 180, int(s * 255), int(v * 255)]]]
    )  # OpenCV hue 0..179
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def build_frame_colors(boxes):
    """Approach B: assign colors per frame so that spatially adjacent boxes get
    maximally different hues. Yellow/red hue bands are excluded (reserved for the
    detect/predict stages). Returns {track_id: bgr}.

    Not stable across frames by design -- optimized for single-frame clarity.
    """
    n = len(boxes)
    if n == 0:
        return {}

    # Allowed hue ranges in degrees, skipping yellow (~35-70) and red/orange
    # (~-25..25). Keep green, cyan, blue, purple, magenta.
    allowed = []
    for h in range(0, 360, 2):
        if 30 <= h <= 72:  # yellow band -> skip
            continue
        if h <= 25 or h >= 340:  # red / orange band -> skip
            continue
        allowed.append(h)
    # Evenly sample n hues across the allowed set (maximally spaced).
    hues = [allowed[int(round(i * (len(allowed) - 1) / max(1, n)))] for i in range(n)]

    # Order boxes left-to-right (by center x), then interleave the spaced hues so
    # that neighbors in space are far apart in hue.
    order = sorted(range(n), key=lambda i: boxes[i][1] + boxes[i][3] / 2.0)  # x + w/2
    # bit-reversal-ish interleave: take hues from alternating ends
    interleaved = []
    lo, hi = 0, len(hues) - 1
    take_low = True
    while lo <= hi:
        if take_low:
            interleaved.append(hues[lo])
            lo += 1
        else:
            interleaved.append(hues[hi])
            hi -= 1
        take_low = not take_low

    color_map = {}
    for slot, box_idx in enumerate(order):
        tid = boxes[box_idx][0]
        color_map[tid] = _hue_to_bgr(interleaved[slot])
    return color_map


def load_frame_boxes(txt_path, frame):
    """Return list of (track_id, x, y, w, h) for the given frame."""
    boxes = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            fr = int(float(p[0]))
            if fr != frame:
                continue
            tid = int(float(p[1]))
            x, y, w, h = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
            boxes.append((tid, x, y, w, h))
    return boxes


def draw(base, boxes, mode, fill_alpha=0.35, show_ids=False, color_map=None):
    img = base.copy()
    overlay = img.copy()
    for tid, x, y, w, h in boxes:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        if mode == "detect":
            color = (0, 220, 255)  # yellow
        elif mode == "predict":
            color = (60, 60, 220)  # red
        else:  # update
            color = color_map[tid]
        # translucent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)
    # solid outline on top (and optional label)
    for tid, x, y, w, h in boxes:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        if mode == "detect":
            color = (0, 220, 255)
        elif mode == "predict":
            color = (60, 60, 220)
        else:
            color = color_map[tid]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if show_ids and mode == "update":
            label = f"ID {tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True, help="MOT results txt")
    ap.add_argument("--img", required=True, help="frame image")
    ap.add_argument("--frame", type=int, required=True, help="frame id in txt")
    ap.add_argument("--out", default=".", help="output dir")
    ap.add_argument(
        "--fill-alpha",
        type=float,
        default=0.35,
        help="box fill opacity 0..1 (default 0.35)",
    )
    ap.add_argument(
        "--show-ids", action="store_true", help="draw ID labels on the update image"
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    boxes = load_frame_boxes(args.txt, args.frame)
    if not boxes:
        raise SystemExit(f"no boxes for frame {args.frame} in {args.txt}")
    base = cv2.imread(args.img)
    if base is None:
        raise SystemExit(f"could not read image {args.img}")

    color_map = build_frame_colors(boxes)
    for fname, mode in [
        ("1_detect.png", "detect"),
        ("2_predict.png", "predict"),
        ("4_update.png", "update"),
    ]:
        out = draw(base, boxes, mode, args.fill_alpha, args.show_ids, color_map)
        path = os.path.join(args.out, fname)
        cv2.imwrite(path, out)
        print(f"wrote {path}  ({len(boxes)} boxes, mode={mode})")


if __name__ == "__main__":
    main()

"""
python draw_tbd_from_txt.py \
    --txt  /home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results_dti/MOT17-09-FRCNN.txt \
    --img  /home/caig/data/MOT17/train/MOT17-09-FRCNN/img1/000060.jpg \
    --frame 60 \
    --out  ./tbd_figs \
    --show-ids
"""
