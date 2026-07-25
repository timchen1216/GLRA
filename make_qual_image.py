#!/usr/bin/env python3
"""
Produce ONE clean annotated image for the paper's qualitative figure
(fig:glra_qualitative): the real recover-frame with the full observed
trajectory (green polyline) plus KF / GPR / det boxes. Labels are drawn with
leader lines and pushed to image margins so they never overlap the boxes,
which sit close together at the recover frame.

Frame-id offset: pass --val_json (auto) or --offset (manual); absolute image
number = dumped(rel) frame + offset.

Usage
-----
python make_qual_image.py \
  --case glra_cases/MOT20-05_t1196_f173.json \
  --img_dir /home/caig/data/MOT20/train/MOT20-05/img1 \
  --val_json /path/to/val_half.json \
  --out fig_glra_qualitative.png

Requires: opencv-python, numpy.
"""

import os, re, json, argparse
import numpy as np
import cv2

# BGR
C_OBS = (73, 110, 44)[::-1]  # green
C_KF = (31, 18, 193)  # red
C_GPR = (137, 78, 29)  # blue
C_DET = (158, 55, 181)  # magenta
WHITE = (255, 255, 255)


def resolve_offset(rec, val_json, manual):
    if val_json:
        vj = json.load(open(val_json))
        seq = rec["seq"]
        for im in vj["images"]:
            if im["file_name"].split("/")[0] == seq and im["frame_id"] == 1:
                real = int(re.search(r"(\d+)\.jpg", im["file_name"]).group(1))
                return real - 1
    return manual


def find_frame(img_dir, abs_id):
    for w in (6, 5, 4, 8):
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(img_dir, f"{abs_id:0{w}d}{ext}")
            if os.path.exists(p):
                return p
    return None


def bc(cx, cy, w, h):
    return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


def dashed_rect(img, x1, y1, x2, y2, color, t=2, dash=9):
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, t)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, t)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, t)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, t)


def labelbox(img, text, anchor, tip, color, scale=0.9, pad=6):
    """Draw a filled label at `anchor` (top-left) with a leader line to `tip`."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = anchor
    x = int(max(4, min(x, img.shape[1] - tw - pad * 2 - 4)))
    y = int(max(th + pad * 2, y))
    # leader line from label edge to tip
    cv2.line(img, (x + tw // 2, y), tip, color, 2, cv2.LINE_AA)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), color, -1)
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 2, cv2.LINE_AA
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out", default="fig_glra_qualitative.png")
    ap.add_argument("--val_json", default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--crop_pad",
        type=float,
        default=2.2,
        help="crop margin around the trajectory, in bbox widths " "(0 = full frame)",
    )
    a = ap.parse_args()

    rec = json.load(open(a.case))
    off = resolve_offset(rec, a.val_json, a.offset)
    abs_f = rec["frame"] + off
    path = find_frame(a.img_dir, abs_f)
    if path is None:
        raise SystemExit(f"no image for abs frame {abs_f} in {a.img_dir}")
    img = cv2.imread(path)
    H, W = img.shape[:2]
    src = "val_json" if a.val_json else ("--offset" if a.offset else "NONE(=0!)")
    print(
        f"[offset] {rec['seq']} off={off} (source: {src})  "
        f"recover rel f{rec['frame']} -> abs {abs_f:06d}.jpg"
    )
    if off == 0:
        print("  WARNING: offset=0 -> you are reading the WRONG absolute frame!")
        print("  Pass --val_json <val_half.json> or --offset (MOT20-05 = 1658).")

    obs = rec["obs"]
    cx = np.array([o[1] for o in obs])
    cy = np.array([o[2] for o in obs])
    g, k, d = rec["gpr_pred"], rec["kf_pred"], rec["det"]

    # trajectory polyline + points
    pts = [(int(x), int(y)) for x, y in zip(cx, cy)]
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], C_OBS, 3, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 4, C_OBS, -1, cv2.LINE_AA)

    # last observed bbox
    lo = obs[-1]
    x1, y1, x2, y2 = bc(lo[1], lo[2], lo[3], lo[4])
    cv2.rectangle(img, (x1, y1), (x2, y2), C_OBS, 2)

    # KF / GPR / det boxes
    kx1, ky1, kx2, ky2 = bc(k["cx"], k["cy"], k["w"], k["h"])
    dashed_rect(img, kx1, ky1, kx2, ky2, C_KF, 2)
    gx1, gy1, gx2, gy2 = bc(g["cx"], g["cy"], g["w"], g["h"])
    dashed_rect(img, gx1, gy1, gx2, gy2, C_GPR, 2)
    dx1, dy1, dx2, dy2 = bc(d["cx"], d["cy"], d["w"], d["h"])
    cv2.rectangle(img, (dx1, dy1), (dx2, dy2), C_DET, 2)

    # crop around trajectory for a tighter, readable figure
    if a.crop_pad > 0:
        bw = float(np.median([o[3] for o in obs]))
        bh = float(np.median([o[4] for o in obs]))
        xs = [cx.min() - bw / 2, cx.max() + bw / 2, kx1, kx2, gx1, gx2, dx1, dx2]
        ys = [cy.min() - bh / 2, cy.max() + bh / 2, ky1, ky2, gy1, gy2, dy1, dy2]
        cxmin, cxmax = min(xs), max(xs)
        cymin, cymax = min(ys), max(ys)
        px = bw * a.crop_pad
        py = bh * a.crop_pad
        X1 = int(max(0, cxmin - px))
        X2 = int(min(W, cxmax + px))
        Y1 = int(max(0, cymin - py))
        Y2 = int(min(H, cymax + py))
    else:
        X1, Y1, X2, Y2 = 0, 0, W, H

    # labels: place at crop margins with leader lines to box centers
    # (draw on full image first, then crop)
    labelbox(
        img,
        "KF",
        (int(k["cx"]) + 60, ky1 - 20),
        (int((kx1 + kx2) / 2), int((ky1 + ky2) / 2)),
        C_KF,
    )
    labelbox(
        img,
        "GPR",
        (int(g["cx"]) + 60, gy2 + 40),
        (int((gx1 + gx2) / 2), int((gy1 + gy2) / 2)),
        C_GPR,
    )
    labelbox(
        img,
        f"det {d['score']:.2f}",
        (dx1 - 150, dy2 + 20),
        (int((dx1 + dx2) / 2), int((dy1 + dy2) / 2)),
        C_DET,
    )

    crop = img[Y1:Y2, X1:X2]
    cv2.imwrite(a.out, crop)
    # also full-frame version
    full_out = os.path.splitext(a.out)[0] + "_full.jpg"
    cv2.imwrite(full_out, img)
    print("wrote", a.out, "and", full_out)
    dk = np.hypot(k["cx"] - d["cx"], k["cy"] - d["cy"])
    dg = np.hypot(g["cx"] - d["cx"], g["cy"] - d["cy"])
    print(f"KF->det {dk:.1f}  GPR->det {dg:.1f}")


if __name__ == "__main__":
    main()
"""
python make_qual_image.py \
  --case glra_cases/MOT20-05_t1196_f173.json \
  --img_dir /home/caig/data/MOT20/train/MOT20-05/img1 \
  --val_json /home/caig/data/MOT17/annotations/val_half.json  \
  --out fig_glra_qualitative.png
"""
