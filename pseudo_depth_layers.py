#!/usr/bin/env python3
"""
pseudo_depth_layers.py

Visualize SparseTrack-style pseudo-depth stratification.

Given an image and a MOTChallenge-format detection/tracking file, this script:
  1. computes each box's pseudo-depth  L_p = H - y_bottom
     (larger L_p  ->  nearer to the camera; smaller L_p -> farther),
  2. partitions the boxes into K depth-ordered subsets by dividing the
     [L_min, L_max] value range uniformly into K intervals (SparseTrack's
     "sparse decomposition": this is range-based, so some layers may be empty
     and layer sizes are naturally uneven), and
  3. draws every box, colored by the depth layer it falls into.

This matches SparseTrack's original partitioning (uniform split of the depth
*value range*, NOT an equal-count split of the boxes).

Layer 0 is the FARTHEST subset (smallest L_p) and layer K-1 is the NEAREST
(largest L_p), following the paper's "near to far" cascade indexing convention
where matching proceeds from the nearest subset. You can flip the labeling with
--near-first if you prefer layer 0 to be the nearest.

Input file format (MOTChallenge, comma- or space-separated):
    frame, id, x, y, w, h, conf, ...
where (x, y) is the top-left corner and (w, h) the box size in pixels.

Usage
-----
Single image (its frame is inferred from the filename's trailing number, or
pass --frame to force one):

    python pseudo_depth_layers.py --img seq/img1/000173.jpg --txt gt.txt \
        --levels 3 --out layered_173.jpg

Whole sequence -> annotated frames in a folder (one PNG per image):

    python pseudo_depth_layers.py --imgs seq/img1 --txt gt.txt \
        --levels 3 --outdir layered_out

Notes
-----
* --levels is K (number of depth subsets). MOT17 uses 3, MOT20 uses 8 in the
  SparseTrack test config (validation may differ); pick to match your setup.
* Pseudo-depth uses the box BOTTOM edge (y + h), consistent with the ground
  -contact-point assumption.
"""

import argparse
import os
import sys
import glob
import re

import numpy as np
import cv2

# --------------------------------------------------------------------------- #
# Distinct, fixed colors per depth layer (BGR). Ordered from far -> near so the
# palette reads intuitively (cool/dim for far, warm/bright for near).
# Extended procedurally if --levels exceeds this list.
# --------------------------------------------------------------------------- #
BASE_LAYER_COLORS = [
    (180, 119, 31),  # blue      (far)
    (14, 127, 255),  # orange
    (44, 160, 44),  # green
    (40, 39, 214),  # red
    (189, 103, 148),  # purple
    (34, 189, 188),  # yellow-green
    (207, 190, 23),  # cyan
    (194, 119, 227),  # pink       (near)
]


def layer_color(layer_idx: int, n_layers: int):
    """Return a stable BGR color for a given layer index."""
    if layer_idx < len(BASE_LAYER_COLORS):
        return BASE_LAYER_COLORS[layer_idx]
    # Fallback: evenly spaced hues on the HSV wheel.
    import colorsys

    hue = (layer_idx / max(1, n_layers)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


# --------------------------------------------------------------------------- #
# Parse the MOT-format file into per-frame boxes.
# --------------------------------------------------------------------------- #
def load_boxes(txt_path: str):
    """Return dict[int frame] -> list of (id, x, y, w, h)."""
    frames = {}
    with open(txt_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 6:
                continue
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                x, y, w, h = (
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
            except ValueError:
                continue
            frames.setdefault(frame, []).append((tid, x, y, w, h))
    if not frames:
        sys.exit(f"[error] No valid rows parsed from {txt_path}")
    return frames


# --------------------------------------------------------------------------- #
# SparseTrack pseudo-depth stratification.
# --------------------------------------------------------------------------- #
def assign_depth_layers(boxes, img_h, n_levels, near_first=False):
    """
    Compute pseudo-depth for each box and assign it to one of n_levels layers
    by uniformly partitioning the [L_min, L_max] VALUE range (SparseTrack style).

    Parameters
    ----------
    boxes : list of (id, x, y, w, h)
    img_h : image height in pixels (for L_p = H - y_bottom)
    n_levels : K, number of depth subsets
    near_first : if True, layer 0 = nearest (largest L_p); otherwise layer 0 =
                 farthest (smallest L_p), matching the paper's near-to-far
                 cascade where indexing starts at the nearest is handled by the
                 caller. Default False -> layer 0 is farthest.

    Returns
    -------
    list of (id, x, y, w, h, L_p, layer_idx)
    """
    if not boxes:
        return []

    # Pseudo-depth: larger = nearer.
    enriched = []
    for tid, x, y, w, h in boxes:
        y_bottom = y + h
        L_p = img_h - y_bottom
        enriched.append([tid, x, y, w, h, L_p])

    Ls = np.array([e[5] for e in enriched], dtype=np.float64)
    L_min, L_max = float(Ls.min()), float(Ls.max())

    # Uniform partition of the value range into n_levels intervals.
    # Edges: L_min = e0 < e1 < ... < e_{K} = L_max
    if L_max - L_min < 1e-9:
        # All boxes at (nearly) the same depth -> single populated layer.
        edges = np.linspace(L_min, L_min + 1.0, n_levels + 1)
    else:
        edges = np.linspace(L_min, L_max, n_levels + 1)

    out = []
    for e in enriched:
        L_p = e[5]
        # Find interval index in [0, n_levels-1]; np.searchsorted on inner edges.
        idx = int(np.searchsorted(edges[1:-1], L_p, side="right"))
        idx = min(max(idx, 0), n_levels - 1)
        # idx here counts from far (small L_p, idx 0) to near (large L_p).
        far_to_near_idx = idx
        if near_first:
            layer = (n_levels - 1) - far_to_near_idx
        else:
            layer = far_to_near_idx
        out.append((e[0], e[1], e[2], e[3], e[4], L_p, layer))
    return out


# --------------------------------------------------------------------------- #
# Drawing.
# --------------------------------------------------------------------------- #
def draw_layers(img, assigned, n_levels, thickness=2, show_id=False, legend=True):
    """Draw all boxes colored by depth layer, with an optional legend."""
    h_img = img.shape[0]

    # Count boxes per layer for the legend.
    counts = [0] * n_levels
    for item in assigned:
        counts[item[6]] += 1

    for tid, x, y, w, h, L_p, layer in assigned:
        color = layer_color(layer, n_levels)
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = f"L{layer}"
        if show_id:
            label = f"{tid}|L{layer}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - base - 3)
        cv2.rectangle(img, (x1, ty1), (x1 + tw + 4, ty1 + th + base + 3), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, ty1 + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if legend:
        _draw_legend(img, n_levels, counts)
    return img


def _draw_legend(img, n_levels, counts):
    """Small legend box: layer color -> 'Layer k (near/far, N boxes)'."""
    pad = 8
    row_h = 22
    box = 14
    width = 230
    height = pad * 2 + row_h * n_levels + row_h  # +1 title row
    x0, y0 = 10, 10

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + height), (60, 60, 60), 1)

    cv2.putText(
        img,
        "Pseudo-depth layers",
        (x0 + pad, y0 + pad + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )

    for k in range(n_levels):
        cy = y0 + pad + row_h + k * row_h
        color = layer_color(k, n_levels)
        cv2.rectangle(img, (x0 + pad, cy), (x0 + pad + box, cy + box), color, -1)
        tag = "near" if k == n_levels - 1 else ("far" if k == 0 else "")
        txt = f"Layer {k}" + (f"  ({tag})" if tag else "")
        txt += f"  x{counts[k]}"
        cv2.putText(
            img,
            txt,
            (x0 + pad + box + 8, cy + box - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )


# --------------------------------------------------------------------------- #
# Frame-index helpers.
# --------------------------------------------------------------------------- #
def frame_from_name(path):
    nums = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    return int(nums[-1]) if nums else None


def process_one(img_path, boxes_for_frame, args):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[warn] cannot read {img_path}")
        return None
    h_img = img.shape[0]
    assigned = assign_depth_layers(
        boxes_for_frame, h_img, args.levels, near_first=args.near_first
    )
    draw_layers(
        img,
        assigned,
        args.levels,
        thickness=args.thickness,
        show_id=args.show_id,
        legend=not args.no_legend,
    )
    return img


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Visualize SparseTrack pseudo-depth stratification: "
        "color each box by its depth layer."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--img", help="Single image path.")
    src.add_argument("--imgs", help="Directory of sequence images.")

    ap.add_argument("--txt", required=True, help="MOTChallenge-format det/track file.")
    ap.add_argument(
        "--levels",
        type=int,
        default=3,
        help="K = number of depth subsets. Default 3 (MOT17). " "MOT20 test uses 8.",
    )
    ap.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame index to use (single-image mode). Overrides "
        "the number parsed from the filename.",
    )
    ap.add_argument(
        "--out",
        default="pseudo_depth_layers.jpg",
        help="Output path (single-image mode).",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (sequence mode). " "Default: <imgs>_layers",
    )
    ap.add_argument(
        "--ext", default=".jpg", help="Image extension for --imgs. Default .jpg"
    )
    ap.add_argument(
        "--thickness", type=int, default=2, help="Box line thickness. Default 2."
    )
    ap.add_argument(
        "--near-first",
        action="store_true",
        help="Make layer 0 the NEAREST subset (largest L_p). "
        "Default: layer 0 is the farthest.",
    )
    ap.add_argument(
        "--show-id",
        action="store_true",
        help="Also print each box's track ID next to its layer.",
    )
    ap.add_argument(
        "--no-legend", action="store_true", help="Do not draw the legend box."
    )
    args = ap.parse_args()

    frames = load_boxes(args.txt)

    # ---- Single image ---- #
    if args.img:
        if not os.path.isfile(args.img):
            sys.exit(f"[error] image not found: {args.img}")
        fidx = args.frame if args.frame is not None else frame_from_name(args.img)
        if fidx is None:
            fidx = 1
        boxes = frames.get(fidx, [])
        if not boxes:
            print(f"[warn] no boxes for frame {fidx}; drawing image as-is.")
        img = process_one(args.img, boxes, args)
        if img is None:
            sys.exit(1)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        cv2.imwrite(args.out, img)
        print(
            f"[done] frame {fidx}: {len(boxes)} boxes over {args.levels} "
            f"layers -> {args.out}"
        )
        return

    # ---- Sequence ---- #
    if not os.path.isdir(args.imgs):
        sys.exit(f"[error] not a directory: {args.imgs}")
    ext = args.ext if args.ext.startswith(".") else "." + args.ext
    paths = sorted(
        glob.glob(os.path.join(args.imgs, "*" + ext)),
        key=lambda p: (frame_from_name(p) if frame_from_name(p) is not None else 0, p),
    )
    if not paths:
        sys.exit(f"[error] no '{ext}' images in {args.imgs}")

    outdir = args.outdir or (args.imgs.rstrip("/\\") + "_layers")
    os.makedirs(outdir, exist_ok=True)

    n = 0
    for p in paths:
        fidx = frame_from_name(p)
        if fidx is None:
            continue
        boxes = frames.get(fidx, [])
        img = process_one(p, boxes, args)
        if img is None:
            continue
        outp = os.path.join(
            outdir, os.path.splitext(os.path.basename(p))[0] + "_layers.png"
        )
        cv2.imwrite(outp, img)
        n += 1
    print(
        f"[done] wrote {n} annotated frames to {outdir}/ "
        f"({args.levels} layers each)."
    )


if __name__ == "__main__":
    main()

"""
python pseudo_depth_layers.py \
    --img /home/caig/data/MOT17/train/MOT17-02-FRCNN/img1/000295.jpg \
    --txt /home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results_dti/MOT17-02-FRCNN.txt \
    --levels 3 \
    --out layered_295.jpg
"""
