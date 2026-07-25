#!/usr/bin/env python3
"""
visualize_tracks.py

Read a sequence of images and a MOTChallenge-format tracking result file,
draw each bounding box with its track ID (one distinct color per ID), and
export an annotated video.

Tracking file format (MOTChallenge, comma- or space-separated):
    frame, id, x, y, w, h, conf, ...
where (x, y) is the top-left corner and (w, h) the box size, all in pixels.
Extra columns after h are ignored.

Usage
-----
Basic:
    python visualize_tracks.py --imgs seq/img1 --txt results.txt --out out.mp4

Common options:
    python visualize_tracks.py \
        --imgs seq/img1 \
        --txt results.txt \
        --out out.mp4 \
        --fps 30 \
        --ext .jpg

If your images are named like 000001.jpg, 000002.jpg (MOTChallenge default),
the defaults just work. If frame N does not map to that pattern, use
--name-format, e.g. --name-format "frame_{:d}.png".
"""

import argparse
import os
import sys
import glob
import re
import colorsys

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# Color handling: one stable, visually distinct color per track ID.
# --------------------------------------------------------------------------- #
def color_for_id(track_id: int):
    """
    Deterministically map an integer ID to a bright, distinct BGR color.

    Uses the golden-ratio hue spacing trick so that consecutive IDs land far
    apart on the color wheel, and the same ID always gets the same color.
    """
    golden_ratio_conjugate = 0.618033988749895
    hue = (track_id * golden_ratio_conjugate) % 1.0
    # High saturation and value -> vivid, readable colors.
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    # OpenCV uses BGR order.
    return (int(b * 255), int(g * 255), int(r * 255))


# --------------------------------------------------------------------------- #
# Load tracking results.
# --------------------------------------------------------------------------- #
def load_tracks(txt_path: str):
    """
    Parse a MOTChallenge-format result file.

    Returns
    -------
    dict[int, list[tuple]]
        Maps frame index -> list of (track_id, x, y, w, h) tuples.
    """
    frames = {}
    n_lines = 0
    with open(txt_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Accept comma- or whitespace-separated values.
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 6:
                continue
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except ValueError:
                # Skip header lines or malformed rows.
                continue
            frames.setdefault(frame, []).append((tid, x, y, w, h))
            n_lines += 1
    if n_lines == 0:
        sys.exit(f"[error] No valid tracking rows parsed from {txt_path}")
    print(
        f"[info] Loaded {n_lines} boxes across {len(frames)} frames " f"from {txt_path}"
    )
    return frames


# --------------------------------------------------------------------------- #
# Load image list.
# --------------------------------------------------------------------------- #
def load_image_paths(img_dir: str, ext: str):
    """
    Collect and naturally sort image paths in img_dir matching the extension.

    Returns a list of absolute paths sorted by the trailing number in the
    filename when available, otherwise lexicographically.
    """
    if not os.path.isdir(img_dir):
        sys.exit(f"[error] Image directory not found: {img_dir}")

    ext = ext if ext.startswith(".") else "." + ext
    paths = glob.glob(os.path.join(img_dir, "*" + ext))
    if not paths:
        # Try a case-insensitive / any-extension fallback.
        paths = [
            p
            for p in glob.glob(os.path.join(img_dir, "*"))
            if p.lower().endswith(ext.lower())
        ]
    if not paths:
        sys.exit(f"[error] No '{ext}' images found in {img_dir}")

    def sort_key(p):
        name = os.path.splitext(os.path.basename(p))[0]
        nums = re.findall(r"\d+", name)
        return (int(nums[-1]) if nums else 0, name)

    paths.sort(key=sort_key)
    print(f"[info] Found {len(paths)} images in {img_dir}")
    return paths


def frame_index_from_path(path: str):
    """Extract the trailing integer in a filename as its frame index."""
    name = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


# --------------------------------------------------------------------------- #
# Drawing.
# --------------------------------------------------------------------------- #
def draw_boxes(img, boxes, thickness=2, font_scale=0.6):
    """Draw all (track_id, x, y, w, h) boxes for one frame onto img in place."""
    for tid, x, y, w, h in boxes:
        color = color_for_id(tid)
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))

        # Bounding box.
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # ID label with a filled background for readability.
        label = str(tid)
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        # Place label just above the top-left corner; clamp to image top.
        ty1 = max(0, y1 - th - baseline - 4)
        ty2 = ty1 + th + baseline + 4
        cv2.rectangle(img, (x1, ty1), (x1 + tw + 6, ty2), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 3, ty2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Overlay MOT tracking boxes + IDs on a sequence and "
        "export a video."
    )
    ap.add_argument(
        "--imgs", required=True, help="Directory of sequence images (e.g. seq/img1)."
    )
    ap.add_argument(
        "--txt", required=True, help="MOTChallenge-format tracking result file."
    )
    ap.add_argument(
        "--out",
        default="output.mp4",
        help="Output video path (.mp4 or .avi). Default: output.mp4",
    )
    ap.add_argument(
        "--fps", type=float, default=30.0, help="Output video frame rate. Default: 30"
    )
    ap.add_argument("--ext", default=".jpg", help="Image file extension. Default: .jpg")
    ap.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Bounding-box line thickness. Default: 2",
    )
    ap.add_argument(
        "--font-scale",
        type=float,
        default=0.6,
        help="ID label font scale. Default: 0.6",
    )
    ap.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Frame index the FIRST image corresponds to. "
        "By default the trailing number in each filename is "
        "used to look up boxes. Set this if your images are "
        "0-based or offset from the txt frame numbering.",
    )
    args = ap.parse_args()

    frames = load_tracks(args.txt)
    img_paths = load_image_paths(args.imgs, args.ext)

    # Probe the first image for video dimensions.
    first = cv2.imread(img_paths[0])
    if first is None:
        sys.exit(f"[error] Failed to read first image: {img_paths[0]}")
    height, width = first.shape[:2]

    # Choose codec from extension.
    ext = os.path.splitext(args.out)[1].lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"[error] Could not open video writer for {args.out}")

    n_written = 0
    n_missing = 0
    for i, path in enumerate(img_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"[warn] Skipping unreadable image: {path}")
            continue
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))

        # Determine which frame index in the txt this image maps to.
        if args.start_frame is not None:
            frame_idx = args.start_frame + i
        else:
            fi = frame_index_from_path(path)
            frame_idx = fi if fi is not None else (i + 1)

        boxes = frames.get(frame_idx, [])
        if not boxes:
            n_missing += 1
        draw_boxes(img, boxes, thickness=args.thickness, font_scale=args.font_scale)

        writer.write(img)
        n_written += 1

    writer.release()
    print(
        f"[done] Wrote {n_written} frames to {args.out} "
        f"({args.fps:g} fps, {width}x{height})."
    )
    if n_missing:
        print(
            f"[note] {n_missing} image(s) had no matching boxes. "
            f"If that seems wrong, check frame-number alignment "
            f"(try --start-frame)."
        )


if __name__ == "__main__":
    main()

"""
python visualize_tracks.py \
    --imgs /home/caig/data/MOT17/train/MOT17-09-FRCNN/img1 \
    --txt /home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results_dti/MOT17-09-FRCNN.txt \
    --out out.mp4 \
    --fps 30
"""
