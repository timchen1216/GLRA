"""
visualize_glra_rescues.py

Visualize cases where GLRA successfully recovered lost tracks on MOT17.

For each rescue event, produces a horizontal strip of images showing:
  - PRE  : last N frames where the track was present in both baseline and GLRA  (green bbox)
  - GAP  : up to N frames where the track was "lost" — show GT bbox (yellow, dashed)
  - POST : first N frames after GLRA rescue — track bbox back in GLRA but not baseline (blue bbox)

Correctness is judged by GT-IoU: the GT identity before and after the gap must match.

Usage:
    python visualize_glra_rescues.py
    python visualize_glra_rescues.py --seqs MOT17-02-FRCNN MOT17-05-FRCNN
    python visualize_glra_rescues.py --min_gap 1 --max_gap 8 --max_cases 20 --out_dir vis_glra
"""

import argparse
import os
import sys
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
GT_ROOT       = Path("/home/caig/data/MOT17/train")
TRACK_BASE    = ROOT / "yolox_mix17_ablation/yolox_mix17_ablation_det/track_results"
TRACK_GLRA    = ROOT / "yolox_mix17_ablation/yolox_mix17_ablation_det/track_results_glra"
ALL_SEQS      = [
    "MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
    "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN",
]

# ── visual constants ───────────────────────────────────────────────────────────
C_PRE    = (60, 200, 60)    # green  — track before gap
C_GAP_GT = (0, 200, 255)    # yellow — GT during gap (track was lost)
C_POST   = (255, 100, 0)    # blue   — GLRA rescue frames
C_GT     = (0, 180, 255)    # light-blue — GT overlay in PRE/POST frames
FONT     = cv2.FONT_HERSHEY_SIMPLEX
PAD_RATIO = 2.5             # context crop: bbox side × this value
STRIP_H   = 320             # height of each frame thumbnail in the strip
CONTEXT   = 4               # frames shown before gap and after rescue
MAX_GAP_SHOW = 4            # max gap frames shown (trimmed if longer)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_tracks(path: Path) -> dict[int, dict[int, tuple]]:
    """Return {tid: {frame: (x,y,w,h)}} with 1-indexed frames."""
    tracks: dict[int, dict[int, tuple]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid  = int(float(parts[0]))
            tid  = int(float(parts[1]))
            xywh = tuple(float(v) for v in parts[2:6])
            tracks[tid][fid] = xywh
    return tracks


def load_gt(path: Path) -> dict[int, dict[int, tuple]]:
    """Return {gt_id: {frame: (x,y,w,h)}} for class=1 (pedestrian), any visibility."""
    gt: dict[int, dict[int, tuple]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            fid  = int(parts[0])
            gid  = int(parts[1])
            xywh = tuple(float(v) for v in parts[2:6])
            cls  = int(parts[7])
            if cls != 1:
                continue
            gt[gid][fid] = xywh
    return gt


def xywh_to_tlbr(x, y, w, h):
    return x, y, x + w, y + h


def iou_xywh(a, b):
    """IoU between two (x,y,w,h) boxes."""
    ax1, ay1, ax2, ay2 = xywh_to_tlbr(*a)
    bx1, by1, bx2, by2 = xywh_to_tlbr(*b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def best_gt_match(xywh, gt_frame: dict[int, tuple], min_iou=0.2):
    """Return (gt_id, iou) of best GT match in a single frame, or (None, 0)."""
    best_gid, best_iou = None, min_iou - 1e-9
    for gid, gxywh in gt_frame.items():
        v = iou_xywh(xywh, gxywh)
        if v > best_iou:
            best_iou, best_gid = v, gid
    return best_gid, best_iou


# ── rescue detection ──────────────────────────────────────────────────────────

def build_base_gt_coverage(base_tracks, gt_by_frame):
    """Return {frame: {gt_id: base_tid}} — which baseline track covers each GT ID per frame."""
    coverage: dict[int, dict[int, int]] = defaultdict(dict)
    for b_tid, b_fdict in base_tracks.items():
        for fid, b_xywh in b_fdict.items():
            for gid, gxywh in gt_by_frame.get(fid, {}).items():
                if iou_xywh(b_xywh, gxywh) >= 0.2:
                    if gid not in coverage[fid]:
                        coverage[fid][gid] = b_tid
    return coverage


def find_rescue_events(base_tracks, glra_tracks, gt_by_id,
                       min_gap=0, max_gap=8, min_rescued=3):
    """
    Find GLRA rescue events for one sequence.

    A rescue event is defined as:
      - Track ID T exists in GLRA for >= min_rescued frames starting at R
      - T does NOT exist in baseline at frame R
      - T was last seen in both baseline and GLRA at frame L
      - gap_len = R - L - 1 (0 = GLRA extended via low-score det; ≥1 = track re-found after gap)
      - max_gap applies only when gap_len >= 1 (un-gapped extensions always included)
      - The GT identity before and after is checked for correctness

    Returns list of dicts with rescue metadata.
    """
    # Build per-frame GT lookup: {frame: {gt_id: xywh}}
    gt_by_frame: dict[int, dict[int, tuple]] = defaultdict(dict)
    for gid, fdict in gt_by_id.items():
        for fid, xywh in fdict.items():
            gt_by_frame[fid][gid] = xywh

    base_gt_cov = build_base_gt_coverage(base_tracks, gt_by_frame)

    common_ids = set(base_tracks) & set(glra_tracks)
    events = []

    for tid in sorted(common_ids):
        g_frames = sorted(glra_tracks[tid])
        b_frames = set(base_tracks[tid])

        # Find frames in GLRA but not baseline (candidate rescue frames)
        rescued_frames = [f for f in g_frames if f not in b_frames]
        if not rescued_frames:
            continue

        # Group rescued frames into contiguous blocks
        blocks: list[list[int]] = []
        cur = [rescued_frames[0]]
        for f in rescued_frames[1:]:
            if f == cur[-1] + 1:
                cur.append(f)
            else:
                blocks.append(cur)
                cur = [f]
        blocks.append(cur)

        for block in blocks:
            if len(block) < min_rescued:
                continue
            recovery_frame = block[0]

            # Last frame where T was in BOTH baseline AND GLRA before this block
            pre_common = [f for f in g_frames if f < recovery_frame and f in b_frames]
            if not pre_common:
                continue
            last_seen = max(pre_common)
            gap_len = recovery_frame - last_seen - 1

            if gap_len > max_gap:
                continue
            if gap_len < min_gap:
                continue

            # GT identity check: must match the same person before and after gap
            pre_xywh  = glra_tracks[tid].get(last_seen)
            post_xywh = glra_tracks[tid].get(recovery_frame)
            if pre_xywh is None or post_xywh is None:
                continue

            pre_gt_id, pre_iou   = best_gt_match(pre_xywh,  gt_by_frame.get(last_seen, {}))
            post_gt_id, post_iou = best_gt_match(post_xywh, gt_by_frame.get(recovery_frame, {}))

            is_correct = (pre_gt_id is not None and pre_gt_id == post_gt_id)

            events.append({
                "tid":            tid,
                "last_seen":      last_seen,
                "recovery_frame": recovery_frame,
                "gap_len":        gap_len,
                "n_rescued":      len(block),
                "rescued_block":  block,
                "pre_gt_id":      pre_gt_id,
                "post_gt_id":     post_gt_id,
                "pre_iou":        pre_iou,
                "post_iou":       post_iou,
                "is_correct":     is_correct,
                "gt_by_frame":    gt_by_frame,
                "base_gt_cov":    base_gt_cov,   # {frame: {gt_id: base_tid}}
                "base_tracks":    base_tracks,
            })

    return events


# ── image helpers ─────────────────────────────────────────────────────────────

def crop_around_bbox(img, x, y, w, h, pad_ratio=PAD_RATIO, target_h=STRIP_H):
    """Return a square crop centred on (x,y,w,h), resized to target_h × target_h."""
    cx, cy = x + w / 2, y + h / 2
    side   = max(w, h) * pad_ratio / 2
    x1 = max(0, int(cx - side))
    y1 = max(0, int(cy - side))
    x2 = min(img.shape[1], int(cx + side))
    y2 = min(img.shape[0], int(cy + side))
    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = img.copy()
    crop = cv2.resize(crop, (target_h, target_h))
    return crop, (x1, y1, x2 - x1, y2 - y1)  # crop_region


def draw_bbox_on_crop(crop, x, y, w, h, crop_region, color, label="",
                      thickness=2, dashed=False):
    """Draw a bbox (original coords) onto a crop, scaling to crop size."""
    crx, cry, crw, crh = crop_region
    H, W = crop.shape[:2]
    sx = W / crw
    sy = H / crh
    px1 = int((x - crx) * sx)
    py1 = int((y - cry) * sy)
    px2 = int((x + w - crx) * sx)
    py2 = int((y + h - cry) * sy)
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(W - 1, px2), min(H - 1, py2)

    if dashed:
        _draw_dashed_rect(crop, px1, py1, px2, py2, color, thickness)
    else:
        cv2.rectangle(crop, (px1, py1), (px2, py2), color, thickness)

    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        lx = max(px1, 0)
        ly = max(py1 - 4, th + 4)
        cv2.rectangle(crop, (lx, ly - th - 2), (lx + tw + 2, ly + 2), color, -1)
        cv2.putText(crop, label, (lx + 1, ly), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=2, dash=8):
    """Draw a dashed rectangle."""
    for pts in [[(x1, y1), (x2, y1)], [(x2, y1), (x2, y2)],
                [(x2, y2), (x1, y2)], [(x1, y2), (x1, y1)]]:
        _draw_dashed_line(img, pts[0], pts[1], color, thickness, dash)


def _draw_dashed_line(img, p1, p2, color, thickness, dash):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    steps = int(length / (dash * 2)) + 1
    for i in range(steps):
        t1 = (2 * i * dash) / length
        t2 = min((2 * i + 1) * dash / length, 1.0)
        a = (int(p1[0] + dx * t1), int(p1[1] + dy * t1))
        b = (int(p1[0] + dx * t2), int(p1[1] + dy * t2))
        cv2.line(img, a, b, color, thickness)


def add_label_bar(crop, text, color, bar_h=24):
    """Add a colored label bar below the crop."""
    bar = np.zeros((bar_h, crop.shape[1], 3), dtype=np.uint8)
    bar[:] = color
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
    tx = max(0, (bar.shape[1] - tw) // 2)
    ty = (bar_h + th) // 2
    cv2.putText(bar, text, (tx, ty), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([crop, bar])


def make_separator(h, w=6):
    sep = np.zeros((h, w, 3), dtype=np.uint8)
    sep[:] = (180, 180, 180)
    return sep


def make_section_divider(h, w=3, total_w=None):
    """Thicker divider between PRE/GAP/POST sections."""
    div = np.zeros((h, w if total_w is None else 4, 3), dtype=np.uint8)
    div[:] = (80, 80, 80)
    return div


# ── per-event visualisation ───────────────────────────────────────────────────

def make_rescue_strip(event, seq_name, img_dir, glra_tracks,
                      context=CONTEXT, max_gap_show=MAX_GAP_SHOW,
                      strip_h=STRIP_H):
    """
    Build and return a horizontal image strip for one rescue event.
    Returns None if images cannot be loaded.
    """
    tid            = event["tid"]
    last_seen      = event["last_seen"]
    recovery_frame = event["recovery_frame"]
    gap_len        = event["gap_len"]
    gt_by_frame    = event["gt_by_frame"]
    pre_gt_id      = event["pre_gt_id"]
    is_correct     = event["is_correct"]

    # Reference bbox for centering the crop (use last-seen bbox)
    ref_xywh = glra_tracks[tid].get(last_seen)
    if ref_xywh is None:
        return None

    def load_img(fid):
        p = img_dir / f"{fid:06d}.jpg"
        if not p.exists():
            return None
        return cv2.imread(str(p))

    strip_frames = []

    # ── PRE frames ────────────────────────────────────────────────────────────
    pre_start = max(1, last_seen - context + 1)
    pre_fids  = list(range(pre_start, last_seen + 1))
    for fid in pre_fids:
        img = load_img(fid)
        if img is None:
            continue
        xywh = glra_tracks[tid].get(fid, ref_xywh)
        crop, cr = crop_around_bbox(img, *xywh, target_h=strip_h)
        # Draw GT in light blue (background)
        if pre_gt_id and pre_gt_id in gt_by_frame.get(fid, {}):
            draw_bbox_on_crop(crop, *gt_by_frame[fid][pre_gt_id], cr,
                              C_GT, thickness=1)
        # Draw track in green
        draw_bbox_on_crop(crop, *xywh, cr, C_PRE,
                          label=f"GLRA:{tid}", thickness=2)
        lbl = f"F{fid} PRE"
        crop = add_label_bar(crop, lbl, (40, 130, 40))
        strip_frames.append(("pre", crop))

    # ── GAP frames (only when gap_len >= 1) ──────────────────────────────────
    gap_fids = list(range(last_seen + 1, recovery_frame))
    if gap_fids:
        show_gap = gap_fids[:max_gap_show]
        for fid in show_gap:
            img = load_img(fid)
            if img is None:
                continue
            gt_xywh = None
            if pre_gt_id and pre_gt_id in gt_by_frame.get(fid, {}):
                gt_xywh = gt_by_frame[fid][pre_gt_id]
            anchor = gt_xywh if gt_xywh else ref_xywh
            crop, cr = crop_around_bbox(img, *anchor, target_h=strip_h)
            if gt_xywh:
                draw_bbox_on_crop(crop, *gt_xywh, cr, C_GAP_GT,
                                  label="GT (lost)", dashed=True, thickness=2)
            cv2.putText(crop, "TRACK LOST", (4, 22), FONT, 0.55,
                        (0, 0, 200), 2, cv2.LINE_AA)
            lbl = f"F{fid} GAP"
            crop = add_label_bar(crop, lbl, (0, 100, 160))
            strip_frames.append(("gap", crop))

        if len(gap_fids) > max_gap_show:
            placeholder = np.zeros((strip_h, strip_h, 3), dtype=np.uint8)
            placeholder[:] = (40, 40, 40)
            n_skip = len(gap_fids) - max_gap_show
            cv2.putText(placeholder, f"... +{n_skip}", (strip_h // 2 - 30, strip_h // 2),
                        FONT, 0.8, (200, 200, 200), 2)
            placeholder = add_label_bar(placeholder, f"{n_skip} more gap", (40, 40, 40))
            strip_frames.append(("gap", placeholder))

    # ── POST frames (GLRA rescue) ─────────────────────────────────────────────
    base_gt_cov  = event.get("base_gt_cov", {})
    base_tracks_ = event.get("base_tracks", {})
    C_BASE_NEW = (40, 40, 200)   # red — baseline new ID for same person

    post_fids = sorted(f for f in glra_tracks[tid] if f >= recovery_frame)
    post_fids = post_fids[:context]
    for i, fid in enumerate(post_fids):
        img = load_img(fid)
        if img is None:
            continue
        xywh = glra_tracks[tid][fid]
        crop, cr = crop_around_bbox(img, *xywh, target_h=strip_h)
        # GT bbox (light blue, thin)
        if pre_gt_id and pre_gt_id in gt_by_frame.get(fid, {}):
            draw_bbox_on_crop(crop, *gt_by_frame[fid][pre_gt_id], cr,
                              C_GT, thickness=1)
        # Baseline's assignment for this GT person (dashed red if different ID)
        base_tid_here = base_gt_cov.get(fid, {}).get(pre_gt_id)
        if base_tid_here is not None and base_tid_here != tid:
            base_xywh = base_tracks_.get(base_tid_here, {}).get(fid)
            if base_xywh:
                draw_bbox_on_crop(crop, *base_xywh, cr, C_BASE_NEW,
                                  label=f"BL:{base_tid_here}", dashed=True, thickness=2)
        # GLRA track (solid orange)
        draw_bbox_on_crop(crop, *xywh, cr, C_POST,
                          label=f"GLRA:{tid}", thickness=2)
        if i == 0:
            cv2.putText(crop, "RESCUED", (4, 22), FONT, 0.6,
                        (0, 100, 255), 2, cv2.LINE_AA)
        lbl = f"F{fid} POST"
        crop = add_label_bar(crop, lbl, (140, 60, 0))
        strip_frames.append(("post", crop))

    if not strip_frames:
        return None

    # ── assemble strip ────────────────────────────────────────────────────────
    total_h = strip_frames[0][1].shape[0]
    pieces = []
    prev_phase = None
    for phase, tile in strip_frames:
        if prev_phase is not None:
            if phase != prev_phase:
                pieces.append(make_section_divider(total_h))
            else:
                pieces.append(make_separator(total_h, 3))
        pieces.append(tile)
        prev_phase = phase

    strip = np.hstack(pieces)

    # ── header bar ────────────────────────────────────────────────────────────
    correct_tag = "CORRECT" if is_correct else "WRONG-ID"
    correct_col = (30, 120, 30) if is_correct else (20, 20, 180)
    header_h = 32
    header = np.zeros((header_h, strip.shape[1], 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    event_type = f"re-found after {gap_len}fr gap" if gap_len > 0 else "extended via low-score det"
    title = (f"{seq_name}  |  Track ID={tid}  |  {event_type}  |"
             f"  rescued {event['n_rescued']}fr  |  {correct_tag}")
    (tw, th), _ = cv2.getTextSize(title, FONT, 0.52, 1)
    cv2.putText(header, title, (8, (header_h + th) // 2), FONT, 0.52,
                (220, 220, 220), 1, cv2.LINE_AA)
    # Colour badge for correct/wrong
    badge_w = 90
    header[:, -badge_w:] = correct_col
    cv2.putText(header, correct_tag, (strip.shape[1] - badge_w + 4,
                (header_h + th) // 2),
                FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([header, strip])


# ── legend ────────────────────────────────────────────────────────────────────

def make_legend(strip_w):
    """Return a legend image of width strip_w."""
    h = 28
    legend = np.zeros((h, max(strip_w, 600), 3), np.uint8)
    legend[:] = (20, 20, 20)
    C_BASE_NEW = (40, 40, 200)
    items = [
        (C_PRE,      "PRE: both systems tracking (solid green)"),
        (C_POST,     "POST: GLRA rescued — same ID continues (solid orange)"),
        (C_BASE_NEW, "baseline new/different ID for same person (dashed red)"),
        (C_GT,       "GT bbox (thin light-blue)"),
        (C_GAP_GT,   "GAP frame: GT only, track lost (dashed yellow)"),
    ]
    x = 6
    for col, txt in items:
        cv2.rectangle(legend, (x, 7), (x + 14, h - 7), col, -1)
        x += 18
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.38, 1)
        cv2.putText(legend, txt, (x, (h + th) // 2), FONT, 0.38,
                    (210, 210, 210), 1, cv2.LINE_AA)
        x += tw + 20
    return legend


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise GLRA rescue events on MOT17")
    parser.add_argument("--seqs",       nargs="+", default=ALL_SEQS,
                        help="Sequence names to process")
    parser.add_argument("--out_dir",    default="vis_glra_rescues",
                        help="Output directory for visualisations")
    parser.add_argument("--min_gap",    type=int, default=0,
                        help="Minimum gap length in frames (0 = include low-score extensions)")
    parser.add_argument("--max_gap",    type=int, default=8,
                        help="Maximum gap length (frames)")
    parser.add_argument("--min_rescued", type=int, default=3,
                        help="Minimum rescued frames for event to qualify")
    parser.add_argument("--max_cases",  type=int, default=30,
                        help="Max cases to visualise (sorted by gap len × rescued)")
    parser.add_argument("--correct_only", action="store_true",
                        help="Only show GT-correct rescues")
    parser.add_argument("--context",    type=int, default=CONTEXT,
                        help="Pre/post context frames shown per event")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_events = []

    for seq in args.seqs:
        base_path = TRACK_BASE / f"{seq}.txt"
        glra_path = TRACK_GLRA / f"{seq}.txt"
        gt_path   = GT_ROOT / seq / "gt" / "gt_val_half.txt"
        img_dir   = GT_ROOT / seq / "img1"

        if not glra_path.exists():
            print(f"[SKIP] {seq}: no GLRA track file at {glra_path}")
            continue
        if not base_path.exists():
            print(f"[SKIP] {seq}: no baseline track file at {base_path}")
            continue
        if not gt_path.exists():
            print(f"[SKIP] {seq}: no GT at {gt_path}")
            continue

        print(f"\n[{seq}] loading tracks & GT ...")
        base_tracks = load_tracks(base_path)
        glra_tracks = load_tracks(glra_path)
        gt_by_id    = load_gt(gt_path)

        events = find_rescue_events(
            base_tracks, glra_tracks, gt_by_id,
            min_gap=args.min_gap, max_gap=args.max_gap,
            min_rescued=args.min_rescued,
        )
        print(f"  found {len(events)} rescue events  "
              f"({sum(e['is_correct'] for e in events)} correct, "
              f"{sum(not e['is_correct'] for e in events)} wrong-ID)")

        for e in events:
            e["seq"]        = seq
            e["img_dir"]    = img_dir
            e["glra_tracks"] = glra_tracks
            e["sort_key"]   = (e["gap_len"] > 0) * 1000 + e["gap_len"] * 10 + e["n_rescued"]
        all_events.extend(events)

    if not all_events:
        print("\nNo rescue events found. Try loosening --min_gap or --min_rescued.")
        return

    # Filter and sort
    if args.correct_only:
        all_events = [e for e in all_events if e["is_correct"]]
    all_events.sort(key=lambda e: e["sort_key"], reverse=True)

    print(f"\nTotal events: {len(all_events)}, visualising top {args.max_cases}")

    # ── Generate images ───────────────────────────────────────────────────────
    all_strips = []
    count = 0
    for e in all_events:
        if count >= args.max_cases:
            break
        strip = make_rescue_strip(
            e, e["seq"], e["img_dir"], e["glra_tracks"],
            context=args.context,
        )
        if strip is None:
            continue
        # Save individual strip
        fname = (f"{e['seq']}_tid{e['tid']}_"
                 f"f{e['last_seen']}-{e['recovery_frame']}"
                 f"_gap{e['gap_len']}"
                 f"_{'ok' if e['is_correct'] else 'wrong'}.jpg")
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  saved {fname}")
        all_strips.append(strip)
        count += 1

    # ── Compose summary sheet ─────────────────────────────────────────────────
    if all_strips:
        max_w = max(s.shape[1] for s in all_strips)
        # Pad each strip to max_w
        padded = []
        for s in all_strips:
            if s.shape[1] < max_w:
                pad = np.zeros((s.shape[0], max_w - s.shape[1], 3), np.uint8)
                s = np.hstack([s, pad])
            padded.append(s)

        legend  = make_legend(max_w)
        divider = np.zeros((6, max_w, 3), np.uint8)
        divider[:] = (60, 60, 60)

        sheet_rows = []
        for i, s in enumerate(padded):
            sheet_rows.append(s)
            if i < len(padded) - 1:
                sheet_rows.append(divider)
        sheet_rows.insert(0, legend)

        sheet = np.vstack(sheet_rows)
        sheet_path = out_dir / "summary_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\nSummary sheet saved to {sheet_path}")
        print(f"Individual frames saved to {out_dir}/")


if __name__ == "__main__":
    main()
