"""
Offline analysis of GLRA diagnostic CSV against MOT ground truth.

For each GLRA recovery record produced by sparse_tracker.py (when
`--glra-diag` is on), determine whether the recovery was CORRECT
(track's true identity at re-association == detection's true identity)
or WRONG (identity swap — the GLRA FP we want to characterize).

Outputs ONE summary CSV with per-sequence rows + a final ALL row, and
ONE per-record CSV with classification appended.  Per-feature medians
(correct vs wrong) appear as additional columns so you can spot the
failure-mode signature at a glance.

Usage:
    python analyze_glra_diag.py \\
        --diag-csv     ./glra_diag.csv \\
        --gt-folder    /home/caig/data/DanceTrack/val \\
        --tracker-out  /home/caig/repo/SparseTrack/yolox_dance_sparse/yolox_dance_sparse_det/track_results_val \\
        --seqmap       /home/caig/data/DanceTrack/val_seqmap.txt \\
        --out          ./glra_diag_analysis.csv \\
        --summary      ./glra_diag_summary.csv

    python3 analyze_glra_diag.py \
    --diag-csv     ./glra_diag_mot20.csv \
    --gt-folder    /home/caig/data/MOT20/train \
    --tracker-out  /home/caig/repo/SparseTrack/yolox_mix20/yolox_mix20_det/track_results \
    --seqmap       /home/caig/data/MOT20/20train_seqmap.txt \
    --out          ./glra_diag_analysis.csv \
    --summary      ./glra_diag_summary.csv
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# ── MOT-format I/O ──────────────────────────────────────────────────────────


def load_mot_file(path):
    """Load an MOTChallenge-format file (GT or tracker output).

    Returns:
        dict[int frame -> list of (id, x1, y1, x2, y2)]
    """
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, header=None)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return {}
    by_frame = defaultdict(list)
    # MOTChallenge: frame, id, x, y, w, h, conf/visibility, class, ...
    for row in df.itertuples(index=False, name=None):
        frame = int(row[0])
        tid = int(row[1])
        x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        by_frame[frame].append((tid, x, y, x + w, y + h))
    return dict(by_frame)


def build_track_history(tracker_data):
    """Index tracker output by track id for fast last-frame-before lookup."""
    hist = defaultdict(list)
    for frame, rows in tracker_data.items():
        for tid, x1, y1, x2, y2 in rows:
            hist[tid].append((frame, (x1, y1, x2, y2)))
    for tid in hist:
        hist[tid].sort()
    return dict(hist)


def last_frame_before(track_hist, track_id, frame_cutoff):
    """Return (frame, box) for the last appearance of track_id strictly
    before frame_cutoff, or (None, None) if none."""
    rows = track_hist.get(int(track_id), [])
    last = None
    for f, b in rows:
        if f < frame_cutoff:
            last = (f, b)
        else:
            break
    return last if last else (None, None)


# ── IoU helpers ─────────────────────────────────────────────────────────────


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def best_gt_match(box, frame_gt):
    """Highest-IoU GT id at this frame for the given box.

    Returns (gt_id, iou) or (None, 0.0).
    """
    best_id, best_iou = None, 0.0
    for tid, x1, y1, x2, y2 in frame_gt:
        i = iou_xyxy(box, (x1, y1, x2, y2))
        if i > best_iou:
            best_iou = i
            best_id = tid
    return best_id, best_iou


# ── Classification ──────────────────────────────────────────────────────────


def classify_record(rec, gt_data, track_hist, iou_thresh=0.3):
    """Determine if a GLRA recovery was correct.

    Returns dict with: is_correct (True/False/None), gt_id_det, gt_id_track,
                       iou_det, iou_track, reason
    """
    frame = int(rec["frame"])
    track_id = int(rec["track_id"])
    det_box = (
        rec["det_cx"] - rec["det_w"] / 2.0,
        rec["det_cy"] - rec["det_h"] / 2.0,
        rec["det_cx"] + rec["det_w"] / 2.0,
        rec["det_cy"] + rec["det_h"] / 2.0,
    )

    # Step 1: which GT id is the detection?
    gt_id_det, iou_det = best_gt_match(det_box, gt_data.get(frame, []))
    if gt_id_det is None or iou_det < iou_thresh:
        return {
            "is_correct": None,
            "gt_id_det": gt_id_det,
            "gt_id_track": None,
            "iou_det": iou_det,
            "iou_track": 0.0,
            "reason": "no_gt_at_det",
        }

    # Step 2: what is the track's true identity (its GT id just before being lost)?
    last_frame, last_box = last_frame_before(track_hist, track_id, frame)
    if last_box is None:
        return {
            "is_correct": None,
            "gt_id_det": gt_id_det,
            "gt_id_track": None,
            "iou_det": iou_det,
            "iou_track": 0.0,
            "reason": "no_prior_track",
        }
    gt_id_track, iou_track = best_gt_match(last_box, gt_data.get(last_frame, []))
    if gt_id_track is None or iou_track < iou_thresh:
        return {
            "is_correct": None,
            "gt_id_det": gt_id_det,
            "gt_id_track": gt_id_track,
            "iou_det": iou_det,
            "iou_track": iou_track,
            "reason": "no_gt_at_track",
        }

    return {
        "is_correct": (gt_id_track == gt_id_det),
        "gt_id_det": gt_id_det,
        "gt_id_track": gt_id_track,
        "iou_det": iou_det,
        "iou_track": iou_track,
        "reason": "ok",
    }


# ── Summary ─────────────────────────────────────────────────────────────────

SUMMARY_FEATURES = [
    "lost_duration",
    "gpr_sigma_px",
    "cost",
    "motion_ratio",
    "implied_speed",
    "nearest_active_iou",
    "nearest_active_dist",
    "n_neighbors",
]


def _agg(sub):
    """Aggregate one DataFrame slice into a single-row summary."""
    valid = sub.dropna(subset=["is_correct"])
    n_total = len(sub)
    n_valid = len(valid)
    n_correct = int((valid["is_correct"] == True).sum())
    n_wrong = int((valid["is_correct"] == False).sum())
    row = {
        "n_total": n_total,
        "n_classifiable": n_valid,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_unclassified": n_total - n_valid,
        "wrong_rate": (n_wrong / n_valid) if n_valid > 0 else float("nan"),
    }
    for f in SUMMARY_FEATURES:
        cv = valid.loc[valid["is_correct"] == True, f]
        wv = valid.loc[valid["is_correct"] == False, f]
        row[f"{f}_correct_med"] = float(cv.median()) if len(cv) else float("nan")
        row[f"{f}_wrong_med"] = float(wv.median()) if len(wv) else float("nan")
        row[f"{f}_correct_p75"] = float(cv.quantile(0.75)) if len(cv) else float("nan")
        row[f"{f}_wrong_p75"] = float(wv.quantile(0.75)) if len(wv) else float("nan")
    return pd.Series(row)


def summarize(df):
    """Returns DataFrame: one row per seq (sorted) + one ALL row at the bottom."""
    per_seq = (
        df.groupby("seq", sort=True).apply(_agg).reset_index()
        if len(df) > 0
        else pd.DataFrame()
    )
    combined = pd.DataFrame([_agg(df)])
    combined.insert(0, "seq", "ALL")
    return pd.concat([per_seq, combined], ignore_index=True)


# ── Main ────────────────────────────────────────────────────────────────────


def parse_seqmap(path):
    """Read a MOTChallenge seqmap (one seq name per line; ignore 'name'
    header and comments)."""
    seqs = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.lower() in ("name",) or s.startswith("#"):
                continue
            seqs.append(s)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--diag-csv",
        required=True,
        help="GLRA diagnostic CSV (from sparse_tracker --glra-diag)",
    )
    ap.add_argument(
        "--gt-folder",
        required=True,
        help="MOT dataset split folder; expects <gt-folder>/<seq>/gt/gt.txt",
    )
    ap.add_argument(
        "--tracker-out",
        required=True,
        help="Folder with per-seq tracker output .txt files",
    )
    ap.add_argument(
        "--seqmap",
        default=None,
        help="Optional seqmap; if omitted uses unique seqs in diag CSV",
    )
    ap.add_argument(
        "--out",
        default="./glra_diag_analysis.csv",
        help="Per-record output CSV (with classification)",
    )
    ap.add_argument(
        "--summary",
        default="./glra_diag_summary.csv",
        help="Summary CSV (per-seq + ALL combined)",
    )
    ap.add_argument(
        "--iou-thresh",
        type=float,
        default=0.3,
        help="Min IoU to call a GT match (default 0.3)",
    )
    args = ap.parse_args()

    diag = pd.read_csv(args.diag_csv)
    if len(diag) == 0:
        print(f"WARNING: {args.diag_csv} is empty — nothing to analyze.")
        return
    print(f"Loaded {len(diag)} GLRA recovery records.")

    if args.seqmap and os.path.exists(args.seqmap):
        seqs = parse_seqmap(args.seqmap)
        print(f"Using {len(seqs)} sequences from {args.seqmap}")
    else:
        seqs = sorted(diag["seq"].dropna().unique().tolist())
        print(f"Using {len(seqs)} sequences from diag CSV.")

    classified_rows = []
    for seq in seqs:
        sub = diag[diag["seq"] == seq]
        if len(sub) == 0:
            continue
        gt_path = os.path.join(args.gt_folder, seq, "gt", "gt.txt")
        track_path = os.path.join(args.tracker_out, seq + ".txt")
        gt_data = load_mot_file(gt_path)
        tracker_data = load_mot_file(track_path)
        track_hist = build_track_history(tracker_data)
        print(
            f"  {seq:30s} records={len(sub):5d}  "
            f"gt_frames={len(gt_data):4d}  track_frames={len(tracker_data):4d}"
        )
        for rec in sub.to_dict(orient="records"):
            c = classify_record(rec, gt_data, track_hist, args.iou_thresh)
            rec.update(c)
            classified_rows.append(rec)

    df = pd.DataFrame(classified_rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote per-record classification → {args.out}  ({len(df)} rows)")

    summary = summarize(df)
    summary.to_csv(args.summary, index=False)
    print(f"Wrote per-seq + combined summary → {args.summary}")

    # Console headline
    all_row = summary.iloc[-1]
    print("\n=== Headline (ALL combined) ===")
    print(
        f"Total records      : {int(all_row['n_total'])}\n"
        f"Classifiable       : {int(all_row['n_classifiable'])}\n"
        f"  Correct          : {int(all_row['n_correct'])}\n"
        f"  Wrong (ID swap)  : {int(all_row['n_wrong'])}\n"
        f"  Wrong rate       : {all_row['wrong_rate']:.2%}"
    )
    print("\nMedian (correct vs wrong); look for the largest Δ — that's the signature:")
    print(
        f"  {'feature':<24s} {'correct':>10s} {'wrong':>10s} {'Δ (wrong-correct)':>20s}"
    )
    for f in SUMMARY_FEATURES:
        cm = all_row.get(f"{f}_correct_med", float("nan"))
        wm = all_row.get(f"{f}_wrong_med", float("nan"))
        delta = wm - cm if not (np.isnan(cm) or np.isnan(wm)) else float("nan")
        print(f"  {f:<24s} {cm:>10.3f} {wm:>10.3f} {delta:>+20.3f}")


if __name__ == "__main__":
    main()
