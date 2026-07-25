#!/usr/bin/env python3
"""
find_idsw.py — Find cases where SparseTrack switches identity on a GT track
but Ours keeps a single consistent id (the ID-stability story for the defense).

Anchoring on GT: for each GT id, per frame we record which sparse track id and
which ours track id covers it (best IoU >= thr). A SparseTrack IDSW = the
sparse covering-id changes from A to B along the same GT track. We keep the
case when Ours holds ONE id across the switch window (the frames around A->B).

Reads sparse/ours/gt txt (same format as glra_compare.py). Ranks the switches
so the clearest ones (long stable ours id, clean A->B) come first, then can
auto-render the top-N as zoom comparison figures via glra_compare.py.

Usage (rank only)
-----------------
python find_idsw.py --sparse S.txt --ours O.txt --gt gt.txt --top 20

Usage (rank + render), needs img:
python find_idsw.py --sparse S.txt --ours O.txt --gt gt.txt \
    --img .../img1 --val-json val_half.json --seq MOT20-05 \
    --render-out out/idsw --top 5
"""

import argparse
import os
from collections import defaultdict, Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

NEUTRAL = (150, 150, 150)  # context / stable id (grey, BGR)
YELLOW = (0, 255, 255)  # ours: the single held id
RED = (0, 0, 255)  # sparse: after the switch (new id)
GREEN = (60, 200, 60)  # sparse: before the switch (original id)


def load_mot(path, gt=False):
    data = defaultdict(list)
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


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def cover_id(dets, gt_box, thr):
    """track id of the det best covering gt_box (IoU>=thr), else None."""
    best, biou = None, thr
    for tid, x, y, w, h in dets:
        v = iou((x, y, w, h), gt_box)
        if v >= biou:
            biou, best = v, tid
    return best


def runs(seq_ids):
    """Compress [(frame,id)] into maximal runs of equal id, skipping None.
    Returns list of (id, start_frame, end_frame, n_frames)."""
    out = []
    cur_id = None
    start = prev = None
    n = 0
    for fr, i in seq_ids:
        if i is None:
            continue
        if i != cur_id:
            if cur_id is not None:
                out.append((cur_id, start, prev, n))
            cur_id, start, n = i, fr, 0
        prev = fr
        n += 1
    if cur_id is not None:
        out.append((cur_id, start, prev, n))
    return out


def ours_id_window(ours_seq, f0, f1):
    """Which ours ids cover the GT across [f0,f1]; return (dominant_id, purity,
    n_covered) where purity = frames_on_dominant / frames_covered."""
    ids = [i for fr, i in ours_seq if f0 <= fr <= f1 and i is not None]
    if not ids:
        return None, 0.0, 0
    c = Counter(ids)
    dom, cnt = c.most_common(1)[0]
    return dom, cnt / len(ids), len(ids)


def find_switches(gt, sparse, ours, thr, win):
    """
    Target case: a GT track that both trackers LOSE for a stretch (a real
    occlusion gap where neither sparse nor ours covers it), and on reappearance
    SparseTrack assigns a NEW id (A before -> B after, A!=B) while Ours holds
    the SAME id across the gap.

    The gap may be (a) frames where GT exists but neither tracker covers it, or
    (b) frames where GT itself is absent (true occlusion) between two covered
    frames. We build the per-frame covered-id sequence over only the frames
    where each tracker actually covers this GT, then look at consecutive
    *covered* frames whose frame numbers are non-adjacent (a hole) OR where the
    covered id flips with a both-lost stretch in between.
    """
    gt_by_id = frames_by_id(gt)
    cases = []
    for gid, track in gt_by_id.items():
        frs = sorted(track)
        # frames where sparse / ours actually cover this GT
        sp_cov = {fr: cover_id(sparse.get(fr, []), track[fr], thr) for fr in frs}
        ou_cov = {fr: cover_id(ours.get(fr, []), track[fr], thr) for fr in frs}
        # frames where EITHER covers (target is "visible" to some tracker)
        seen = [fr for fr in frs if sp_cov[fr] is not None or ou_cov[fr] is not None]
        if len(seen) < 2:
            continue
        # walk consecutive seen-frames; a gap = a jump where the in-between
        # frames are all both-lost (no coverage by either tracker)
        for a_idx in range(len(seen) - 1):
            pre, post = seen[a_idx], seen[a_idx + 1]
            if post - pre <= 1:
                continue  # adjacent, no gap
            # verify every frame strictly between is both-lost
            both_lost = True
            for f in range(pre + 1, post):
                s = cover_id(sparse.get(f, []), track[f], thr) if f in track else None
                o = cover_id(ours.get(f, []), track[f], thr) if f in track else None
                if s is not None or o is not None:
                    both_lost = False
                    break
            if not both_lost:
                continue
            sp_a, sp_b = sp_cov[pre], sp_cov[post]
            ou_a, ou_b = ou_cov[pre], ou_cov[post]
            sparse_switch = sp_a is not None and sp_b is not None and sp_a != sp_b
            ours_kept = ou_a is not None and ou_b is not None and ou_a == ou_b
            if not (sparse_switch and ours_kept):
                continue
            a_len = _run_len_back_seen(sp_cov, frs, pre, sp_a)
            b_len = _run_len_fwd_seen(sp_cov, frs, post, sp_b)
            ou_len = _run_len_back_seen(ou_cov, frs, pre, ou_a) + _run_len_fwd_seen(
                ou_cov, frs, post, ou_b
            )
            cases.append(
                {
                    "gt_id": gid,
                    "gap_start": pre + 1,
                    "gap_end": post - 1,
                    "gap_len": post - pre - 1,
                    "reappear": post,
                    "sw_frame": post,
                    "sparse_a": sp_a,
                    "sparse_b": sp_b,
                    "a_len": a_len,
                    "b_len": b_len,
                    "ours_id": ou_a,
                    "ours_cov": ou_len,
                    "ours_purity": 1.0,
                    "start": pre,
                    "end": post,
                }
            )
    return cases


def _run_len_back_seen(cov, frs, frame, want):
    """consecutive covered frames (by frame number) ending at `frame` == want."""
    n, f = 0, frame
    while f in cov and cov[f] == want:
        n += 1
        f -= 1
    return n


def _run_len_fwd_seen(cov, frs, frame, want):
    n, f = 0, frame
    while f in cov and cov[f] == want:
        n += 1
        f += 1
    return n


def score(c):
    """Clear cases first: a real occlusion gap, stable ids on both sides of it,
    both sparse segments well-established so the switch is unambiguous."""
    return (
        2.0 * c["gap_len"]  # a real disappearance
        + min(c["a_len"], c["b_len"])  # both sparse ids established
        + 0.3 * c["ours_cov"]
    )  # ours well-anchored around gap


def read_frame(img_dir, fr, fallback, offset=0):
    real = fr + offset
    for w in (6, 5, 4, 8):
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(img_dir, f"{real:0{w}d}{ext}")
            if os.path.isfile(p):
                im = cv2.imread(p)
                if im is not None:
                    return im
    return np.zeros(fallback, np.uint8)


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
    raise ValueError(f"no offset for '{seq}' in {val_json}")


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


def draw_ctx(img, dets, skip=None):
    for tid, x, y, w, h in dets:
        if skip is not None and (x, y, w, h) == skip:
            continue
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), NEUTRAL, 1)


def draw_tgt(img, box, color, tid):
    x, y, w, h = box
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 3)
    cv2.putText(
        img,
        f"id{tid}",
        (int(x), max(0, int(y) - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def render_idsw(
    c,
    sparse,
    ours,
    gt_by_id,
    img_dir,
    out_dir,
    thr,
    pad,
    offset,
    fallback,
    fps,
    zoom_size,
    zoom_pad,
):
    """Full + zoom video/keyframe for one IDSW case."""
    gt_track = gt_by_id[c["gt_id"]]
    tag = (
        f"gt{c['gt_id']}_gap{c['gap_start']}-{c['gap_end']}"
        f"_sw{c['sparse_a']}to{c['sparse_b']}"
    )
    kf_dir = os.path.join(out_dir, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)
    H, W = fallback[:2]

    f_lo, f_hi = c["gap_start"] - pad, c["gap_end"] + pad
    win_boxes = [gt_track[f] for f in range(f_lo, f_hi + 1) if f in gt_track] or [
        (0, 0, W, H)
    ]
    crop = crop_box(fallback, win_boxes, zoom_pad, zoom_size)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_full = cv2.VideoWriter(
        os.path.join(out_dir, f"{tag}.mp4"), fourcc, fps, (W * 2 + 20, H + 34)
    )
    zw, zh = zoom_size
    vw_zoom = cv2.VideoWriter(
        os.path.join(out_dir, f"{tag}_zoom.mp4"), fourcc, fps, (zw * 2 + 20, zh + 34)
    )

    def compose(fr, crop_rect=None, out_size=None):
        base = read_frame(img_dir, fr, fallback, offset)
        left, right = base.copy(), base.copy()
        gbox = gt_track.get(fr)
        # sparse target
        sid = cover_id(sparse.get(fr, []), gbox, thr) if gbox is not None else None
        sbox = None
        if sid is not None:
            for tid, x, y, w, h in sparse[fr]:
                if tid == sid and iou((x, y, w, h), gbox) >= thr:
                    sbox = (x, y, w, h)
                    break
        # ours target
        oid = cover_id(ours.get(fr, []), gbox, thr) if gbox is not None else None
        obox = None
        if oid is not None:
            for tid, x, y, w, h in ours[fr]:
                if tid == oid and iou((x, y, w, h), gbox) >= thr:
                    obox = (x, y, w, h)
                    break
        draw_ctx(left, sparse.get(fr, []), skip=sbox)
        draw_ctx(right, ours.get(fr, []), skip=obox)
        if sbox is not None:
            col = GREEN if fr < c["sw_frame"] else RED
            draw_tgt(left, sbox, col, sid)
        if obox is not None:
            col = YELLOW if oid == c["ours_id"] else NEUTRAL
            draw_tgt(right, obox, col, oid)
        if crop_rect is not None:
            x1, y1, x2, y2 = crop_rect
            left, right = left[y1:y2, x1:x2], right[y1:y2, x1:x2]
            if out_size is not None:
                left = cv2.resize(left, out_size)
                right = cv2.resize(right, out_size)
        left = label_bar(left, f"SparseTrack   frame {fr}")
        right = label_bar(right, f"Ours (GLRA)   frame {fr}")
        gap = np.full((left.shape[0], 20, 3), 255, np.uint8)
        return np.hstack([left, gap, right])

    for fr in range(f_lo, f_hi + 1):
        if fr < 1:
            continue
        vw_full.write(compose(fr))
        vw_zoom.write(compose(fr, crop, zoom_size))
    vw_full.release()
    vw_zoom.release()

    fr = c["sw_frame"]
    cv2.imwrite(os.path.join(kf_dir, f"{tag}.png"), compose(fr))
    cv2.imwrite(os.path.join(kf_dir, f"{tag}_zoom.png"), compose(fr, crop, zoom_size))
    return tag


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument(
        "--win",
        type=int,
        default=8,
        help="frames on each side of switch to check ours stability",
    )
    ap.add_argument("--top", type=int, default=20)
    # optional render
    ap.add_argument("--img", default=None)
    ap.add_argument("--val-json", default=None)
    ap.add_argument("--seq", default=None)
    ap.add_argument("--render-out", default=None)
    ap.add_argument("--pad", type=int, default=20)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--zoom-w", type=int, default=320)
    ap.add_argument("--zoom-h", type=int, default=480)
    ap.add_argument("--zoom-pad", type=float, default=1.5)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    gt = load_mot(args.gt, gt=True)
    sparse = load_mot(args.sparse)
    ours = load_mot(args.ours)

    cases = find_switches(gt, sparse, ours, args.iou_thr, args.win)
    for c in cases:
        c["score"] = score(c)
    cases.sort(key=lambda c: c["score"], reverse=True)

    hdr = (
        f"{'#':>3} {'score':>6} {'GTid':>5} {'gapS':>5} {'gapE':>5} "
        f"{'gLen':>4} {'reap':>5} {'spA':>5} {'spB':>5} "
        f"{'aLen':>4} {'bLen':>4} {'ourID':>5} {'oCov':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, c in enumerate(cases[: args.top]):
        print(
            f"{i:>3} {c['score']:>6.1f} {c['gt_id']:>5} {c['gap_start']:>5} "
            f"{c['gap_end']:>5} {c['gap_len']:>4} {c['reappear']:>5} "
            f"{c['sparse_a']:>5} {c['sparse_b']:>5} {c['a_len']:>4} {c['b_len']:>4} "
            f"{c['ours_id']:>5} {c['ours_cov']:>4}"
        )
    print(
        f"\n{len(cases)} cases: both trackers lose the GT for gLen frames, "
        f"then SparseTrack reappears as a NEW id (spA->spB) while Ours keeps ourID."
    )

    if not (args.img and args.render_out):
        return
    if not cases:
        print("nothing to render.")
        return

    offset = args.offset
    if args.val_json:
        seq = args.seq or os.path.basename(os.path.dirname(os.path.normpath(args.img)))
        offset = resolve_offset(args.val_json, seq)
        print(f"[offset] {seq}: offset={offset}")

    # probe frame size
    fallback = None
    for fr in sorted(sparse):
        s = read_frame(args.img, fr, (0, 0, 0), offset)
        if s.size:
            fallback = (s.shape[0], s.shape[1], 3)
            break
    if fallback is None:
        print("no readable frames in --img")
        return

    gt_by_id = frames_by_id(gt)
    os.makedirs(args.render_out, exist_ok=True)
    zoom_size = (args.zoom_w, args.zoom_h)
    for i, c in enumerate(cases[: args.top]):
        sub = os.path.join(args.render_out, f"idsw{i:02d}")
        os.makedirs(sub, exist_ok=True)
        tag = render_idsw(
            c,
            sparse,
            ours,
            gt_by_id,
            args.img,
            sub,
            args.iou_thr,
            args.pad,
            offset,
            fallback,
            args.fps,
            zoom_size,
            args.zoom_pad,
        )
        print(f"[render {i}] {tag}")
    print(f"\nDone. Browse {args.render_out}/idsw*/keyframes/*_zoom.png")


if __name__ == "__main__":
    main()

"""
python find_idsw.py \
    --sparse /home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results_sparsetrack/MOT20-05.txt \
    --ours   /home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results_glra/MOT20-05.txt \
    --gt     /home/caig/data/MOT20/train/MOT20-05/gt/gt.txt \
    --win 8 --top 20

python find_idsw.py \
    --sparse /home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results_sparsetrack/MOT20-03.txt \
    --ours   /home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results_glra/MOT20-03.txt \
    --gt     /home/caig/data/MOT20/train/MOT20-03/gt/gt.txt \
    --img    /home/caig/data/MOT20/train/MOT20-03/img1 \
    --val-json /home/caig/data/MOT20/annotations/val_half.json --seq MOT20-03 \
    --render-out out/idsw --top 5

"""
