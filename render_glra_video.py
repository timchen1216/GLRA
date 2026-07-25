#!/usr/bin/env python3
"""
Render a GLRA recovery case as a VIDEO (1 fps, one frame per source frame).

Per frame:
  - draw the track bbox + center point at that frame (if observed);
    on lost frames (no observation) draw NO bbox and label "LOST".
  - a magnified inset (zoom-in) of the whole-trajectory bounding region,
    pinned to a bottom corner chosen so it does not cover the track.

Final frame (recover frame) additionally shows:
  - the full observed-history polyline
  - the GPR predicted position/box and the KF predicted position/box
  - the matched low-confidence detection
  - and the SAME GPR/KF/det overlays are also drawn inside the inset
    (predictions appear in the inset ONLY on the final frame).

Frame-id mapping: GLRA dumps use val-half (relative) ids; img1/ uses absolute
numbers.  Pass --val_json (auto) or --offset (manual): absolute = rel + offset.

Usage
-----
python render_glra_video.py \
  --case glra_cases/MOT17-09-FRCNN_t268_f86.json \
  --img_dir /home/caig/data/MOT17/train/MOT17-09-FRCNN/img1 \
  --val_json /path/to/val_half.json \
  --out glra_t268.mp4

Requires: opencv-python, numpy.
"""

import os, re, json, argparse
import numpy as np
import cv2

# BGR colors
C_OBS = (73, 110, 44)[::-1]  # green
C_CEN = (60, 200, 60)  # bright green center dot
C_KF = (31, 18, 193)  # red
C_GPR = (137, 78, 29)  # blue
C_DET = (158, 55, 181)  # magenta
C_TXT = (255, 255, 255)
C_LOST = (40, 40, 220)  # LOST label (red)
C_INSET_BORDER = (200, 200, 200)


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
    for width in (6, 5, 4, 8):
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(img_dir, f"{abs_id:0{width}d}{ext}")
            if os.path.exists(p):
                return p
    return None


def box_from_center(cx, cy, w, h):
    return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


def draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=2, dash=10):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def label(img, text, x, y, color, scale=0.6):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(0, min(x, img.shape[1] - tw - 4))
    y = max(th + 6, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 2), color, -1)
    cv2.putText(
        img,
        text,
        (x + 3, y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        C_TXT,
        1,
        cv2.LINE_AA,
    )


def traj_bounds(obs, pad_frac=0.6):
    cx = np.array([o[1] for o in obs])
    cy = np.array([o[2] for o in obs])
    w = np.array([o[3] for o in obs])
    h = np.array([o[4] for o in obs])
    x1 = (cx - w / 2).min()
    x2 = (cx + w / 2).max()
    y1 = (cy - h / 2).min()
    y2 = (cy + h / 2).max()
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad_frac
    x2 += bw * pad_frac
    y1 -= bh * pad_frac
    y2 += bh * pad_frac
    return x1, y1, x2, y2


def make_inset(frame, region, overlays, out_w=520):
    """Crop `region` (x1,y1,x2,y2) from frame, scale to width out_w, draw overlays
    (list of ('rect'/'dash'/'dot'/'line', pts, color)) in region-local coords."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in region]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = frame[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]
    scale = out_w / cw
    inset = cv2.resize(crop, (out_w, int(ch * scale)), interpolation=cv2.INTER_LINEAR)

    def tf(px, py):
        return int((px - x1) * scale), int((py - y1) * scale)

    for kind, pts, color in overlays:
        if kind == "rect":
            a = tf(pts[0], pts[1])
            b = tf(pts[2], pts[3])
            cv2.rectangle(inset, a, b, color, 2)
        elif kind == "dash":
            a = tf(pts[0], pts[1])
            b = tf(pts[2], pts[3])
            draw_dashed_rect(inset, a[0], a[1], b[0], b[1], color, 2, dash=7)
        elif kind == "dot":
            cv2.circle(inset, tf(pts[0], pts[1]), 5, color, -1)
        elif kind == "line":
            p = [tf(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]
            for i in range(1, len(p)):
                cv2.line(inset, p[i - 1], p[i], color, 2, cv2.LINE_AA)
    cv2.rectangle(
        inset, (0, 0), (inset.shape[1] - 1, inset.shape[0] - 1), C_INSET_BORDER, 2
    )
    return inset, (x1, y1, x2, y2)


def paste_inset(frame, inset, track_center, region):
    """Pin inset to the bottom corner that does not overlap the track center."""
    H, W = frame.shape[:2]
    ih, iw = inset.shape[:2]
    margin = 20
    tcx = track_center[0] if track_center else W / 2
    # choose left or right bottom depending on where the track is
    if tcx > W / 2:
        ox = margin  # track on right -> inset bottom-left
    else:
        ox = W - iw - margin  # track on left  -> inset bottom-right
    oy = H - ih - margin
    roi = frame[oy : oy + ih, ox : ox + iw]
    cv2.addWeighted(inset, 1.0, roi, 0.0, 0, roi)
    frame[oy : oy + ih, ox : ox + iw] = inset
    label(frame, "zoom", ox + 4, oy + 22, (60, 60, 60), 0.55)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out", default="glra_video.mp4")
    ap.add_argument("--val_json", default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--inset_w", type=int, default=560)
    ap.add_argument(
        "--hold_last",
        type=int,
        default=3,
        help="repeat the final annotated frame N times",
    )
    a = ap.parse_args()

    rec = json.load(open(a.case))
    offset = resolve_offset(rec, a.val_json, a.offset)
    print(
        f"[offset] {rec['seq']}: offset={offset} "
        f"(val f{rec['frame']} -> abs {rec['frame']+offset:06d})"
    )

    obs = rec["obs"]  # [rel_frame, cx, cy, w, h]
    obs_by_f = {int(o[0]): o for o in obs}
    f_first = int(obs[0][0])
    f_last = int(rec["frame"])  # recover frame
    all_frames = list(range(f_first, f_last + 1))
    observed_set = set(obs_by_f.keys())

    region = traj_bounds(obs)  # whole-trajectory zoom region

    # probe first available frame to get size
    size = None
    writer = None

    g, k, d = rec["gpr_pred"], rec["kf_pred"], rec["det"]

    def render_frame(rel_f, is_final):
        abs_f = rel_f + offset
        path = find_frame(a.img_dir, abs_f)
        if path is None:
            return None
        frame = cv2.imread(path)
        if frame is None:
            return None

        # --- main overlay ---
        track_center = None
        overlays_inset = []

        # full history polyline (context on every frame up to current)
        hist = [o for o in obs if o[0] <= rel_f]
        if len(hist) >= 2:
            pts = [(int(o[1]), int(o[2])) for o in hist]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], C_OBS, 2, cv2.LINE_AA)
            line_pts = []
            for o in hist:
                line_pts += [o[1], o[2]]
            overlays_inset.append(("line", line_pts, C_OBS))

        if rel_f in observed_set:
            o = obs_by_f[rel_f]
            x1, y1, x2, y2 = box_from_center(o[1], o[2], o[3], o[4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_OBS, 2)
            cv2.circle(frame, (int(o[1]), int(o[2])), 5, C_CEN, -1)
            track_center = (int(o[1]), int(o[2]))
            overlays_inset.append(("rect", (x1, y1, x2, y2), C_OBS))
            overlays_inset.append(("dot", (o[1], o[2]), C_CEN))
            label(frame, f"track {rec['track_id']}  f{rel_f}", x1, y1, C_OBS)
        else:
            # lost frame: no bbox, annotate LOST at last known center
            label(frame, f"LOST  f{rel_f}", 40, 70, C_LOST, 0.8)

        # header
        label(
            frame,
            f"{rec['seq']}  track {rec['track_id']}  "
            f"lost {rec['lost_duration']}  recover f{rec['frame']}",
            20,
            34,
            (40, 40, 40),
            0.6,
        )

        # --- final frame: predictions on main + inset ---
        if is_final:
            # KF
            if k["cx"] is not None:
                x1, y1, x2, y2 = box_from_center(k["cx"], k["cy"], k["w"], k["h"])
                draw_dashed_rect(frame, x1, y1, x2, y2, C_KF, 2)
                cv2.circle(frame, (int(k["cx"]), int(k["cy"])), 5, C_KF, -1)
                label(frame, "KF", x1, y2 + 22, C_KF)
                overlays_inset.append(("dash", (x1, y1, x2, y2), C_KF))
                overlays_inset.append(("dot", (k["cx"], k["cy"]), C_KF))
            # GPR
            x1, y1, x2, y2 = box_from_center(g["cx"], g["cy"], g["w"], g["h"])
            draw_dashed_rect(frame, x1, y1, x2, y2, C_GPR, 2)
            cv2.circle(frame, (int(g["cx"]), int(g["cy"])), 5, C_GPR, -1)
            label(frame, "GPR", x1, y1, C_GPR)
            overlays_inset.append(("dash", (x1, y1, x2, y2), C_GPR))
            overlays_inset.append(("dot", (g["cx"], g["cy"]), C_GPR))
            # det
            x1, y1, x2, y2 = box_from_center(d["cx"], d["cy"], d["w"], d["h"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_DET, 2)
            label(frame, f"det {d['score']:.2f}", x2 - 70, y2 + 22, C_DET)
            overlays_inset.append(("rect", (x1, y1, x2, y2), C_DET))

        return frame

    for rel_f in all_frames:
        is_final = rel_f == f_last
        frame = render_frame(rel_f, is_final)
        if frame is None:
            print(f"  [skip] no image for rel f{rel_f} (abs {rel_f+offset})")
            continue
        if writer is None:
            size = (frame.shape[1], frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(a.out, fourcc, a.fps, size)
        writer.write(frame)
        if is_final:
            for _ in range(max(0, a.hold_last - 1)):
                writer.write(frame)
            cv2.imwrite(os.path.splitext(a.out)[0] + "_final.jpg", frame)

    if writer is not None:
        writer.release()
        print("wrote", a.out, "and", os.path.splitext(a.out)[0] + "_final.jpg")
    else:
        print("no frames written -- check img_dir / offset")


if __name__ == "__main__":
    main()
"""
python render_glra_video.py \
  --case glra_cases/MOT17-09-FRCNN_t268_f86.json \
  --img_dir /home/caig/data/MOT17/train/MOT17-09-FRCNN/img1 \
  --val_json /home/caig/data/MOT17/annotations/val_half.json \
  --out glra_t268.mp4

  python render_glra_video.py \
  --case glra_cases/MOT20-05_t1196_f173.json \
  --img_dir /home/caig/data/MOT20/train/MOT20-05/img1 \
  --val_json /home/caig/data/MOT20/annotations/val_half.json \
  --out glra_t1196.mp4
"""
