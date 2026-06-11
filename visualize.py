"""
visualize.py  ──  SparseTracker 軌跡狀態視覺化工具
================================================================
用法:
    python visualize.py \
        --seq_dir  /path/to/MOT17-02/img1 \
        --det_dir  /path/to/MOT17-02/det/det.txt \
        --out_dir  ./vis_output \
        [--lost_display kf|gpr|both] \
        [--show_glra_recovery] \
        [--save_video]

--lost_display 選項:
    kf   : 只畫 KF 預測位置（實線紅框）           [預設]
    gpr  : 只畫 GPR 預測位置（實線橘框）
    both : 兩者都畫（KF 實線紅框 + GPR 虛線橘框）

--show_glra_recovery:
    在 GLRA 成功接回的那一幀，額外疊加：
      • 橘色虛線框  ── GPR 預測位置
      • 白色虛線框  ── 匹配到的 detection 位置
      • 橘→白 箭頭  ── 兩者中心連線，顯示 GPR 預測誤差
      • 右下角標籤  ── cost / sigma / lost N frames

顏色說明:
    ■ 綠色  Active (A)       — tracked & is_activated
    ■ 青色  Unconfirmed (U)  — tracked & !is_activated
    ■ 紅色  Lost-KF (L)      — KF 預測位置（實線）
    ■ 橘色  Lost-GPR (G)     — GPR 預測位置（虛線）
    ■ 灰色  Removed (R)      — 本幀剛被移除（需 --show_removed）
    ── GLRA recovery overlay ──
    ■ 橘虛線  GPR 預測框
    ■ 白虛線  matched detection 框
    → 橘白箭頭 GPR center → det center
================================================================
"""

import argparse
import os
import sys
import glob
import cv2
import numpy as np

# ── 顏色 (BGR) ───────────────────────────────────────────────────
COLOR = {
    "active": (57, 255, 20),
    "unconfirmed": (0, 220, 255),
    "lost_kf": (0, 60, 255),
    "lost_gpr": (0, 165, 255),
    "removed": (120, 120, 120),
    # GLRA recovery overlay
    "glra_gpr": (0, 165, 255),  # 橘：GPR 預測框
    "glra_det": (255, 255, 255),  # 白：matched detection 框
    "glra_arrow": (0, 200, 255),  # 橘白箭頭
}
LABEL = {
    "active": "A",
    "unconfirmed": "U",
    "lost_kf": "L",
    "lost_gpr": "G",
    "removed": "R",
}
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICK = 1


# ── 基本繪圖工具 ─────────────────────────────────────────────────


def _clip_tlwh(tlwh, img_shape):
    ih, iw = img_shape[:2]
    x, y = max(0, int(tlwh[0])), max(0, int(tlwh[1]))
    x2 = min(iw, int(tlwh[0] + tlwh[2]))
    y2 = min(ih, int(tlwh[1] + tlwh[3]))
    return x, y, x2, y2


def draw_bbox(img, tlwh, track_id, style, score=None, dashed=False):
    x, y, x2, y2 = _clip_tlwh(tlwh, img.shape)
    if x2 <= x or y2 <= y:
        return
    color = COLOR[style]
    if dashed:
        _draw_dashed_rect(img, (x, y), (x2, y2), color, THICKNESS)
    else:
        cv2.rectangle(img, (x, y), (x2, y2), color, THICKNESS)

    # ID 標籤（左上）
    id_text = f"#{track_id}"
    (tw, th), bl = cv2.getTextSize(id_text, FONT, FONT_SCALE, FONT_THICK)
    ly = max(y - 4, th + 2)
    cv2.rectangle(img, (x, ly - th - 2), (x + tw + 2, ly + bl), color, -1)
    cv2.putText(
        img,
        id_text,
        (x + 1, ly - 1),
        FONT,
        FONT_SCALE,
        (0, 0, 0),
        FONT_THICK,
        cv2.LINE_AA,
    )

    # 狀態 + score（右下）
    st = LABEL.get(style, style)
    if score is not None:
        st += f" {score:.2f}"
    (tw2, th2), _ = cv2.getTextSize(st, FONT, FONT_SCALE - 0.05, FONT_THICK)
    rx = max(0, x2 - tw2 - 2)
    cv2.rectangle(img, (rx, y2 - th2 - 4), (x2, y2), color, -1)
    cv2.putText(
        img,
        st,
        (rx + 1, y2 - 2),
        FONT,
        FONT_SCALE - 0.05,
        (0, 0, 0),
        FONT_THICK,
        cv2.LINE_AA,
    )


def draw_gpr_sigma(img, tlwh, sigma_px):
    cx = int(tlwh[0] + tlwh[2] / 2)
    cy = int(tlwh[1] + tlwh[3] / 2)
    cv2.circle(img, (cx, cy), max(3, int(sigma_px)), COLOR["lost_gpr"], 1, cv2.LINE_AA)


def _draw_dashed_rect(img, pt1, pt2, color, thickness, dash=10):
    x1, y1 = pt1
    x2, y2 = pt2
    for s, e in _dash_segs(x1, y1, x2, y1, dash):
        cv2.line(img, s, e, color, thickness)
    for s, e in _dash_segs(x2, y1, x2, y2, dash):
        cv2.line(img, s, e, color, thickness)
    for s, e in _dash_segs(x2, y2, x1, y2, dash):
        cv2.line(img, s, e, color, thickness)
    for s, e in _dash_segs(x1, y2, x1, y1, dash):
        cv2.line(img, s, e, color, thickness)


def _dash_segs(x1, y1, x2, y2, dash):
    length = max(1, int(np.hypot(x2 - x1, y2 - y1)))
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    segs, i = [], 0
    while i < length:
        s = (int(x1 + dx * i), int(y1 + dy * i))
        e = (int(x1 + dx * min(i + dash, length)), int(y1 + dy * min(i + dash, length)))
        segs.append((s, e))
        i += dash * 2
    return segs


# ── GLRA recovery overlay ────────────────────────────────────────


def draw_glra_recovery(img, event: dict):
    """
    event 來自 tracker.glra_events，包含:
        frame_id, track_id, lost_frames,
        gpr_tlwh (may be None), gpr_sigma (may be None),
        det_tlwh, det_score, cost
    在圖上畫：
      1. 橘色虛線框  — GPR 預測位置
      2. sigma 圓圈  — GPR 不確定性
      3. 白色虛線框  — matched detection 位置
      4. 橘白箭頭    — GPR center → det center
      5. 右下角資訊標籤
    """
    tid = event["track_id"]
    gpr_tlwh = event["gpr_tlwh"]  # None if GPR had insufficient obs
    gpr_sigma = event["gpr_sigma"]
    det_tlwh = event["det_tlwh"]
    det_score = event["det_score"]
    cost = event["cost"]
    lost_f = event["lost_frames"]

    det_cx = int(det_tlwh[0] + det_tlwh[2] / 2)
    det_cy = int(det_tlwh[1] + det_tlwh[3] / 2)

    # 1. matched detection 框（白色虛線，粗一點）
    dx, dy, dx2, dy2 = _clip_tlwh(det_tlwh, img.shape)
    if dx2 > dx and dy2 > dy:
        _draw_dashed_rect(img, (dx, dy), (dx2, dy2), COLOR["glra_det"], THICKNESS + 1)

    if gpr_tlwh is not None:
        gpr_cx = int(gpr_tlwh[0] + gpr_tlwh[2] / 2)
        gpr_cy = int(gpr_tlwh[1] + gpr_tlwh[3] / 2)

        # 2. GPR 預測框（橘色虛線）
        gx, gy, gx2, gy2 = _clip_tlwh(gpr_tlwh, img.shape)
        if gx2 > gx and gy2 > gy:
            _draw_dashed_rect(img, (gx, gy), (gx2, gy2), COLOR["glra_gpr"], THICKNESS)

        # 3. sigma 圓圈
        if gpr_sigma is not None:
            cv2.circle(
                img,
                (gpr_cx, gpr_cy),
                max(3, int(gpr_sigma)),
                COLOR["glra_gpr"],
                1,
                cv2.LINE_AA,
            )

        # 4. 箭頭：GPR center → det center
        cv2.arrowedLine(
            img,
            (gpr_cx, gpr_cy),
            (det_cx, det_cy),
            COLOR["glra_arrow"],
            2,
            cv2.LINE_AA,
            tipLength=0.2,
        )

        # 5. 標籤（貼在 det 框右下）
        err_px = np.hypot(det_cx - gpr_cx, det_cy - gpr_cy)
        lines = [
            f"GLRA #{tid}",
            f"lost {lost_f}f",
            f"cost {cost:.3f}",
            f"err {err_px:.1f}px",
        ]
        if gpr_sigma is not None:
            lines.append(f"sig {gpr_sigma:.1f}px")
    else:
        # GPR 沒有足夠觀測，只標注 det 框和文字
        lines = [f"GLRA #{tid}", f"lost {lost_f}f", f"cost {cost:.3f}", "GPR:N/A"]

    _draw_info_box(img, det_cx, det_cy, lines, COLOR["glra_arrow"])


def _draw_info_box(img, cx, cy, lines, color):
    """在 (cx, cy) 旁邊畫多行資訊框。"""
    fscale, fthick = 0.42, 1
    row_h = 16
    max_w = max(cv2.getTextSize(l, FONT, fscale, fthick)[0][0] for l in lines)
    ih, iw = img.shape[:2]

    # 盡量放在右下，超出邊界則往左/上偏
    bx = min(cx + 6, iw - max_w - 8)
    by = min(cy + 6, ih - len(lines) * row_h - 6)
    bx = max(0, bx)
    by = max(0, by)

    # 半透明背景
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (bx - 2, by - 2),
        (bx + max_w + 4, by + len(lines) * row_h + 2),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (bx, by + (i + 1) * row_h - 2),
            FONT,
            fscale,
            color,
            fthick,
            cv2.LINE_AA,
        )


# ── HUD ──────────────────────────────────────────────────────────


def draw_legend(img, lost_display):
    items = [("Active (A)", "active"), ("Unconfirmed (U)", "unconfirmed")]
    if lost_display in ("kf", "both"):
        items.append(("Lost-KF (L)", "lost_kf"))
    if lost_display in ("gpr", "both"):
        items.append(("Lost-GPR (G)", "lost_gpr"))
    items.append(("Removed (R)", "removed"))
    items.append(("GLRA recovery", "glra_arrow"))

    x0, y0, row_h, box_w, pad = img.shape[1] - 230, 10, 22, 16, 6
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x0 - pad, y0 - pad),
        (x0 + 225, y0 + len(items) * row_h + pad),
        (30, 30, 30),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (label, key) in enumerate(items):
        y = y0 + i * row_h + row_h // 2
        cv2.rectangle(
            img, (x0, y - box_w // 2), (x0 + box_w, y + box_w // 2), COLOR[key], -1
        )
        cv2.putText(
            img,
            label,
            (x0 + box_w + 6, y + 5),
            FONT,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_frame_info(img, frame_id, counts, n_glra):
    lines = [
        f"Frame: {frame_id}",
        f"Active:      {counts.get('active', 0)}",
        f"Unconfirmed: {counts.get('unconfirmed', 0)}",
        f"Lost:        {counts.get('lost', 0)}",
        f"Removed:     {counts.get('removed', 0)}",
        f"GLRA rec:    {n_glra}",
    ]
    x0, y0, text_h = 8, 20, 20
    overlay = img.copy()
    cv2.rectangle(
        overlay, (0, 0), (225, y0 + len(lines) * text_h + 4), (30, 30, 30), -1
    )
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x0, y0 + i * text_h),
            FONT,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ── 主流程 ───────────────────────────────────────────────────────


def run_visualize(args):
    try:
        from tracker.sparse_tracker import SparseTracker, STrack
        from tracker.basetrack import TrackState
    except ImportError:
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent)
        from tracker.sparse_tracker import SparseTracker, STrack
        from tracker.basetrack import TrackState

    tracker = SparseTracker(args)
    lost_display = args.lost_display

    img_paths = sorted(
        glob.glob(os.path.join(args.seq_dir, "*.jpg"))
        + glob.glob(os.path.join(args.seq_dir, "*.png"))
    )
    if not img_paths:
        raise FileNotFoundError(f"找不到圖片: {args.seq_dir}")

    det_by_frame = {}
    if args.det_dir and os.path.isfile(args.det_dir):
        with open(args.det_dir) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                fid = int(parts[0])
                x, y, w, h = (
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
                det_by_frame.setdefault(fid, []).append(
                    [x, y, x + w, y + h, float(parts[6])]
                )
    else:
        print("[警告] 未提供偵測檔案，將使用空偵測。")

    os.makedirs(args.out_dir, exist_ok=True)

    video_writer = None
    if getattr(args, "save_video", False):
        sample = cv2.imread(img_paths[0])
        h, w = sample.shape[:2]
        vpath = os.path.join(args.out_dir, "visualization.mp4")
        video_writer = cv2.VideoWriter(
            vpath, cv2.VideoWriter_fourcc(*"mp4v"), getattr(args, "fps", 30), (w, h)
        )
        print(f"[INFO] 輸出 video → {vpath}")

    print(
        f"[INFO] lost_display={lost_display}  "
        f"show_glra_recovery={args.show_glra_recovery}  "
        f"共 {len(img_paths)} 幀"
    )

    total_glra = 0

    for idx, img_path in enumerate(img_paths):
        frame_id = idx + 1
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 無法讀取 {img_path}，跳過")
            continue

        dets_np = np.array(det_by_frame.get(frame_id, []), dtype=np.float32)
        if len(dets_np) == 0:
            dets_np = np.empty((0, 5), dtype=np.float32)

        # tracker.glra_events 是累積的，記錄本幀之前加入的 index
        prev_event_count = len(tracker.glra_events)
        _ = tracker.update(_make_output_results(dets_np), curr_img=img)
        # 本幀新增的 events
        frame_events = tracker.glra_events[prev_event_count:]
        n_glra_this_frame = len(frame_events)
        total_glra += n_glra_this_frame

        # ── 分類 tracks ──
        active_tracks = [t for t in tracker.tracked_stracks if t.is_activated]
        unconfirmed_tracks = [t for t in tracker.tracked_stracks if not t.is_activated]
        lost_tracks = list(tracker.lost_stracks)
        removed_tracks = list(tracker.removed_stracks)

        counts = dict(
            active=len(active_tracks),
            unconfirmed=len(unconfirmed_tracks),
            lost=len(lost_tracks),
            removed=len(removed_tracks),
        )

        # ── 繪圖（由底層到頂層）──

        # Removed
        if getattr(args, "show_removed", False):
            for t in removed_tracks:
                draw_bbox(img, t.tlwh, t.track_id, "removed", t.score)

        # Lost tracks（KF / GPR）
        # GLRA 成功接回的 track_id：本幀不畫 lost，改由 recovery overlay 顯示
        glra_recovered_ids = {e["track_id"] for e in frame_events}

        for t in lost_tracks:
            if t.track_id in glra_recovered_ids:
                continue  # 由 recovery overlay 接管
            if lost_display in ("kf", "both"):
                draw_bbox(img, t.tlwh, t.track_id, "lost_kf", t.score, dashed=False)
            if lost_display in ("gpr", "both"):
                try:
                    from tracker.matching import gpr_predict_bbox
                except ImportError:
                    from matching import gpr_predict_bbox
                result = gpr_predict_bbox(t, frame_id, min_obs=args.gpr_min_obs)
                if result is not None:
                    pred_tlbr, sigma_px = result
                    pw = pred_tlbr[2] - pred_tlbr[0]
                    ph = pred_tlbr[3] - pred_tlbr[1]
                    pred_tlwh = np.array([pred_tlbr[0], pred_tlbr[1], pw, ph])
                    draw_bbox(
                        img, pred_tlwh, t.track_id, "lost_gpr", t.score, dashed=True
                    )
                    if args.show_sigma:
                        draw_gpr_sigma(img, pred_tlwh, sigma_px)
                elif lost_display == "gpr":
                    draw_bbox(img, t.tlwh, t.track_id, "removed", t.score, dashed=True)

        # Unconfirmed
        for t in unconfirmed_tracks:
            draw_bbox(img, t.tlwh, t.track_id, "unconfirmed", t.score)

        # Active
        for t in active_tracks:
            draw_bbox(img, t.tlwh, t.track_id, "active", t.score)

        # ── GLRA recovery overlay（最上層）──
        if args.show_glra_recovery:
            for event in frame_events:
                draw_glra_recovery(img, event)

        draw_frame_info(img, frame_id, counts, n_glra_this_frame)
        draw_legend(img, lost_display)

        cv2.imwrite(os.path.join(args.out_dir, f"{frame_id:06d}.jpg"), img)
        if video_writer is not None:
            video_writer.write(img)

        if frame_id % 50 == 0 or frame_id == len(img_paths):
            print(
                f"  [{frame_id:4d}/{len(img_paths)}]  "
                f"active={counts['active']:3d}  lost={counts['lost']:3d}  "
                f"GLRA_rec={n_glra_this_frame}"
            )

    if video_writer is not None:
        video_writer.release()
        print("[INFO] Video 已儲存。")
    print(f"[DONE] 輸出至 {args.out_dir}  (GLRA 總回收: {total_glra})")


# ── 輔助 ─────────────────────────────────────────────────────────


class _FakeOutput:
    class _T:
        def __init__(self, d):
            self._d = d

        def cpu(self):
            return self

        def numpy(self):
            return self._d

        @property
        def tensor(self):
            return self

    def __init__(self, dets_np):
        boxes = dets_np[:, :4] if len(dets_np) else np.empty((0, 4), np.float32)
        scores = dets_np[:, 4] if len(dets_np) else np.empty((0,), np.float32)
        self.pred_boxes = _FakeOutput._T(boxes)
        self.scores = _FakeOutput._T(scores)


def _make_output_results(dets_np):
    return _FakeOutput(dets_np)


# ── CLI ──────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="SparseTracker Track-State Visualizer")
    p.add_argument("--seq_dir", required=True)
    p.add_argument("--det_dir", default=None)
    p.add_argument("--out_dir", default="./vis_output")

    p.add_argument("--lost_display", default="kf", choices=["kf", "gpr", "both"])
    p.add_argument("--show_sigma", action="store_true")
    p.add_argument(
        "--show_glra_recovery",
        action="store_true",
        help="在 GLRA 成功接回的幀顯示：GPR預測框、det框、箭頭、誤差資訊",
    )

    p.add_argument("--track_thresh", type=float, default=0.6)
    p.add_argument("--match_thresh", type=float, default=0.8)
    p.add_argument("--confirm_thresh", type=float, default=0.7)
    p.add_argument("--track_buffer", type=int, default=30)
    p.add_argument("--frame_rate", type=int, default=30)
    p.add_argument("--down_scale", type=int, default=2)
    p.add_argument("--depth_levels", type=int, default=10)
    p.add_argument("--depth_levels_low", type=int, default=10)
    p.add_argument("--val_ann", type=str, default="val.json")
    p.add_argument("--mot20", action="store_true")

    p.add_argument("--use_diou", action="store_true")
    p.add_argument("--use_glra", action="store_true")
    p.add_argument("--gpr_history_len", type=int, default=30)
    p.add_argument("--glra_thresh", type=float, default=0.7)
    p.add_argument("--gpr_max_lost", type=int, default=5)
    p.add_argument("--gpr_min_obs", type=int, default=3)
    p.add_argument("--use_gmc_history", action="store_true", default=True)
    p.add_argument("--glra_adaptive", action="store_true")
    p.add_argument("--glra_sigma_scale", type=float, default=30.0)
    p.add_argument("--glra_thresh_range", type=float, default=0.25)
    p.add_argument("--glra_max_thresh", type=float, default=0.85)

    p.add_argument("--show_removed", action="store_true")
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--fps", type=int, default=30)

    return p.parse_args()


if __name__ == "__main__":
    run_visualize(parse_args())

"""
python3 visualize.py \
    --seq_dir  /home/caig/data/MOT17/train/MOT17-02-FRCNN/img1 \
    --det_dir  /home/caig/data/MOT17/train/MOT17-02-FRCNN/det/det.txt \
    --out_dir  ./vis_output/MOT17-02 \
    --use_diou \
    --lost_display gpr \
    --show_glra_recovery \
    --show_sigma \
    --use_glra \
    --save_video

"""
