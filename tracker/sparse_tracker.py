import numpy as np
from collections import deque

import os.path as osp
import copy
import torch
import torch.nn.functional as F
from tracker import pbcvt
from .kalman_filter import KalmanFilter
from .matching import *
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    gpr_maxlen = 30  # overridden by SparseTracker.__init__ via args.gpr_history_len

    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.deep_vector = self._get_deep_vec()

        self.score = score
        self.tracklet_len = 0
        self.gpr_obs = []  # list of (frame_id, cx, cy, w, h)

    def _get_deep_vec(self):
        cx = self._tlwh[0] + 0.5 * self._tlwh[2]
        y2 = self._tlwh[1] + self._tlwh[3]
        lendth = 2000 - y2
        return np.asarray([cx, y2, lendth], dtype=np.float64)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xywh(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        # Record observation for GPR history
        xywh = self.xywh
        self.gpr_obs.append((frame_id, xywh[0], xywh[1], xywh[2], xywh[3]))
        if len(self.gpr_obs) > STrack.gpr_maxlen:
            self.gpr_obs.pop(0)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh)
        )

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        # Record observation for GPR history
        xywh = self.xywh
        self.gpr_obs.append((frame_id, xywh[0], xywh[1], xywh[2], xywh[3]))
        if len(self.gpr_obs) > STrack.gpr_maxlen:
            self.gpr_obs.pop(0)

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)  # np.kron
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc_history(stracks, H=np.eye(2, 3)):
        """
        Apply the same affine warp used for KF means to the GPR observation
        history of each track, so historical (cx, cy) stay in the same
        coordinate system as the current frame.

        Only the position (cx, cy) is warped; w and h are scale-invariant
        for the affine warp used by SparseTrack's GMC.
        """
        if len(stracks) == 0:
            return
        R = H[:2, :2]  # (2, 2)
        t = H[:2, 2]  # (2,)
        for track in stracks:
            if len(track.gpr_obs) == 0:
                continue
            new_obs = []
            for fid, cx, cy, w, h in track.gpr_obs:
                pt = R @ np.array([cx, cy]) + t
                new_obs.append((fid, float(pt[0]), float(pt[1]), w, h))
            track.gpr_obs = new_obs

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @property
    # @jit(nopython=True)
    def deep_vec(self):
        """Convert bounding box to format `((top left, bottom right)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        cx = ret[0] + 0.5 * ret[2]
        y2 = ret[1] + ret[3]
        lendth = 2000 - y2
        return np.asarray([cx, y2, lendth], dtype=np.float64)

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class SparseTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args

        if "_half.json" in args.val_ann:
            self.det_thresh = args.track_thresh + 0.1
        else:
            self.det_thresh = args.track_thresh + 0.05

        self.pre_img = None
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.down_scale = args.down_scale
        self.layers = args.depth_levels
        self.mot20 = getattr(args, "mot20", False)
        self.use_diou = getattr(args, "use_diou", False)

        # GLRA config
        self.use_glra = getattr(args, "use_glra", False)
        self.gpr_history_len = getattr(args, "gpr_history_len", 30)
        self.glra_thresh = getattr(args, "glra_thresh", 0.7)
        self.gpr_min_lost = getattr(args, "gpr_min_lost", 2)
        self.gpr_max_lost = getattr(args, "gpr_max_lost", 5)
        self.gpr_min_obs = getattr(args, "gpr_min_obs", 3)
        self.glra_sigma_cap = getattr(args, "glra_sigma_cap", None)
        self.use_gmc_history = getattr(args, "use_gmc_history", False)
        STrack.gpr_maxlen = self.gpr_history_len
        # ── GLRA delayed confirmation ────────────────────────────────────
        self.glra_confirm = getattr(args, "glra_confirm", False)
        # 一致性檢查門檻:外插多一幀,稍微放寬 base thresh
        self.glra_confirm_thresh = getattr(
            args, "glra_confirm_thresh", min(self.glra_thresh + 0.1, 0.6)
        )
        self.glra_pending_recs = []  # 上一幀救回、等待確認的紀錄
        self.glra_confirm_grace = getattr(args, "glra_confirm_grace", 0)
        self.retro_outputs = []  # 確認成功後要回填的 (frame, tlwh, id, score)
        # GLRA per-sequence stats
        self.glra_stats = {
            "attempts": 0,
            "recovered": 0,
            "sigmas": [],
            "lost_frames": [],
            "confirmed": 0,
            "reverted": 0,
            "reverted_unmatched": 0,  # 沒被再次匹配而回滾
            "reverted_inconsistent": 0,  # 有匹配但軌跡不一致而回滾
            "confirm_costs_pass": [],  # 通過者的 1-DIoU
            "confirm_costs_fail": [],  # 被打掉者的 1-DIoU
        }
        # ── 路線一:候選品質 gates ───────────────────────────────────────
        # fragment gate: 候選 det 被任一 active track 包含(交集/det面積)超過
        #   containment 門檻 → 視為遮擋碎片,剔除。用 containment 而非 IoU,
        #   因碎片面積小、IoU 不高但幾乎整個落在遮擋者框內。
        self.glra_frag_gate = getattr(args, "glra_frag_gate", False)
        self.glra_frag_contain = getattr(args, "glra_frag_contain", 0.7)
        # height gate: 候選 det 高度與 GPR 預測高度偏差超過比例 → 剔除。
        #   行人遮擋碎片多是下半身被切,高度縮水是最穩定的訊號。
        self.glra_height_gate = getattr(args, "glra_height_gate", False)
        self.glra_height_tol = getattr(args, "glra_height_tol", 0.3)
        # density gate: GPR 預測點附近若有 >max_neighbors 個候選 det,
        #   association 視為歧義,跳過該 lost track。對應 MOT17-03 / DanceTrack
        #   val 的 dense scene 失敗模式 (FP spike from identity swap).
        self.glra_density_gate = getattr(args, "glra_density_gate", False)
        self.glra_density_expand = getattr(args, "glra_density_expand", 1.0)
        self.glra_density_max_neighbors = getattr(args, "glra_density_max_neighbors", 1)
        self.glra_stats_extra = {
            "gated_frag": 0,
            "gated_height_pairs": 0,
            "gated_density": 0,
        }

        # GLRA diagnostic logging.  When `glra_diag` is True, every committed
        # GLRA recovery appends one row of per-match features to
        # `glra_diag_path` (a single CSV across all sequences).  The per-row
        # `seq` column is taken from `glra_diag_seq` so an external eval
        # script can tag records by sequence by setting
        #     args.glra_diag_seq = "<seq_name>"
        # before each new tracker instance, OR by calling
        # `tracker.set_diag_seq(seq_name)` directly.
        self.glra_diag = getattr(args, "glra_diag", False)
        # GLRA figure-dump: when True, every committed GLRA recovery writes
        # one self-contained JSON (obs history + GPR pred + KF extrapolation
        # + matched detection) for offline plotting of Figure 4.
        self.glra_dump = getattr(args, "glra_dump", False)
        self.glra_dump_dir = getattr(args, "glra_dump_dir", "./glra_cases")
        self.glra_diag_path = getattr(args, "glra_diag_path", "./glra_diag.csv")
        self.glra_diag_seq = getattr(args, "glra_diag_seq", "unknown")
        self._glra_diag_header_written = False

        # GLRA uncertainty-adaptive threshold config
        # When enabled, each lost track's matching threshold is relaxed
        # proportional to GPR predictive uncertainty (σ in pixels):
        #   sigma_penalty = clip(σ / glra_sigma_scale, 0, 1) * glra_thresh_range
        #   adaptive_thresh_i = min(glra_thresh + sigma_penalty, glra_max_thresh)
        # Rationale: high σ → prediction unreliable → loosen gate so a
        # spatially displaced re-appearance can still be recovered.
        # Low σ → confident prediction → keep gate tight to avoid FP.
        self.glra_adaptive = getattr(args, "glra_adaptive", False)
        self.glra_sigma_scale = getattr(args, "glra_sigma_scale", 30.0)
        self.glra_thresh_range = getattr(args, "glra_thresh_range", 0.25)
        self.glra_max_thresh = getattr(args, "glra_max_thresh", 0.85)

        # ── GLRA tiered (per-lost_duration) threshold ───────────────────────
        # Diagnostic on DanceTrack val revealed that the cost band [0.35, 0.45]
        # (where GLRA is more lenient than DCM stage 2's 0.35) contains
        # disproportionately many identity-swap recoveries, and that wrong
        # rate rises sharply with `lost_duration` (lost=1: ~16%; lost≥4: ~45%).
        # `glra_tiered_thresh` lets the caller use different match thresholds
        # at different lost_durations, e.g. {1: 0.45, 2: 0.35, 3: 0.35}.
        # Tracks with lost_duration not in the dict fall back to
        # `glra_tiered_fallback`; if that is None, such tracks are not matched.
        # Overrides `glra_adaptive` when set.
        self.glra_tiered_thresh = getattr(args, "glra_tiered_thresh", None)
        self.glra_tiered_fallback = getattr(args, "glra_tiered_fallback", None)

        # ── TBD-stage dump (for slide figures) ──────────────────────────────
        # Set STAGE_DUMP_FRAMES via env, e.g. STAGE_DUMP_FRAMES="173" or "170,173,176"
        import os as _os

        _f = _os.environ.get("STAGE_DUMP_FRAMES", "")
        self._stage_dump_frames = {int(x) for x in _f.split(",") if x.strip()}
        self._stage_dump = []  # list of per-frame records
        self._stage_cur = None  # record being built this frame

    # ── GLRA diagnostic logging ─────────────────────────────────────────────
    def set_diag_seq(self, seq_name):
        """Tag subsequent GLRA recoveries with this sequence name in the
        diagnostic CSV.  Call once per sequence from the eval loop."""
        self.glra_diag_seq = str(seq_name)

    _GLRA_DIAG_FIELDS = [
        "seq",
        "frame",
        "track_id",
        "lost_duration",
        "gpr_sigma_px",
        "cost",
        "track_recent_speed",
        "implied_speed",
        "motion_ratio",
        "nearest_active_iou",
        "nearest_active_dist",
        "n_neighbors",
        "track_last_cx",
        "track_last_cy",
        "track_last_w",
        "track_last_h",
        "pred_cx",
        "pred_cy",
        "pred_w",
        "pred_h",
        "det_cx",
        "det_cy",
        "det_w",
        "det_h",
        "det_score",
    ]

    def _record_glra_diag(
        self,
        track,
        det,
        pred_tlbr,
        sigma_px,
        cost,
        active_tracks,
        glra_dets,
    ):
        """Append one row of per-recovery diagnostic features to the diag CSV.

        Captured BEFORE the GLRA update commits, so track.tlbr / track.gpr_obs
        reflect the pre-recovery state (last seen position, observation history).
        Features chosen to test four failure hypotheses:
          - fragment-of-active     → nearest_active_iou / nearest_active_dist
          - motion-implausible     → motion_ratio
          - long-horizon GPR drift → lost_duration × gpr_sigma_px
          - high-cost squeeze-in   → cost (relative to glra_thresh)
        """
        import os
        import csv

        # Track features (pre-update)
        if track.gpr_obs:
            last_fid, last_cx, last_cy, last_w, last_h = track.gpr_obs[-1]
        else:
            last_cx = last_cy = last_w = last_h = 0.0
        lost_duration = self.frame_id - track.end_frame

        # Recent motion speed (px/frame), averaged over up to 5 most recent obs
        recent_speed = 0.0
        if len(track.gpr_obs) >= 2:
            recent = track.gpr_obs[-5:]
            speeds = []
            for i in range(1, len(recent)):
                df = recent[i][0] - recent[i - 1][0]
                if df <= 0:
                    continue
                dx = recent[i][1] - recent[i - 1][1]
                dy = recent[i][2] - recent[i - 1][2]
                speeds.append(((dx * dx + dy * dy) ** 0.5) / df)
            if speeds:
                recent_speed = float(np.mean(speeds))

        det_tlbr = np.asarray(det.tlbr, dtype=np.float64)
        det_cx = float((det_tlbr[0] + det_tlbr[2]) / 2.0)
        det_cy = float((det_tlbr[1] + det_tlbr[3]) / 2.0)
        det_w = float(det_tlbr[2] - det_tlbr[0])
        det_h = float(det_tlbr[3] - det_tlbr[1])

        # Implied jump from track's last seen position to the matched detection
        dist_to_det = ((det_cx - last_cx) ** 2 + (det_cy - last_cy) ** 2) ** 0.5
        implied_speed = dist_to_det / max(lost_duration, 1)
        motion_ratio = implied_speed / max(recent_speed, 1e-6)

        # Competition from active tracks already updated this frame
        nearest_active_iou = 0.0
        nearest_active_dist = -1.0
        if active_tracks:
            active_tlbrs = np.asarray([t.tlbr for t in active_tracks], dtype=np.float64)
            iou_row = ious(det_tlbr[None, :], active_tlbrs)[0]
            nearest_active_iou = float(iou_row.max())
            act_cx = (active_tlbrs[:, 0] + active_tlbrs[:, 2]) / 2.0
            act_cy = (active_tlbrs[:, 1] + active_tlbrs[:, 3]) / 2.0
            d = np.sqrt((act_cx - det_cx) ** 2 + (act_cy - det_cy) ** 2)
            nearest_active_dist = float(d.min())

        # Candidate-pool density at the GPR prediction
        if glra_dets:
            glra_det_tlbrs = np.asarray([d.tlbr for d in glra_dets], dtype=np.float64)
        else:
            glra_det_tlbrs = np.zeros((0, 4), dtype=np.float64)
        n_neighbors = count_candidates_in_box(
            pred_tlbr, glra_det_tlbrs, expand=self.glra_density_expand
        )

        pred_cx = float((pred_tlbr[0] + pred_tlbr[2]) / 2.0)
        pred_cy = float((pred_tlbr[1] + pred_tlbr[3]) / 2.0)
        pred_w = float(pred_tlbr[2] - pred_tlbr[0])
        pred_h = float(pred_tlbr[3] - pred_tlbr[1])

        row = {
            "seq": self.glra_diag_seq,
            "frame": int(self.frame_id),
            "track_id": int(track.track_id),
            "lost_duration": int(lost_duration),
            "gpr_sigma_px": float(sigma_px),
            "cost": float(cost),
            "track_recent_speed": float(recent_speed),
            "implied_speed": float(implied_speed),
            "motion_ratio": float(motion_ratio),
            "nearest_active_iou": float(nearest_active_iou),
            "nearest_active_dist": float(nearest_active_dist),
            "n_neighbors": int(n_neighbors),
            "track_last_cx": float(last_cx),
            "track_last_cy": float(last_cy),
            "track_last_w": float(last_w),
            "track_last_h": float(last_h),
            "pred_cx": pred_cx,
            "pred_cy": pred_cy,
            "pred_w": pred_w,
            "pred_h": pred_h,
            "det_cx": det_cx,
            "det_cy": det_cy,
            "det_w": det_w,
            "det_h": det_h,
            "det_score": float(det.score),
        }

        diag_dir = os.path.dirname(self.glra_diag_path)
        if diag_dir:
            os.makedirs(diag_dir, exist_ok=True)
        needs_header = (
            not os.path.exists(self.glra_diag_path)
            or os.path.getsize(self.glra_diag_path) == 0
        )
        with open(self.glra_diag_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._GLRA_DIAG_FIELDS)
            if needs_header:
                w.writeheader()
            w.writerow(row)

    # ── GLRA figure dump ─────────────────────────────────────────────────────
    def _dump_glra_case(self, track, det, pred_tlbr, sigma_px, cost):
        """Write one self-contained JSON per GLRA recovery for offline plotting.

        Captured BEFORE the update commits, so track.gpr_obs is the pre-loss
        observation history and track.mean is the KF state propagated at
        constant velocity through the lost interval (its cx, cy is the KF's
        uncorrected position prediction at the current frame).
        """
        import os, json

        os.makedirs(self.glra_dump_dir, exist_ok=True)

        # observation history: list of [frame, cx, cy, w, h]
        obs = [
            [int(f), float(cx), float(cy), float(w), float(h)]
            for (f, cx, cy, w, h) in track.gpr_obs
        ]

        # KF constant-velocity extrapolation at current frame == track.mean[:2].
        # (During lost frames the tracker propagates mean without correction.)
        if track.mean is not None:
            kf_cx, kf_cy = float(track.mean[0]), float(track.mean[1])
            kf_w, kf_h = float(track.mean[2]), float(track.mean[3])
        else:
            kf_cx = kf_cy = kf_w = kf_h = None

        det_tlbr = np.asarray(det.tlbr, dtype=np.float64)
        rec = {
            "seq": self.glra_diag_seq,
            "frame": int(self.frame_id),
            "track_id": int(track.track_id),
            "lost_duration": int(self.frame_id - track.end_frame),
            "end_frame": int(track.end_frame),  # last observed frame
            "obs": obs,  # green history
            "gpr_pred": {  # blue GPR prediction
                "cx": float((pred_tlbr[0] + pred_tlbr[2]) / 2.0),
                "cy": float((pred_tlbr[1] + pred_tlbr[3]) / 2.0),
                "w": float(pred_tlbr[2] - pred_tlbr[0]),
                "h": float(pred_tlbr[3] - pred_tlbr[1]),
                "sigma_px": float(sigma_px),
            },
            "kf_pred": {  # red KF extrapolation
                "cx": kf_cx,
                "cy": kf_cy,
                "w": kf_w,
                "h": kf_h,
            },
            "det": {  # magenta matched det
                "cx": float((det_tlbr[0] + det_tlbr[2]) / 2.0),
                "cy": float((det_tlbr[1] + det_tlbr[3]) / 2.0),
                "w": float(det_tlbr[2] - det_tlbr[0]),
                "h": float(det_tlbr[3] - det_tlbr[1]),
                "score": float(det.score),
            },
            "cost": float(cost),
        }
        fname = f"{self.glra_diag_seq}_t{track.track_id}_f{self.frame_id}.json"
        with open(os.path.join(self.glra_dump_dir, fname), "w") as f:
            json.dump(rec, f, indent=2)

    # ────────────────────────────────────────────────────────────────────────

    def get_deep_range(self, obj, step):
        col = []
        for t in obj:
            lend = (t.deep_vec)[2]
            col.append(lend)
        max_len, mix_len = max(col), min(col)
        if max_len != mix_len:
            deep_range = np.arange(mix_len, max_len, (max_len - mix_len + 1) / step)
            if deep_range[-1] < max_len:
                deep_range = np.concatenate(
                    [
                        deep_range,
                        np.array(
                            [max_len],
                        ),
                    ]
                )
                deep_range[0] = np.floor(deep_range[0])
                deep_range[-1] = np.ceil(deep_range[-1])
        else:
            deep_range = [
                mix_len,
            ]
        mask = self.get_sub_mask(deep_range, col)
        return mask

    def get_sub_mask(self, deep_range, col):
        mix_len = deep_range[0]
        max_len = deep_range[-1]
        if max_len == mix_len:
            lc = mix_len
        mask = []
        for d in deep_range:
            if d > deep_range[0] and d < deep_range[-1]:
                mask.append((col >= lc) & (col < d))
                lc = d
            elif d == deep_range[-1]:
                mask.append((col >= lc) & (col <= d))
                lc = d
            else:
                lc = d
                continue
        return mask

    def DCM(
        self,
        detections,
        tracks,
        activated_starcks,
        refind_stracks,
        levels,
        thresh,
        is_fuse,
    ):
        if len(detections) > 0:
            det_mask = self.get_deep_range(detections, levels)
        else:
            det_mask = []

        if len(tracks) != 0:
            track_mask = self.get_deep_range(tracks, levels)
        else:
            track_mask = []

        u_detection, u_tracks, res_det, res_track = [], [], [], []
        if len(track_mask) != 0:
            if len(track_mask) < len(det_mask):
                for i in range(len(det_mask) - len(track_mask)):
                    idx = np.argwhere(det_mask[len(track_mask) + i] == True)
                    for idd in idx:
                        res_det.append(detections[idd[0]])
            elif len(track_mask) > len(det_mask):
                for i in range(len(track_mask) - len(det_mask)):
                    idx = np.argwhere(track_mask[len(det_mask) + i] == True)
                    for idd in idx:
                        res_track.append(tracks[idd[0]])

            for dm, tm in zip(det_mask, track_mask):
                det_idx = np.argwhere(dm == True)
                trk_idx = np.argwhere(tm == True)

                # search det
                det_ = []
                for idd in det_idx:
                    det_.append(detections[idd[0]])
                det_ = det_ + u_detection
                # search trk
                track_ = []
                for idt in trk_idx:
                    track_.append(tracks[idt[0]])
                # update trk
                track_ = track_ + u_tracks

                dists = (
                    diou_distance(track_, det_)
                    if self.use_diou
                    else iou_distance(track_, det_)
                )
                if (not self.args.mot20) and is_fuse:
                    dists = fuse_score(dists, det_)
                matches, u_track_, u_det_ = linear_assignment(dists, thresh)
                for itracked, idet in matches:
                    track = track_[itracked]
                    det = det_[idet]
                    if track.state == TrackState.Tracked:
                        track.update(det_[idet], self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)
                u_tracks = [track_[t] for t in u_track_]
                u_detection = [det_[t] for t in u_det_]

            u_tracks = u_tracks + res_track
            u_detection = u_detection + res_det

        else:
            u_detection = detections

        return activated_starcks, refind_stracks, u_tracks, u_detection

    def update(self, output_results, curr_img=None):
        self.frame_id += 1
        if self.frame_id == 1:
            self.pre_img = None

        # init stracks
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        # current detections
        bboxes = output_results.pred_boxes.tensor.cpu().numpy()  # x1y1x2y2
        scores = output_results.scores.cpu().numpy()

        # divide high-score dets and low-scores dets
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        # ── stage dump: DETECT (high-score dets, x1y1x2y2, no IDs) ──────────
        if self.frame_id in self._stage_dump_frames:
            self._stage_cur = {
                "frame": int(self.frame_id),
                "detect": [[float(v) for v in box] for box in dets],
                "predict": [],
                "update": [],
            }
        else:
            self._stage_cur = None

        # tracks preprocess
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # init high-score dets
        if len(dets) > 0:
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s)
                for (tlbr, s) in zip(dets, scores_keep)
            ]
        else:
            detections = []
        # get strack_pool
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # predict the current location with KF
        STrack.multi_predict(strack_pool)
        # ── stage dump: PREDICT (KF-predicted boxes, per-track color) ───────
        # Captured before any matching, so track_id is last frame's identity.
        if self._stage_cur is not None:
            self._stage_cur["predict"] = [
                [float(v) for v in t.tlbr] + [int(t.track_id)] for t in strack_pool
            ]

        if not self.mot20:
            # use GMC: for mot20 dancetrack--unenabled GMC: 368 - 373
            if self.pre_img is not None:
                warp = pbcvt.GMC(curr_img, self.pre_img, self.down_scale)
            else:
                warp = np.eye(3, 3)
            STrack.multi_gmc(strack_pool, warp[:2, :])
            STrack.multi_gmc(unconfirmed, warp[:2, :])

            # Compensate GPR observation history for camera motion
            if self.use_glra and self.use_gmc_history and self.pre_img is not None:
                STrack.multi_gmc_history(strack_pool, warp[:2, :])
            # Warp pending-recovery backups so a rollback at t+1 lands in
            # the current frame's coordinate system (matters for 05/10/11/13)
            if self.glra_confirm and self.glra_pending_recs:
                H = warp[:2, :]
                R = H[:2, :2]
                R8x8 = np.kron(np.eye(4, dtype=float), R)
                t_vec = H[:2, 2]
                for rec in self.glra_pending_recs:
                    b = rec["backup"]
                    mean = R8x8.dot(b["mean"])
                    mean[:2] += t_vec
                    b["mean"] = mean
                    b["covariance"] = R8x8.dot(b["covariance"]).dot(R8x8.T)
                    b["gpr_obs"] = [
                        (
                            fid,
                            R[0, 0] * cx + R[0, 1] * cy + t_vec[0],
                            R[1, 0] * cx + R[1, 1] * cy + t_vec[1],
                            w,
                            h,
                        )
                        for fid, cx, cy, w, h in b["gpr_obs"]
                    ]

        # DCM
        activated_starcks, refind_stracks, u_track, u_detection_high = self.DCM(
            detections,
            strack_pool,
            activated_starcks,
            refind_stracks,
            self.layers,
            self.args.match_thresh,
            is_fuse=True,
        )

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s)
                for (tlbr, s) in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [t for t in u_track if t.state == TrackState.Tracked]

        # DCM
        activated_starcks, refind_stracks, u_strack, u_detection_sec = self.DCM(
            detections_second,
            r_tracked_stracks,
            activated_starcks,
            refind_stracks,
            self.args.depth_levels_low,
            0.3,
            is_fuse=False,
        )
        for track in u_strack:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        gpr_stracks = joint_stracks(
            self.lost_stracks, lost_stracks
        )  # stracks available for GLRA

        def _lost_frames(track):
            return self.frame_id - track.end_frame

        gpr_stracks = [
            t
            for t in gpr_stracks
            if self.gpr_min_lost <= _lost_frames(t) <= self.gpr_max_lost
            and not getattr(t, "glra_pending", False)
        ]

        # ── GLRA: GPR Lost-track Re-Association ──────────────────────────────
        # Try to recover unmatched lost tracks using GPR trajectory prediction
        # against leftover low-score detections from second DCM.
        if self.use_glra and len(gpr_stracks) > 0 and len(u_detection_sec) > 0:
            # ── 路線一:候選品質 gates ──────────────────────────────────
            glra_dets = u_detection_sec
            if self.glra_frag_gate or self.glra_height_gate:
                # 遮擋者 = 本幀仍存活的 active track(剛 update 過的)
                obstacle_tlbrs = [
                    t.tlbr for t in activated_starcks if t.state == TrackState.Tracked
                ]
                # 每個候選 lost track 的 GPR 預測高度(供 height gate)
                pred_h = {}
                if self.glra_height_gate:
                    for t in gpr_stracks:
                        p = gpr_predict_bbox(t, self.frame_id, self.gpr_min_obs)
                        if p is not None:
                            ptlbr, _ = p
                            pred_h[t.track_id] = max(ptlbr[3] - ptlbr[1], 1.0)

                kept = []
                for det in glra_dets:
                    # fragment gate
                    if self.glra_frag_gate and len(obstacle_tlbrs) > 0:
                        c = containment(det.tlbr, obstacle_tlbrs)
                        if c.max() >= self.glra_frag_contain:
                            self.glra_stats_extra["gated_frag"] += 1
                            continue
                    kept.append(det)
                glra_dets = kept

            if len(glra_dets) == 0:
                glra_matches, u_glra_tracks = [], list(range(len(gpr_stracks)))
            else:
                glra_dists, effective_thresh = glra_distance(
                    gpr_stracks,
                    glra_dets,
                    self.frame_id,
                    use_diou=self.use_diou,
                    min_obs=self.gpr_min_obs,
                    sigma_cap=self.glra_sigma_cap,
                    # Adaptive mode: pass base_thresh so glra_distance computes
                    # per-track sigma_penalty and pre-gates the cost matrix.
                    # effective_thresh (max adaptive threshold used) is returned
                    # and passed to linear_assignment as cost_limit.
                    base_thresh=self.glra_thresh if self.glra_adaptive else None,
                    sigma_scale=self.glra_sigma_scale,
                    thresh_range=self.glra_thresh_range,
                    max_thresh=self.glra_max_thresh,
                    # Density gate: skip lost tracks whose GPR-predicted
                    # neighborhood is too crowded with candidate detections
                    # (ambiguous → high FP risk).
                    density_gate=self.glra_density_gate,
                    density_expand=self.glra_density_expand,
                    density_max_neighbors=self.glra_density_max_neighbors,
                    density_stats=self.glra_stats_extra,
                    # Tiered per-lost_duration threshold (overrides adaptive).
                    tiered_thresh=self.glra_tiered_thresh,
                    tiered_fallback=self.glra_tiered_fallback,
                )
                # height gate:把高度不一致的 (track, det) 配對成本設為無窮大
                if self.glra_height_gate and len(pred_h) > 0:
                    for ti, t in enumerate(gpr_stracks):
                        ph = pred_h.get(t.track_id)
                        if ph is None:
                            continue
                        for di, det in enumerate(glra_dets):
                            dh = det.tlbr[3] - det.tlbr[1]
                            if abs(dh - ph) / ph > self.glra_height_tol:
                                glra_dists[ti, di] = np.inf
                                self.glra_stats_extra["gated_height_pairs"] += 1
                match_thresh = (
                    effective_thresh
                    if (self.glra_adaptive or self.glra_tiered_thresh is not None)
                    else self.glra_thresh
                )
                glra_matches, u_glra_tracks, _ = linear_assignment(
                    glra_dists, match_thresh
                )

            self.glra_stats["attempts"] += len(gpr_stracks)
            self.glra_stats["recovered"] += len(glra_matches)
            for itracked, idet in glra_matches:
                t = gpr_stracks[itracked]
                self.glra_stats["lost_frames"].append(self.frame_id - t.end_frame)

            # ── Diagnostic logging (before commit) ───────────────────────
            # Recompute pred_tlbr / sigma per matched track so we can record
            # them with the recovery row.  Only when --glra-diag is set.
            if self.glra_diag and len(glra_matches) > 0:
                for itracked, idet in glra_matches:
                    track = gpr_stracks[itracked]
                    det = glra_dets[idet]
                    pred = gpr_predict_bbox(track, self.frame_id, self.gpr_min_obs)
                    if pred is None:
                        continue
                    pred_tlbr, sigma_px = pred
                    cost = float(glra_dists[itracked, idet])
                    self._record_glra_diag(
                        track,
                        det,
                        pred_tlbr,
                        sigma_px,
                        cost,
                        active_tracks=activated_starcks,
                        glra_dets=glra_dets,
                    )

            # ── Figure dump (before commit) ──────────────────────────────
            if self.glra_dump and len(glra_matches) > 0:
                for itracked, idet in glra_matches:
                    track = gpr_stracks[itracked]
                    det = glra_dets[idet]
                    pred = gpr_predict_bbox(track, self.frame_id, self.gpr_min_obs)
                    if pred is None:
                        continue
                    pred_tlbr, sigma_px = pred
                    cost = float(glra_dists[itracked, idet])
                    self._dump_glra_case(track, det, pred_tlbr, sigma_px, cost)

            # Remove already-marked-lost tracks that GLRA recovers
            recovered_ids = {gpr_stracks[i].track_id for i, _ in glra_matches}
            lost_stracks = [t for t in lost_stracks if t.track_id not in recovered_ids]

            for itracked, idet in glra_matches:
                track = gpr_stracks[itracked]
                det = glra_dets[idet]

                backup = None
                if self.glra_confirm:
                    # 救援前狀態快照(供回滾)
                    backup = dict(
                        mean=track.mean.copy(),
                        covariance=track.covariance.copy(),
                        gpr_obs=list(track.gpr_obs),
                        frame_id=track.frame_id,
                        score=track.score,
                        tracklet_len=track.tracklet_len,
                    )

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

                if self.glra_confirm:
                    track.glra_pending = True
                    self.glra_pending_recs.append(
                        dict(
                            track=track,
                            backup=backup,
                            out_tlwh=track.tlwh.copy(),
                            out_score=track.score,
                            recovered_frame=self.frame_id,
                        )
                    )
        # ─────────────────────────────────────────────────────────────────────

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [d for d in u_detection_high]
        dists = (
            diou_distance(unconfirmed, detections)
            if self.use_diou
            else iou_distance(unconfirmed, detections)
        )
        if not self.args.mot20:
            dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(
            dists, thresh=self.args.confirm_thresh
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # get scores of lost tracks
        # ── GLRA delayed confirmation: resolve last frame's pendings ─────
        if self.glra_confirm and self.glra_pending_recs:
            still_pending = []
            for rec in self.glra_pending_recs:
                if rec["recovered_frame"] == self.frame_id:
                    still_pending.append(rec)  # 本幀新增的,下一幀再結算
                    continue

                track = rec["track"]
                matched_again = (
                    track.frame_id == self.frame_id
                    and track.state == TrackState.Tracked
                )

                consistent = True
                confirm_cost = None
                if matched_again:
                    # 軌跡一致性:用「救援前」的歷史外插到本幀,
                    # 防止錯接到別人身上後持續吃對方偵測(identity theft)
                    shim = type("S", (), {"gpr_obs": rec["backup"]["gpr_obs"]})()
                    pred = gpr_predict_bbox(shim, self.frame_id, self.gpr_min_obs)
                    if pred is not None:
                        pred_tlbr, _ = pred
                        sim = dious(
                            pred_tlbr[None, :].astype(np.float64),
                            track.tlbr[None, :].astype(np.float64),
                        )[0, 0]
                        confirm_cost = 1.0 - sim
                        consistent = confirm_cost <= self.glra_confirm_thresh

                if matched_again and consistent:
                    track.glra_pending = False
                    self.glra_stats["confirmed"] += 1
                    if confirm_cost is not None:
                        self.glra_stats["confirm_costs_pass"].append(confirm_cost)
                    self.retro_outputs.append(
                        (
                            rec["recovered_frame"],
                            rec["out_tlwh"],
                            track.track_id,
                            rec["out_score"],
                        )
                    )
                elif (
                    not matched_again
                    and (self.frame_id - rec["recovered_frame"])
                    <= self.glra_confirm_grace
                ):
                    # 未匹配但還在寬限期:track 已被正常流程 mark_lost,
                    # 留在 lost pool 等下一幀的 stage-1 關聯,先不回滾
                    still_pending.append(rec)
                else:
                    # 回滾:KF / GPR 歷史 / 時間戳全部還原,救援視同未發生
                    if not matched_again:
                        self.glra_stats["reverted_unmatched"] += 1
                    else:
                        self.glra_stats["reverted_inconsistent"] += 1
                        if confirm_cost is not None:
                            self.glra_stats["confirm_costs_fail"].append(confirm_cost)
                    b = rec["backup"]
                    track.mean = b["mean"]
                    track.covariance = b["covariance"]
                    track.gpr_obs = b["gpr_obs"]
                    track.frame_id = b["frame_id"]  # end_frame 跟著還原
                    track.score = b["score"]
                    track.tracklet_len = b["tracklet_len"]
                    track.glra_pending = False
                    track.mark_lost()
                    self.glra_stats["reverted"] += 1
                    self.tracked_stracks = [
                        t for t in self.tracked_stracks if t.track_id != track.track_id
                    ]
                    if all(t.track_id != track.track_id for t in self.lost_stracks):
                        self.lost_stracks.append(track)
            self.glra_pending_recs = still_pending
        output_stracks = [
            track
            for track in self.tracked_stracks
            if track.is_activated and not getattr(track, "glra_pending", False)
        ]
        # ── stage dump: UPDATE (final tracks) + flush record ────────────────
        if self._stage_cur is not None:
            self._stage_cur["update"] = [
                [float(v) for v in t.tlbr] + [int(t.track_id)] for t in output_stracks
            ]
            self._stage_dump.append(self._stage_cur)
            self._stage_cur = None
        self.pre_img = curr_img
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())
