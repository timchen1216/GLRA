import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
import math
from cython_bbox import bbox_overlaps as bbox_ious
from tracker import kalman_filter
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64),
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def dious(atlbrs, btlbrs):
    """
    Compute DIoU between two sets of boxes in tlbr format (vectorized).
    DIoU = IoU - rho^2(center_a, center_b) / c^2
    where c is the diagonal of the smallest enclosing box.
    """
    atlbrs = np.asarray(atlbrs, dtype=np.float64)
    btlbrs = np.asarray(btlbrs, dtype=np.float64)
    result = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if result.size == 0:
        return result

    a = atlbrs[:, None, :]  # (n, 1, 4)
    b = btlbrs[None, :, :]  # (1, m, 4)

    # Intersection
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)

    # Squared center distance
    ca = (atlbrs[:, :2] + atlbrs[:, 2:]) / 2.0  # (n, 2)
    cb = (btlbrs[:, :2] + btlbrs[:, 2:]) / 2.0  # (m, 2)
    rho2 = np.sum((ca[:, None, :] - cb[None, :, :]) ** 2, axis=-1)

    # Squared diagonal of smallest enclosing box
    ex1 = np.minimum(a[..., 0], b[..., 0])
    ey1 = np.minimum(a[..., 1], b[..., 1])
    ex2 = np.maximum(a[..., 2], b[..., 2])
    ey2 = np.maximum(a[..., 3], b[..., 3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2

    diou = iou - np.where(c2 > 0, rho2 / c2, 0.0)
    return diou


def containment(det_tlbr, obstacle_tlbrs):
    """每個 obstacle 對 det 的包含度 = 交集面積 / det 面積。回傳 shape [N_obstacle]."""
    if len(obstacle_tlbrs) == 0:
        return np.zeros(0, dtype=np.float64)
    d = det_tlbr
    da = max((d[2] - d[0]), 0.0) * max((d[3] - d[1]), 0.0)
    if da <= 0:
        return np.zeros(len(obstacle_tlbrs), dtype=np.float64)
    obs = np.asarray(obstacle_tlbrs, dtype=np.float64)
    ix1 = np.maximum(d[0], obs[:, 0])
    iy1 = np.maximum(d[1], obs[:, 1])
    ix2 = np.minimum(d[2], obs[:, 2])
    iy2 = np.minimum(d[3], obs[:, 3])
    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    return (iw * ih) / da


def diou_distance(atracks, btracks):
    """
    Compute cost based on DIoU.
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _dious = dious(np.asarray(atlbrs), np.asarray(btlbrs))
    cost_matrix = np.clip(1.0 - _dious, 0.0, 1.0)
    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric="cosine"):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray(
        [track.curr_feat for track in detections], dtype=np.float64
    )
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray(
        [track.smooth_feat for track in tracks], dtype=np.float64
    )
    cost_matrix = np.maximum(
        0.0, cdist(track_features, det_features, metric)
    )  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric="maha"
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def greedy_assignment_iou(dist, thresh):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < thresh:
            dist[:, j] = 1.0
            matched_indices.append([j, i])
    return np.array(matched_indices, np.int32).reshape(-1, 2)


def greedy_assignment(dists, threshs):
    matches = greedy_assignment_iou(dists.T, threshs)
    u_det = [d for d in range(dists.shape[1]) if not (d in matches[:, 1])]
    u_track = [d for d in range(dists.shape[0]) if not (d in matches[:, 0])]
    return matches, u_track, u_det


def fuse_score_matrix(cost_matrix, detections, tracks):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix

    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    trk_scores = np.array([trk.score for trk in tracks])
    trk_scores = np.expand_dims(trk_scores, axis=1).repeat(cost_matrix.shape[1], axis=1)
    mid_scores = (det_scores + trk_scores) / 2
    fuse_sim = iou_sim * mid_scores
    fuse_cost = 1 - fuse_sim

    return fuse_cost


def gpr_predict_bbox(track, target_frame, min_obs=3):
    """
    Fit GPR on track.gpr_obs history and predict bbox center at target_frame.
    Returns (pred_tlbr, sigma_px) or None if insufficient history.

    pred_tlbr : np.ndarray shape [4] – predicted bbox in tlbr format.
    sigma_px  : float – RMS positional uncertainty in pixels (back-transformed
                from z-score space), used by glra_distance for adaptive
                thresholding.

    Normalisation design
    ────────────────────
    Input  (frame index): min-max scaled to [0, 1] over the observation
           window.  DotProduct and RBF both operate on a consistent [0,1]
           domain regardless of absolute frame numbers, and the extrapolation
           target (> 1) stays on the same positive axis so DotProduct never
           crosses zero.

    Output (cx, cy): z-score normalised so the GP prior mean equals the
           data mean rather than collapsing to 0 on extrapolation.  The
           per-track std also acts as an implicit scale factor so that a
           single fixed noise_level=0.05 is effective across tracks with
           very different motion magnitudes.

    sigma back-transform: z-score std is multiplied by the original
           per-axis std and combined as RMS (geometrically correct) rather
           than a plain average, giving an unbiased pixel-space uncertainty.

    track.gpr_obs: list of (frame_id, cx, cy, w, h)
    """
    obs = track.gpr_obs
    if len(obs) < min_obs:
        return None

    frames = np.array([o[0] for o in obs], dtype=np.float64).reshape(-1, 1)
    cxs = np.array([o[1] for o in obs], dtype=np.float64)
    cys = np.array([o[2] for o in obs], dtype=np.float64)
    ws = np.array([o[3] for o in obs], dtype=np.float64)
    hs = np.array([o[4] for o in obs], dtype=np.float64)

    f_min, f_max = frames[0, 0], frames[-1, 0]
    if f_max == f_min:
        return None

    # ── Input: min-max scale to [0, 1] ──────────────────────────────────────
    frames_norm = (frames - f_min) / (f_max - f_min)
    target_norm = np.array([[(target_frame - f_min) / (f_max - f_min)]])

    # ── Output: z-score so prior mean = data mean ───────────────────────────
    cx_mean, cx_std = cxs.mean(), cxs.std() + 1e-6
    cy_mean, cy_std = cys.mean(), cys.std() + 1e-6
    cxs_z = (cxs - cx_mean) / cx_std
    cys_z = (cys - cy_mean) / cy_std

    kernel = RBF(length_scale=0.5) + WhiteKernel(noise_level=0.05)
    try:
        gpr_cx = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
        gpr_cy = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
        gpr_cx.fit(frames_norm, cxs_z)
        gpr_cy.fit(frames_norm, cys_z)

        pred_cx_z, sigma_cx_z = gpr_cx.predict(target_norm, return_std=True)
        pred_cy_z, sigma_cy_z = gpr_cy.predict(target_norm, return_std=True)

        pred_cx = float(pred_cx_z[0]) * cx_std + cx_mean
        pred_cy = float(pred_cy_z[0]) * cy_std + cy_mean

        # Back-transform sigma to pixel space and combine as RMS.
        # sigma_z * per_axis_std converts z-score uncertainty to pixels;
        # RMS over x/y is geometrically correct (vs plain average).
        sigma_px = float(
            np.sqrt(
                ((sigma_cx_z[0] * cx_std) ** 2 + (sigma_cy_z[0] * cy_std) ** 2) / 2.0
            )
        )
    except Exception:
        return None

    # Use recent average for w/h (size rarely changes drastically)
    pred_w = float(np.mean(ws[-3:]))
    pred_h = float(np.mean(hs[-3:]))

    pred_tlbr = np.array(
        [
            pred_cx - pred_w / 2.0,
            pred_cy - pred_h / 2.0,
            pred_cx + pred_w / 2.0,
            pred_cy + pred_h / 2.0,
        ],
        dtype=np.float64,
    )
    return pred_tlbr, sigma_px


def count_candidates_in_box(pred_tlbr, det_tlbrs, expand=1.0):
    """
    Count how many candidate detections have their center inside the
    predicted bounding box (optionally expanded around its center by `expand`).

    Used by the GLRA density gate: when multiple detection centers fall inside
    the GPR prediction's local neighborhood, the lost-track recovery is
    ambiguous (high risk of identity swap), and GLRA should be suppressed for
    that lost track.

    Why predicted bbox rather than σ or a fixed pixel radius
    ────────────────────────────────────────────────────────
    σ from GPR is small on slow / well-observed tracks (~5–30 px) and a σ-based
    radius is too tight in dense scenes where the failure actually occurs.
    A fixed pixel radius is not scale-invariant across MOT17 (1920×1080),
    MOT20 (very dense, full HD), and DanceTrack (1280×720 with smaller
    persons).  Using the predicted bbox directly anchors the gate to the
    object's own scale at that location, giving a single hyper-parameter
    (`expand`) that transfers across datasets.

    Args:
        pred_tlbr   : np.ndarray shape (4,) – predicted bbox (x1, y1, x2, y2).
        det_tlbrs   : np.ndarray shape (m, 4) – candidate bboxes (tlbr).
        expand      : float – multiplier on box w/h around its center.
                              1.0 = use predicted bbox as-is (≈ body width);
                              1.5–2.0 = slightly wider neighborhood.

    Returns:
        count : int – number of detections whose center lies inside the
                      (expanded) predicted bbox.
    """
    if det_tlbrs.size == 0:
        return 0
    cx = (pred_tlbr[0] + pred_tlbr[2]) / 2.0
    cy = (pred_tlbr[1] + pred_tlbr[3]) / 2.0
    half_w = (pred_tlbr[2] - pred_tlbr[0]) * expand / 2.0
    half_h = (pred_tlbr[3] - pred_tlbr[1]) * expand / 2.0
    x1, x2 = cx - half_w, cx + half_w
    y1, y2 = cy - half_h, cy + half_h

    det_cx = (det_tlbrs[:, 0] + det_tlbrs[:, 2]) / 2.0
    det_cy = (det_tlbrs[:, 1] + det_tlbrs[:, 3]) / 2.0
    inside = (det_cx >= x1) & (det_cx <= x2) & (det_cy >= y1) & (det_cy <= y2)
    return int(inside.sum())


def glra_distance(
    lost_tracks,
    detections,
    target_frame,
    use_diou=True,
    min_obs=3,
    base_thresh=None,
    sigma_scale=30.0,
    thresh_range=0.25,
    max_thresh=0.85,
    sigma_cap=None,
    density_gate=False,
    density_expand=1.0,
    density_max_neighbors=1,
    density_stats=None,
    tiered_thresh=None,
    tiered_fallback=None,
):
    """
    GLRA (GPR Lost-track Re-Association) cost matrix with optional
    uncertainty-adaptive thresholding.

    For each lost track, predict its bbox at target_frame via GPR, then compute
    IoU (or DIoU) distance against the candidate detections.
    Tracks with insufficient history get cost=1.0 (unmatchable).

    Adaptive thresholding (enabled when base_thresh is not None):
        GPR provides a predictive std σ (pixels) alongside the predicted bbox.
        A per-track penalty is computed as:
            sigma_penalty = clip(σ / sigma_scale, 0, 1) * thresh_range
            adaptive_thresh_i  = min(base_thresh + sigma_penalty, max_thresh)
        Cells where cost > adaptive_thresh_i are pre-gated to 1.0 so that
        linear_assignment cannot select them.  The caller should pass the
        returned `effective_thresh` as the cost_limit to linear_assignment so
        that loosened per-track gates (> base_thresh) are actually reachable.

    Args:
        lost_tracks  : list[STrack] – unmatched tracks after second DCM
        detections   : list[STrack] – unmatched low-score detections
        target_frame : int          – current frame_id
        use_diou     : bool         – use DIoU instead of IoU
        min_obs      : int          – minimum history length to attempt GPR
        base_thresh  : float|None   – base matching threshold; if None,
                                      adaptive mode is disabled and the cost
                                      matrix is returned un-gated (caller uses
                                      its own fixed threshold).
        sigma_scale  : float        – pixel std that maps to the full
                                      thresh_range bonus (default 30 px).
        thresh_range : float        – maximum extra threshold allowed above
                                      base_thresh (default 0.25).
        max_thresh   : float        – absolute ceiling for adaptive threshold
                                      (default 0.85).
        sigma_cap    : float|None   – if set, any track whose GPR σ exceeds
                                      this cap is left un-matchable (row=1.0).
        density_gate : bool         – if True, suppress GLRA for any lost
                                      track whose GPR-predicted neighborhood
                                      contains more than `density_max_neighbors`
                                      candidate detections (ambiguous match).
        density_expand : float      – multiplier on predicted bbox w/h used to
                                      define the local neighborhood
                                      (1.0 = predicted bbox as-is).
        density_max_neighbors : int – allowed number of candidate detections
                                      inside the neighborhood; >this triggers
                                      the gate.  Default 1 (1 candidate OK,
                                      2+ = ambiguous → skip).
        density_stats : dict|None   – if a dict with key "gated_density" is
                                      provided, gated-track counts accumulate
                                      into it (for ablation logging).
        tiered_thresh : dict|None   – per-`lost_duration` matching thresholds,
                                      e.g. ``{1: 0.45, 2: 0.35, 3: 0.35}``.
                                      Each lost track's threshold is looked up
                                      by its actual ``lost_duration`` (=
                                      ``target_frame - track.end_frame``).
                                      Cells above the per-track threshold are
                                      pre-gated to 1.0.  When set, the caller
                                      should pass ``effective_thresh`` as
                                      ``linear_assignment``'s ``cost_limit``
                                      so the widest per-track gate is reachable.
                                      Overrides ``base_thresh`` (no sigma penalty
                                      applied in tiered mode).
        tiered_fallback : float|None – threshold for lost_durations *not* in
                                      the ``tiered_thresh`` dict.  If ``None``
                                      (default), such tracks are left
                                      unmatchable (cost row stays at 1.0).
                                      Useful as a safety net so ``gpr_max_lost``
                                      is the only knob controlling the upper
                                      bound; or set explicitly to extend the
                                      tiered policy with a relaxed fallback.

    Returns:
        cost_matrix    : np.ndarray  shape (len(lost_tracks), len(detections))
        effective_thresh : float     – max adaptive threshold actually used;
                                       equals base_thresh when adaptive mode is
                                       disabled (pass this to linear_assignment).
    """
    cost_matrix = np.ones((len(lost_tracks), len(detections)), dtype=np.float64)
    if tiered_thresh is not None:
        # In tiered mode, the widest gate is unknown a priori; accumulate from
        # per-track thresholds during the loop.  Start at 0.0 so max() works.
        effective_thresh = 0.0
    else:
        effective_thresh = base_thresh if base_thresh is not None else 1.0

    if cost_matrix.size == 0:
        return cost_matrix, effective_thresh

    det_tlbrs = np.asarray([t.tlbr for t in detections], dtype=np.float64)

    # ── GPR prediction for each eligible lost track ──────────────────────────
    pred_tlbrs = []
    pred_sigmas = []
    valid_idx = []
    for i, track in enumerate(lost_tracks):
        result = gpr_predict_bbox(track, target_frame, min_obs)
        if result is not None:
            pred_tlbr, sigma_px = result
            pred_tlbrs.append(pred_tlbr)
            pred_sigmas.append(sigma_px)
            valid_idx.append(i)

    if len(pred_tlbrs) == 0:
        return cost_matrix, effective_thresh

    pred_arr = np.asarray(pred_tlbrs, dtype=np.float64)
    if use_diou:
        sim = dious(pred_arr, det_tlbrs)  # shape (k, m)
    else:
        sim = ious(pred_arr, det_tlbrs)  # shape (k, m)

    for local_i, global_i in enumerate(valid_idx):
        # print(
        #     f"GLRA sigma: {pred_sigmas[local_i]:.1f} px, lost {target_frame - lost_tracks[global_i].end_frame} frames"
        # )
        if sigma_cap is not None and pred_sigmas[local_i] > sigma_cap:
            continue

        # ── Density gate ─────────────────────────────────────────────
        # If multiple candidate detections fall inside the GPR-predicted
        # local neighborhood, the recovery is ambiguous (e.g. dense crowd,
        # close-interaction dance).  Suppressing GLRA for this track keeps
        # safe recoveries (isolated re-appearances) while avoiding the FP
        # spike observed on MOT17-03 and DanceTrack val.
        if density_gate:
            n_neighbors = count_candidates_in_box(
                pred_arr[local_i], det_tlbrs, expand=density_expand
            )
            if n_neighbors > density_max_neighbors:
                if density_stats is not None:
                    density_stats["gated_density"] = (
                        density_stats.get("gated_density", 0) + 1
                    )
                continue  # leave row at 1.0 (unmatchable)

        costs = np.clip(1.0 - sim[local_i], 0.0, 1.0)

        if tiered_thresh is not None:
            # ── Tiered per-track threshold (by lost_duration) ─────────────
            # Overrides the sigma-adaptive mode below: only one per-track
            # threshold policy can be active at a time.  The threshold is
            # looked up by the track's actual lost_duration; tracks whose
            # lost_duration is not in the dict fall back to `tiered_fallback`,
            # or are left un-matchable if fallback is None.
            ld = int(target_frame - lost_tracks[global_i].end_frame)
            tt = tiered_thresh.get(ld, tiered_fallback)
            if tt is None:
                continue  # leave row at 1.0 (unmatchable)
            effective_thresh = max(effective_thresh, tt)
            costs = np.where(costs <= tt, costs, 1.0)
        elif base_thresh is not None:
            # ── Uncertainty-adaptive per-track threshold ─────────────────
            sigma = pred_sigmas[local_i]
            sigma_penalty = np.clip(sigma / sigma_scale, 0.0, 1.0) * thresh_range
            adaptive_thresh_i = min(base_thresh + sigma_penalty, max_thresh)
            # Track the widest gate used so the caller can set cost_limit
            effective_thresh = max(effective_thresh, adaptive_thresh_i)
            # Pre-gate: any cell above the per-track threshold is set to 1.0
            # so linear_assignment cannot select it regardless of cost_limit.
            costs = np.where(costs <= adaptive_thresh_i, costs, 1.0)

        cost_matrix[global_i] = costs

    return cost_matrix, effective_thresh


def BIoU_distance(atracks, btracks, sigma=0.4):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    atlbrs, btlbrs = [], []
    for trk in atracks:
        x1, y1, w, h = trk.tlwh
        delta_h, delta_w = h * sigma, w * sigma
        x1_ = x1 - delta_w
        y1_ = y1 - delta_h
        x2_ = x1 + w + delta_w
        y2_ = y1 + h + delta_h
        bbox_new = np.array([x1_, y1_, x2_, y2_], dtype=np.float32)
        atlbrs.append(bbox_new)

    for trk in btracks:
        x1, y1, w, h = trk.tlwh
        delta_h, delta_w = h * sigma, w * sigma
        x1_ = x1 - delta_w
        y1_ = y1 - delta_h
        x2_ = x1 + w + delta_w
        y2_ = y1 + h + delta_h
        bbox_new = np.array([x1_, y1_, x2_, y2_], dtype=np.float32)
        btlbrs.append(bbox_new)

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix
