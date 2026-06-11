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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


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
    cost_matrix = 1 - _dious
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
    Returns predicted tlbr (np.ndarray shape [4]) or None if insufficient history.

    Both input (frame indices) and output (cx, cy) are z-score normalised so
    the GP prior mean stays at the data mean rather than collapsing to 0 on
    extrapolation.

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

    # ── Normalise inputs to [0, 1] ──────────────────────────────────────────
    frames_norm = (frames - f_min) / (f_max - f_min)
    target_norm = np.array([[(target_frame - f_min) / (f_max - f_min)]])

    # ── Normalise outputs (z-score) so prior mean = data mean ───────────────
    cx_mean, cx_std = cxs.mean(), cxs.std() + 1e-6
    cy_mean, cy_std = cys.mean(), cys.std() + 1e-6
    cxs_z = (cxs - cx_mean) / cx_std
    cys_z = (cys - cy_mean) / cy_std

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    try:
        gpr_cx = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
        gpr_cy = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
        gpr_cx.fit(frames_norm, cxs_z)
        gpr_cy.fit(frames_norm, cys_z)
        pred_cx = float(gpr_cx.predict(target_norm)[0]) * cx_std + cx_mean
        pred_cy = float(gpr_cy.predict(target_norm)[0]) * cy_std + cy_mean
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
    return pred_tlbr


def glra_distance(lost_tracks, detections, target_frame, use_diou=True, min_obs=3):
    """
    GLRA (GPR Lost-track Re-Association) cost matrix.

    For each lost track, predict its bbox at target_frame via GPR, then compute
    IoU (or DIoU) distance against the candidate detections.
    Tracks with insufficient history get cost=1.0 (unmatchable).

    :param lost_tracks:  list[STrack]  – unmatched tracks after second DCM
    :param detections:   list[STrack]  – unmatched low-score detections
    :param target_frame: int           – current frame_id
    :param use_diou:     bool          – use DIoU instead of IoU
    :param min_obs:      int           – minimum history length to attempt GPR
    :rtype: np.ndarray  shape (len(lost_tracks), len(detections))
    """
    cost_matrix = np.ones((len(lost_tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix

    det_tlbrs = np.asarray([t.tlbr for t in detections], dtype=np.float64)

    pred_tlbrs = []
    valid_idx = []
    for i, track in enumerate(lost_tracks):
        pred = gpr_predict_bbox(track, target_frame, min_obs)
        if pred is not None:
            pred_tlbrs.append(pred)
            valid_idx.append(i)

    if len(pred_tlbrs) == 0:
        return cost_matrix

    pred_arr = np.asarray(pred_tlbrs, dtype=np.float64)
    if use_diou:
        sim = dious(pred_arr, det_tlbrs)  # shape (k, m)
    else:
        sim = ious(pred_arr, det_tlbrs)  # shape (k, m)

    for local_i, global_i in enumerate(valid_idx):
        cost_matrix[global_i] = 1.0 - sim[local_i]

    return cost_matrix


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
