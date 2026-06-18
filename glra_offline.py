"""
GLRA as offline post-processing.

Drop-in counterpart to ``interpolation.py``: takes the per-sequence track
result txt files written by the tracker and produces a new set of txt
files with (1) fragmented tracklets re-associated via GPR trajectory
prediction and (2) the resulting gaps filled with the GPR posterior mean.

The intent is a like-for-like comparison against DTI in the offline
ablation: DTI does linear interpolation only; this script does GPR
re-association + GPR interpolation, using the same fixed kernel as the
online GLRA module so the offline number reflects ``GLRA-as-post-
processing'' rather than a different model.

sparsetrack.py is not touched.
"""

import copy
import glob
import os

import motmetrics as mm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel

mm.lap.default_solver = "lap"


# -----------------------------------------------------------------------------
# Evaluator (copied verbatim from interpolation.py so this file is standalone)
# -----------------------------------------------------------------------------
class Evaluator(object):
    def __init__(self, data_root, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type
        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type == "mot"
        gt_filename = os.path.join(self.data_root, self.seq_name, "gt", "gt.txt")
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)
        self.gt_ignore_frame_dict = read_results(
            gt_filename, self.data_type, is_ignore=True
        )

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(
                lambda a: np.asarray(a, dtype=int), [match_is, match_js]
            )
            match_ious = iou_distance[match_is, match_js]
            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
        trk_tlwhs = trk_tlwhs[keep]
        trk_ids = trk_ids[keep]
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)
        self.acc.update(gt_ids, trk_ids, iou_distance)
        return None

    def eval_file(self, filename):
        self.reset_accumulator()
        result_frame_dict = read_results(filename, self.data_type, is_gt=False)
        frames = sorted(list(set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)
        return self.acc

    @staticmethod
    def get_summary(
        accs,
        names,
        metrics=("mota", "num_switches", "idp", "idr", "idf1", "precision", "recall"),
    ):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)
        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs, metrics=metrics, names=names, generate_overall=True
        )
        return summary


def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ("mot", "lab"):
        return read_mot_results(filename, is_gt, is_ignore)
    raise ValueError("Unknown data type: {}".format(data_type))


def read_mot_results(filename, is_gt, is_ignore):
    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            for line in f.readlines():
                linelist = line.split(",")
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())
                if is_gt:
                    if "MOT16-" in filename or "MOT17-" in filename:
                        label = int(float(linelist[7]))
                        mark = int(float(linelist[6]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    score = 1
                elif is_ignore:
                    if "MOT16-" in filename or "MOT17-" in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])
                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])
                results_dict[fid].append((tlwh, target_id, score))
    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)
    return tlwhs, ids, scores


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def eval_mota(data_root, txt_path):
    accs = []
    seqs = sorted([s for s in os.listdir(data_root)])
    for seq in seqs:
        video_out_path = os.path.join(txt_path, seq + ".txt")
        evaluator = Evaluator(data_root, seq, "mot")
        accs.append(evaluator.eval_file(video_out_path))
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


def write_results_score(filename, results):
    save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
    with open(filename, "w") as f:
        for i in range(results.shape[0]):
            frame_data = results[i]
            frame_id = int(frame_data[0])
            track_id = int(frame_data[1])
            x1, y1, w, h = frame_data[2:6]
            line = save_format.format(
                frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1
            )
            f.write(line)


# -----------------------------------------------------------------------------
# GLRA-specific helpers
# -----------------------------------------------------------------------------
def make_kernel(rbf_ls=30.0, white_noise=1.0, dot_sigma0=0.0):
    """Same fixed-hyperparameter kernel family as the online GLRA module.

    DotProduct + RBF + WhiteKernel, all bounds fixed, ``optimizer=None``
    when used to fit. Tune ``rbf_ls`` / ``white_noise`` here if you want
    the offline post-processing to mirror specific online settings.
    """
    return (
        DotProduct(sigma_0=dot_sigma0, sigma_0_bounds="fixed")
        + RBF(length_scale=rbf_ls, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=white_noise, noise_level_bounds="fixed")
    )


def _fit_gpr(frames, values, t0, kernel_kwargs):
    """Fit a 1-D GPR mapping (frame - t0) -> value."""
    X = (frames.astype(np.float64) - t0).reshape(-1, 1)
    y = values.astype(np.float64)
    gpr = GaussianProcessRegressor(
        kernel=make_kernel(**kernel_kwargs),
        optimizer=None,
        normalize_y=False,
        alpha=1e-10,
    )
    gpr.fit(X, y)
    return gpr


def _predict_gpr(gpr, frames, t0):
    X = (np.asarray(frames, dtype=np.float64) - t0).reshape(-1, 1)
    mean, std = gpr.predict(X, return_std=True)
    return mean, std


class Tracklet(object):
    """One ID's worth of consecutive (or near-consecutive) detections."""

    def __init__(self, track_id, rows):
        idx = np.argsort(rows[:, 0])
        rows = rows[idx]
        self.id = int(track_id)
        self.rows = rows
        self.frames = rows[:, 0].astype(np.int64)
        x, y, w, h = rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
        self.cx = x + w / 2.0
        self.cy = y + h / 2.0
        self.w = w
        self.h = h
        self.score = rows[:, 6]

    @property
    def start(self):
        return int(self.frames[0])

    @property
    def end(self):
        return int(self.frames[-1])

    @property
    def length(self):
        return len(self.frames)


def load_sequence(seq_txt):
    seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=",")
    if seq_data.ndim == 1:
        seq_data = seq_data.reshape(1, -1)
    ids = np.unique(seq_data[:, 1].astype(np.int64))
    tracklets = []
    for tid in ids:
        mask = seq_data[:, 1] == tid
        tracklets.append(Tracklet(tid, seq_data[mask]))
    return tracklets


# -----------------------------------------------------------------------------
# Stage 1: re-association
# -----------------------------------------------------------------------------
def reassociate(
    tracklets,
    max_gap=30,
    n_min=5,
    n_history=30,
    sigma_cap=50.0,
    dist_gate=100.0,
    kernel_kwargs=None,
):
    """Greedy GPR-gated re-association of fragmented tracklets.

    For every ordered pair (A, B) with ``1 <= B.start - A.end - 1 <= max_gap``
    and both tracklets at least ``n_min`` frames long, fit GPR on A's last
    ``n_history`` observations and predict (cx, cy) at B.start. Accept the
    pair if the posterior std stays under ``sigma_cap`` and the predicted
    centre is within ``dist_gate`` pixels of B's first detection. Resolve
    competing matches greedily by ascending centre distance, then chain via
    union-find so that A -> B -> C collapses into a single group.
    """
    if kernel_kwargs is None:
        kernel_kwargs = {}

    eligible = [t for t in tracklets if t.length >= n_min]
    eligible.sort(key=lambda t: t.start)
    n = len(eligible)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Collect candidate matches.
    candidates = []
    for i, A in enumerate(eligible):
        # Fit A's GPR once (predict-only model) and re-use for every B candidate.
        tail = slice(max(0, A.length - n_history), A.length)
        t0 = int(A.frames[tail.start])
        try:
            gpr_x = _fit_gpr(A.frames[tail], A.cx[tail], t0, kernel_kwargs)
            gpr_y = _fit_gpr(A.frames[tail], A.cy[tail], t0, kernel_kwargs)
        except Exception:
            continue

        for j, B in enumerate(eligible):
            if i == j:
                continue
            gap = B.start - A.end - 1
            if gap < 1 or gap > max_gap:
                continue

            mx, sx = _predict_gpr(gpr_x, [B.start], t0)
            my, sy = _predict_gpr(gpr_y, [B.start], t0)
            if sx[0] > sigma_cap or sy[0] > sigma_cap:
                continue

            dx = B.cx[0] - mx[0]
            dy = B.cy[0] - my[0]
            dist = float(np.hypot(dx, dy))
            if dist > dist_gate:
                continue

            candidates.append((dist, i, j))

    # Greedy assignment with union-find merging.
    candidates.sort(key=lambda x: x[0])
    used_as_pred = set()
    used_as_succ = set()
    for _, i, j in candidates:
        if i in used_as_pred or j in used_as_succ:
            continue
        union(i, j)
        used_as_pred.add(i)
        used_as_succ.add(j)

    groups_map = {}
    for i, t in enumerate(eligible):
        r = find(i)
        groups_map.setdefault(r, []).append(t)

    short = [t for t in tracklets if t.length < n_min]
    groups = list(groups_map.values()) + [[t] for t in short]
    for g in groups:
        g.sort(key=lambda t: t.start)
    return groups


# -----------------------------------------------------------------------------
# Stage 2: gap filling with GPR posterior mean
# -----------------------------------------------------------------------------
def fill_gaps_gpr(group, max_gap_fill=30, n_history=30, kernel_kwargs=None):
    """Merge ``group`` into a single tracklet, filling each inter-tracklet
    gap with the predecessor's GPR posterior mean for (cx, cy, w, h)."""
    if kernel_kwargs is None:
        kernel_kwargs = {}

    new_id = group[0].id
    all_rows = []
    for t in group:
        for k in range(t.length):
            all_rows.append(
                [
                    t.frames[k],
                    new_id,
                    t.rows[k, 2],
                    t.rows[k, 3],
                    t.rows[k, 4],
                    t.rows[k, 5],
                    t.score[k],
                    -1,
                    -1,
                    -1,
                ]
            )

    for k in range(len(group) - 1):
        A = group[k]
        B = group[k + 1]
        gap_start = A.end + 1
        gap_end = B.start - 1
        if gap_end < gap_start:
            continue
        if gap_end - gap_start + 1 > max_gap_fill:
            continue

        tail = slice(max(0, A.length - n_history), A.length)
        t0 = int(A.frames[tail.start])
        try:
            gpr_x = _fit_gpr(A.frames[tail], A.cx[tail], t0, kernel_kwargs)
            gpr_y = _fit_gpr(A.frames[tail], A.cy[tail], t0, kernel_kwargs)
            gpr_w = _fit_gpr(A.frames[tail], A.w[tail], t0, kernel_kwargs)
            gpr_h = _fit_gpr(A.frames[tail], A.h[tail], t0, kernel_kwargs)
        except Exception:
            continue

        fr = np.arange(gap_start, gap_end + 1)
        mx, _ = _predict_gpr(gpr_x, fr, t0)
        my, _ = _predict_gpr(gpr_y, fr, t0)
        mw, _ = _predict_gpr(gpr_w, fr, t0)
        mh, _ = _predict_gpr(gpr_h, fr, t0)
        mw = np.maximum(mw, 1.0)
        mh = np.maximum(mh, 1.0)

        for ii, f in enumerate(fr):
            x = mx[ii] - mw[ii] / 2.0
            y = my[ii] - mh[ii] / 2.0
            all_rows.append([f, new_id, x, y, mw[ii], mh[ii], 1.0, -1, -1, -1])

    arr = np.asarray(all_rows, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 10)
    return arr[arr[:, 0].argsort()]


# -----------------------------------------------------------------------------
# Top-level entry point (mirrors ``dti`` in interpolation.py)
# -----------------------------------------------------------------------------
def glra_offline(
    txt_path,
    save_path,
    max_gap=30,
    n_min=5,
    n_history=30,
    sigma_cap=50.0,
    dist_gate=100.0,
    rbf_ls=30.0,
    white_noise=1.0,
    dot_sigma0=0.0,
):
    """Run GLRA-style re-association + GPR gap filling on every sequence
    txt under ``txt_path`` and write the result to ``save_path``.

    Parameters
    ----------
    max_gap : maximum lost-track gap to attempt re-association across
              (the offline analogue of ``glra_min_lost`` / re-assoc window).
    n_min   : minimum tracklet length to be considered as a re-assoc endpoint.
    n_history : how many of A's tail observations are used to fit GPR.
    sigma_cap : reject a match if the GPR posterior std at B.start exceeds
                this (pixels).
    dist_gate : reject a match if predicted-vs-observed centre distance
                exceeds this (pixels).
    rbf_ls, white_noise, dot_sigma0 :
        Fixed kernel hyper-parameters. Defaults are placeholders; set them
        to whatever the online module is using so the offline number reflects
        the same model.
    """
    kernel_kwargs = dict(rbf_ls=rbf_ls, white_noise=white_noise, dot_sigma0=dot_sigma0)
    seq_txts = sorted(
        p
        for p in glob.glob(os.path.join(txt_path, "*.txt"))
        if os.path.basename(p).startswith(("MOT17", "MOT20"))
    )
    for seq_txt in seq_txts:
        seq_name = os.path.basename(seq_txt)
        tracklets = load_sequence(seq_txt)

        groups = reassociate(
            tracklets,
            max_gap=max_gap,
            n_min=n_min,
            n_history=n_history,
            sigma_cap=sigma_cap,
            dist_gate=dist_gate,
            kernel_kwargs=kernel_kwargs,
        )

        seq_results = []
        for g in groups:
            merged = fill_gaps_gpr(
                g,
                max_gap_fill=max_gap,
                n_history=n_history,
                kernel_kwargs=kernel_kwargs,
            )
            if merged.size > 0:
                seq_results.append(merged)

        if seq_results:
            seq_results = np.vstack(seq_results)
            seq_results = seq_results[seq_results[:, 0].argsort()]
        else:
            seq_results = np.zeros((0, 10), dtype=np.float64)

        save_seq_txt = os.path.join(save_path, seq_name)
        write_results_score(save_seq_txt, seq_results)


if __name__ == "__main__":
    # MOT17 train:
    data_root = "/home/caig/data/MOT17/train"
    txt_path = "/home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results"
    save_path = (
        "/home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results_glra"
    )

    # MOT17 test:
    # data_root = "/home/caig/data/MOT17/test"
    # txt_path  = "/home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results_test"
    # save_path = "/home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results_test_glra"

    # MOT20 train:
    # data_root = "/home/caig/data/MOT20/train"
    # txt_path  = "/home/caig/repo/SparseTrack/yolox_mix20/yolox_mix20_det/track_results"
    # save_path = "/home/caig/repo/SparseTrack/yolox_mix20/yolox_mix20_det/track_results_glra"

    # MOT20 test:
    # data_root = "/home/caig/data/MOT20/test"
    # txt_path  = "/home/caig/repo/SparseTrack/yolox_mix20/yolox_mix20_det/track_results_test"
    # save_path = "/home/caig/repo/SparseTrack/yolox_mix20/yolox_mix20_det/track_results_test_glra"

    mkdir_if_missing(save_path)

    glra_offline(
        txt_path,
        save_path,
        max_gap=20,  # match DTI's n_dti for a fair comparison
        n_min=5,  # match DTI's n_min for a fair comparison
        n_history=30,
        sigma_cap=50.0,
        dist_gate=100.0,
        rbf_ls=30.0,
        white_noise=1.0,
        dot_sigma0=0.0,
    )

    print("Before GLRA-offline: ")
    eval_mota(data_root, txt_path)
    print("After GLRA-offline:")
    eval_mota(data_root, save_path)
