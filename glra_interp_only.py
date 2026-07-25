"""
GLRA-style interpolation as offline post-processing.

Strict counterpart to ``interpolation.py``: the *scope* is identical to
DTI - only frame gaps inside a single track-id are filled, no re-association
across tracklets happens. The only difference is the interpolation curve:

  DTI                  : linear between the two endpoint frames.
  GLRA-interp (here)   : Gaussian-Process posterior mean fit on a window of
                         up to ``n_history`` frames from BOTH sides of the
                         gap (true interpolation, both ends anchored).

This is the fair like-for-like comparison the advisor asked for:
"both are interpolation, the difference should be small."

The DotProduct + RBF + WhiteKernel kernel family with fixed bounds and
``optimizer=None`` mirrors the online GLRA module, so the offline number
reflects the same trajectory model.

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
# Evaluator + helpers (copied from interpolation.py so this file is standalone)
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
# GPR machinery (fixed-hyperparameter, optimizer-disabled, same as online GLRA)
# -----------------------------------------------------------------------------
def make_kernel(rbf_ls=30.0, white_noise=1.0, dot_sigma0=0.0, use_dotproduct=True):
    """Fixed-bounds kernel. Set ``use_dotproduct=False`` for a pure
    RBF + WhiteKernel kernel (no linear trend term), which matches the
    "true interpolation" setting better since both anchors are clamped.

    Note: setting ``dot_sigma0=0`` does NOT disable DotProduct -- its
    kernel function becomes ``k(x, x') = x * x'`` which is still an
    active linear kernel. Use the boolean toggle instead.
    """
    rbf = RBF(length_scale=rbf_ls, length_scale_bounds="fixed")
    white = WhiteKernel(noise_level=white_noise, noise_level_bounds="fixed")
    if use_dotproduct:
        return DotProduct(sigma_0=dot_sigma0, sigma_0_bounds="fixed") + rbf + white
    return rbf + white


def _fit_gpr(frames, values, t0, kernel_kwargs):
    """Fit 1-D GPR mapping (frame - t0) -> value. ``t0`` shifts frame numbers
    to small magnitudes so the DotProduct term stays well-conditioned."""
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
    mean, _ = gpr.predict(X, return_std=True)
    return mean


# -----------------------------------------------------------------------------
# Top-level entry point (mirrors ``dti`` in interpolation.py)
# -----------------------------------------------------------------------------
def glra_interp(
    txt_path,
    save_path,
    n_min=5,
    n_dti=20,
    n_history=15,
    rbf_ls=30.0,
    white_noise=1.0,
    dot_sigma0=0.0,
    use_dotproduct=True,
):
    """GPR interpolation of within-tracklet frame gaps.

    Drop-in replacement for DTI - signature, scope, and gating thresholds
    (``n_min``, ``n_dti``) are intentionally identical. The only change is
    that the missing frames between gap endpoints are filled using the
    posterior mean of a GP fit on a window of up to ``n_history`` frames
    from each side, instead of by linear interpolation.

    Parameters
    ----------
    n_min : minimum tracklet length (frames) to attempt any filling.
    n_dti : maximum gap (in frames) to interpolate across. A gap of
            ``right_frame - left_frame`` is filled iff it lies strictly
            between 1 and ``n_dti``.
    n_history : window of context observations taken from each side of
                the gap to fit the GP on.
    rbf_ls, white_noise, dot_sigma0: fixed kernel hyper-parameters.
    """
    kernel_kwargs = dict(
        rbf_ls=rbf_ls,
        white_noise=white_noise,
        dot_sigma0=dot_sigma0,
        use_dotproduct=use_dotproduct,
    )
    seq_txts = sorted(
        p
        for p in glob.glob(os.path.join(txt_path, "*.txt"))
        if os.path.basename(p).startswith(("MOT17", "MOT20"))
    )
    for seq_txt in seq_txts:
        seq_name = os.path.basename(seq_txt)
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=",")
        if seq_data.ndim == 1:
            seq_data = seq_data.reshape(1, -1)
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)

        for track_id in range(min_id, max_id + 1):
            index = seq_data[:, 1] == track_id
            tracklet = seq_data[index]
            tracklet_filled = tracklet
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]

            if n_frame > n_min:
                # Ensure chronological order.
                order = np.argsort(tracklet[:, 0])
                tracklet = tracklet[order]
                frames = tracklet[:, 0]

                x = tracklet[:, 2]
                y = tracklet[:, 3]
                w = tracklet[:, 4]
                h = tracklet[:, 5]
                cx = x + w / 2.0
                cy = y + h / 2.0

                frames_filled = {}

                for i in range(1, n_frame):
                    right_frame = int(frames[i])
                    left_frame = int(frames[i - 1])
                    gap = right_frame - left_frame
                    # Same gate as DTI: only fill gaps strictly between 1
                    # and n_dti frames.
                    if not (1 < gap < n_dti):
                        continue

                    # Context window: up to n_history frames from each side,
                    # forming a single training set for true interpolation.
                    L0 = max(0, i - n_history)
                    L1 = i  # exclusive
                    R0 = i
                    R1 = min(n_frame, i + n_history)
                    ctx_idx = np.concatenate([np.arange(L0, L1), np.arange(R0, R1)])
                    ctx_frames = frames[ctx_idx]
                    t0 = int(ctx_frames.min())

                    try:
                        gpr_cx = _fit_gpr(ctx_frames, cx[ctx_idx], t0, kernel_kwargs)
                        gpr_cy = _fit_gpr(ctx_frames, cy[ctx_idx], t0, kernel_kwargs)
                        gpr_w = _fit_gpr(ctx_frames, w[ctx_idx], t0, kernel_kwargs)
                        gpr_h = _fit_gpr(ctx_frames, h[ctx_idx], t0, kernel_kwargs)
                    except Exception:
                        continue

                    fill_frames = np.arange(left_frame + 1, right_frame)
                    mcx = _predict_gpr(gpr_cx, fill_frames, t0)
                    mcy = _predict_gpr(gpr_cy, fill_frames, t0)
                    mw = np.maximum(_predict_gpr(gpr_w, fill_frames, t0), 1.0)
                    mh = np.maximum(_predict_gpr(gpr_h, fill_frames, t0), 1.0)

                    for k, f in enumerate(fill_frames):
                        bx = mcx[k] - mw[k] / 2.0
                        by = mcy[k] - mh[k] / 2.0
                        frames_filled[int(f)] = (bx, by, mw[k], mh[k])

                num_new = len(frames_filled)
                if num_new > 0:
                    data_new = np.zeros((num_new, 10), dtype=np.float64)
                    keys = list(frames_filled.keys())
                    for n, f in enumerate(keys):
                        bx, by, bw, bh = frames_filled[f]
                        data_new[n, 0] = f
                        data_new[n, 1] = track_id
                        data_new[n, 2:6] = [bx, by, bw, bh]
                        data_new[n, 6:] = [1, -1, -1, -1]
                    tracklet_filled = np.vstack((tracklet, data_new))

            seq_results = np.vstack((seq_results, tracklet_filled))

        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        write_results_score(save_seq_txt, seq_results)


if __name__ == "__main__":
    # data_root = "/home/caig/data/MOT17/train"
    # txt_path = "/home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results"
    # save_path = (
    #     "/home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results_dti"
    # )

    data_root = "/home/caig/data/MOT17/train"
    txt_path = "/home/caig/repo/SparseTrack/yolox_mix17_ablation/yolox_mix17_ablation_det/track_results"
    save_path = "/home/caig/repo/SparseTrack/yolox_mix17_ablation/yolox_mix17_ablation_det/track_results_glra"

    # data_root = "/home/caig/data/MOT17/test"
    # txt_path = (
    #     "/home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results_test"
    # )
    # save_path = (
    #     "/home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results_test_dti"
    # )

    # data_root = "/home/caig/data/MOT20/train"
    # txt_path = "/home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results"
    # save_path = "/home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results_dti"

    # data_root = "/home/caig/data/MOT20/test"
    # txt_path = (
    #     "/home/caig/repo/SparseTrack/yolox_mix20/yolox_mix20_det/track_results_test"
    # )

    # save_path = (
    #     "/home/caig/repo/SparseTrack/yolox_mix20/yolox_mix20_det/track_results_test_dti"
    # )

    mkdir_if_missing(save_path)
    # Match DTI's main(): n_min=5, n_dti=20.
    glra_interp(
        txt_path,
        save_path,
        n_min=5,
        n_dti=20,
        n_history=15,
        rbf_ls=30.0,
        white_noise=1e-6,
        dot_sigma0=0.0,
        use_dotproduct=False,
    )

    print("Before GLRA-interp:")
    eval_mota(data_root, txt_path)
    print("After GLRA-interp:")
    eval_mota(data_root, save_path)
