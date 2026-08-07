"""Fast drop-in replacement for the sklearn GPR used by GLRA.

Reproduces sklearn's ``GaussianProcessRegressor`` configured as

    kernel = RBF(length_scale) + WhiteKernel(noise_level)
    n_restarts_optimizer=0, normalize_y=False, alpha=1e-10

including the per-track marginal-likelihood hyper-parameter optimisation
(L-BFGS-B over log length_scale / log noise_level), but calling LAPACK
directly instead of going through sklearn's kernel objects and scipy's
validating wrappers.  Agreement with sklearn is ~1e-6 in z-score units.
"""
import numpy as np
from scipy.linalg import get_lapack_funcs
from scipy.optimize import minimize

_JITTER = 1e-10                                     # sklearn's default `alpha`
_LOG_BOUNDS = np.log([[1e-5, 1e5], [1e-5, 1e5]])    # sklearn kernel.bounds
_potrf, _potrs, _potri = get_lapack_funcs(("potrf", "potrs", "potri"),
                                          (np.empty((1, 1)),))
_LOG2PI = np.log(2.0 * np.pi)


def _nll_and_grad(theta, sqd, y, n, diag):
    """Negative log-marginal-likelihood and gradient wrt theta = log(hyperparams)."""
    ls2 = np.exp(2.0 * theta[0])          # length_scale ** 2
    nl = np.exp(theta[1])                 # noise_level

    Krbf = np.exp(sqd * (-0.5 / ls2))
    K = Krbf.copy()
    K[diag] += nl + _JITTER

    L, info = _potrf(K, lower=1, clean=1, overwrite_a=1)
    if info != 0:
        return np.inf, np.zeros(2)

    alpha, info = _potrs(L, y, lower=1)
    if info != 0:
        return np.inf, np.zeros(2)

    lml = -0.5 * y.dot(alpha) - np.log(L[diag]).sum() - 0.5 * n * _LOG2PI

    Kinv, info = _potri(L, lower=1, overwrite_c=0)
    if info != 0:
        return np.inf, np.zeros(2)
    Kinv = Kinv + Kinv.T                  # potri fills one triangle only
    Kinv[diag] *= 0.5

    W = np.outer(alpha, alpha) - Kinv     # 0.5 * tr(W dK/dtheta)
    g0 = 0.5 * np.einsum("ij,ij->", W, Krbf * (sqd / ls2))
    g1 = 0.5 * nl * W[diag].sum()
    return -lml, -np.array([g0, g1])


def fit(x, targets, length_scale=0.5, noise_level=0.05):
    """Fit one independent GP per array in `targets`, all sharing inputs `x`."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    n = x.size
    diag = np.diag_indices(n)
    sqd = (x[:, None] - x[None, :]) ** 2   # shared across targets
    theta0 = np.log([length_scale, noise_level])

    states = []
    for y in targets:
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
        res = minimize(_nll_and_grad, theta0, args=(sqd, y, n, diag),
                       method="L-BFGS-B", jac=True, bounds=_LOG_BOUNDS)
        theta = res.x if np.isfinite(res.fun) else theta0

        ls2 = float(np.exp(2.0 * theta[0]))
        nl = float(np.exp(theta[1]))
        K = np.exp(sqd * (-0.5 / ls2))
        K[diag] += nl + _JITTER
        L, info = _potrf(K, lower=1, clean=1, overwrite_a=1)
        if info != 0:
            return None
        alpha, info = _potrs(L, y, lower=1)
        if info != 0:
            return None
        states.append((x, L, alpha, ls2, nl))
    return states


def predict(state, x_star):
    """Posterior mean and std at one query point (matches sklearn's predict)."""
    x, L, alpha, ls2, nl = state
    ks = np.exp((x - x_star) ** 2 * (-0.5 / ls2))
    mean = float(ks.dot(alpha))
    v, info = _potrs(L, ks, lower=1)
    var = (1.0 + nl) - ks.dot(v)          # sklearn: kernel.diag(X*) = 1 + noise_level
    return mean, float(np.sqrt(var)) if var > 0.0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Fit / prediction cache
# ─────────────────────────────────────────────────────────────────────────────
# A track's `gpr_obs` is FROZEN for as long as the track is lost: nothing is
# appended until it is re-activated.  GLRA is invoked on every frame in the
# [gpr_min_lost, gpr_max_lost] window, and within a single frame
# gpr_predict_bbox is called up to 4x for the same track (height gate,
# glra_distance, diag logging, confirm rollback).  All of those share the same
# observation history, so the expensive marginal-likelihood optimisation only
# has to happen ONCE per (track, history) pair.
#
# The cache key is the observation history itself, so it is invalidated
# automatically by a new detection being appended, by the sliding window
# dropping the oldest entry, and by GMC warping the historical centres.

_FIT_CACHE = {}      # obs_key            -> fit bundle
_PRED_CACHE = {}     # (obs_key, target)  -> (pred_tlbr, sigma_px)
_MAX_ENTRIES = 4096

stats = {"fit_calls": 0, "fit_hits": 0, "pred_hits": 0}


def reset_cache():
    """Call once per sequence so memory does not grow across a whole benchmark."""
    _FIT_CACHE.clear()
    _PRED_CACHE.clear()


def _evict_if_needed():
    if len(_FIT_CACHE) > _MAX_ENTRIES:
        _FIT_CACHE.clear()
    if len(_PRED_CACHE) > _MAX_ENTRIES:
        _PRED_CACHE.clear()


# Backend selection.  "fast" = LAPACK reimplementation (~6x faster per fit,
# agrees with sklearn to ~1e-6 in z-score units, i.e. far below a pixel).
# "sklearn" = original code path, bit-identical results, cache still active.
import os
BACKEND = os.environ.get("GLRA_GPR_BACKEND", "fast").lower()
