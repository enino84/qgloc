# -*- coding: utf-8 -*-
"""
The assimilation loop and the radius parameterizations.

The radius is passed to pyteda as a full per-component array, which
``resolve_radius`` accepts, so a radius per cluster costs nothing beyond
expanding the cluster labels. Three parameterizations are provided and they
differ only in how few numbers generate that array:

``uniform``   one radius for the whole state.
``per_field`` one radius per field. Two numbers, and the one with the clearest
              physical justification: ``q`` carries fine filaments while
              ``psi`` is the solution of a Helmholtz problem on ``q`` and is
              therefore smooth, so their correlation scales genuinely differ.
``clustered`` one radius per cluster of ``q``, where the clusters come from
              ensemble-derived features. This is the parameterization the
              experiments are built around: it lets the radius follow the flow
              while keeping the number of estimated parameters at K rather than
              at the state dimension, which is the ratio that makes it
              identifiable. Only ``q`` is clustered, because only ``q`` is
              estimated.
"""
from __future__ import annotations

import numpy as np

from pyteda.analysis.analysis_factory import AnalysisFactory
from pyteda.observation import IsotropicDiagonal


# ----------------------------------------------------------------------
def cluster_features(X, block, g):
    """Per-point features of a field, computed from the ensemble alone.

    Three features: the log background variance, and the ensemble correlation
    with the neighbours at lag one and lag two, averaged over the four
    directions. The lags are a direct sample estimate of how quickly
    correlation falls away, which is exactly the quantity a localization radius
    is supposed to respect. Everything here comes from the forecast ensemble,
    so a grouping built on it is admissible in an operational setting.
    """
    A = X[block] - X[block].mean(axis=1, keepdims=True)
    var = A.var(axis=1)

    F = A.reshape(g, g, -1)
    Fn = F - F.mean(axis=2, keepdims=True)
    sd = np.sqrt((Fn ** 2).sum(axis=2))
    sd[sd == 0] = 1e-12
    Fn = Fn / sd[:, :, None]

    def lag(d):
        return 0.25 * sum((Fn * np.roll(Fn, s, ax)).sum(axis=2)
                          for ax, s in ((0, d), (0, -d), (1, d), (1, -d)))

    return np.column_stack([np.log10(var + 1e-30), lag(1).ravel(), lag(2).ravel()])


def cluster_labels(X, block, g, K, seed=0):
    """K groups of grid points of one field, by ensemble features."""
    from sklearn.cluster import KMeans
    Z = cluster_features(X, block, g)
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)
    return KMeans(n_clusters=int(K), n_init=4, random_state=seed).fit(Z).labels_


def cluster_auto(X, block, g, K_range=(2, 8), seed=0, sample=2000):
    """Cluster a field, choosing the number of groups by silhouette.

    K is not a tuning knob to be set by hand: it is read off the ensemble like
    the features themselves, so the whole grouping stays admissible. The
    silhouette score compares how well each point fits its own group against
    the nearest other group, and the K that maximizes it is taken.

    The score is evaluated on a random subsample, because it is quadratic in
    the number of points and a field here has thousands.

    One caveat worth stating rather than discovering later: silhouette picks
    the K that separates the features most cleanly, which is not necessarily
    the K that localizes best. Whether the two coincide is measurable by
    comparing this choice against the K that minimizes the error.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    Z = cluster_features(X, block, g)
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)
    rng = np.random.default_rng(seed)
    sub = (rng.choice(Z.shape[0], size=min(sample, Z.shape[0]), replace=False)
           if Z.shape[0] > sample else np.arange(Z.shape[0]))

    best = (None, -np.inf, None)
    scores = {}
    for k in range(max(2, K_range[0]), K_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(Z)
        lab = km.labels_
        if len(np.unique(lab[sub])) < 2:
            continue
        sc = float(silhouette_score(Z[sub], lab[sub]))
        scores[k] = sc
        if sc > best[1]:
            best = (lab, sc, k)
    if best[0] is None:
        return np.zeros(Z.shape[0], dtype=int), dict(K=1, silhouette=np.nan,
                                                     scores={})
    return best[0], dict(K=int(best[2]), silhouette=float(best[1]),
                         scores=scores)


# ----------------------------------------------------------------------
class RadiusSpec:
    """Map a parameter vector to a per-component radius array."""

    def __init__(self, bed, kind="per_field", K=1, labels=None):
        self.bed, self.kind, self.K_per_field = bed, kind, int(K)
        self.labels = labels
        if kind == "uniform":
            self.K = 1
        elif kind == "per_field":
            self.K = len(bed.blocks)
        elif kind == "clustered":
            if labels is None:
                raise ValueError("clustered needs labels for q")
            lab = labels["q"] if isinstance(labels, dict) else labels
            self.labels = {"q": np.asarray(lab, dtype=int)}
            self.K = int(self.labels["q"].max()) + 1
            self.K_per_field = self.K
        else:
            raise ValueError(f"unknown parameterization '{kind}'")

    def expand(self, theta):
        """One radius per component of the full state.

        The psi block is filled with the same values as q. It is never used
        (only the q block is estimated) but keeping the array full-length means
        callers do not have to know which block is which.
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        r = np.empty(self.bed.n)
        if self.kind == "uniform":
            r[:] = theta[0]
            return r
        if self.kind == "per_field":
            for pos, b in enumerate(self.bed.blocks.values()):
                r[b] = theta[pos]
            return r
        lab = self.labels["q"]
        vals = theta[lab]
        for b in self.bed.blocks.values():
            r[b] = vals
        return r

    def label(self):
        return f"{self.kind}(K={self.K})"


# ----------------------------------------------------------------------
def forecast(bed, X, x_true, rng, stride=None, offset=0):
    """Propagate the ensemble and the truth, and draw the observations.

    Split out from the analysis deliberately. Optimizing a radius means scoring
    many candidates against the *same* forecast, and propagating the ensemble
    inside the objective would make every evaluation cost an integration of
    every member -- minutes per search instead of seconds. The forecast is
    formed once; a candidate radius then costs one analysis.
    """
    cfg = bed.cfg
    T = np.array([0.0, cfg.obs_freq])
    N = X.shape[1]

    Xf = np.stack([bed.model.propagate(X[:, e], T) for e in range(N)], axis=1)
    xb = Xf.mean(axis=1)
    Xf = xb[:, None] + cfg.inflation * (Xf - xb[:, None])
    x_true = bed.model.propagate(x_true, T)

    idx = bed.checkerboard(stride or cfg.obs_stride, offset=offset)
    xtn = bed.normalize(x_true)
    y = xtn[idx] + cfg.obs_std * rng.standard_normal(idx.size)
    return dict(Xf=Xf, xb=xb, x_true=x_true, idx=idx, y=y,
                Xn=bed.normalize(Xf), xtn=xtn)


def analyse_fast(bed, fc, r_vector, mask=None, rebuild_psi=False):
    """EnKF with modified Cholesky, through the cached precision builder.

    The same estimator pyteda computes, but rows are cached by
    (component, radius). Without this the objective rebuilds every regression
    from scratch on every candidate, which is what made a calibration cell take
    four minutes instead of seconds: the fast precision existed but was never
    wired into the analysis path.

    The builder lives on the testbed and is rebound when the ensemble changes,
    so the cache spans a whole search and is invalidated between cycles.
    """
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve

    from .precision import PrecisionBuilder

    cfg = bed.cfg
    qb = bed.qblock
    nq = bed.nq

    # Only the q block is estimated: psi is a diagnostic the model recomputes
    # at every step, so an analysis of it would be discarded. Observations are
    # taken on q; their indices are mapped into the reduced block.
    idx_full = fc["idx"] if mask is None else fc["idx"][mask]
    y = fc["y"] if mask is None else fc["y"][mask]
    keep_q = (idx_full >= qb.start) & (idx_full < qb.stop)
    idx = idx_full[keep_q] - qb.start
    y = y[keep_q]

    Xn = fc["Xn"][qb, :]
    N = Xn.shape[1]

    if getattr(bed, "_pb", None) is None:
        bed._pb = PrecisionBuilder(bed.model, nq,
                                   alpha=getattr(cfg, "ridge_alpha", 0.01),
                                   active_mask=~bed.boundary_mask()[qb])
    DX = fc.get("DXq")
    if DX is None:
        DX = Xn - Xn.mean(axis=1, keepdims=True)
        fc["DXq"] = DX
    bed._pb.bind(DX)
    Binv = bed._pb.build(np.asarray(r_vector)[qb] if np.size(r_vector) == bed.n
                         else r_vector, sparse=True)

    p = idx.size
    H = sps.csr_matrix((np.ones(p), (np.arange(p), idx)), shape=(p, nq))
    rinv = 1.0 / cfg.obs_std ** 2
    A = (Binv + rinv * (H.T @ H)).tocsc()

    rng = np.random.default_rng(0)
    noise = cfg.obs_std * rng.standard_normal((p, N))
    D = (y[:, None] + noise) - (H @ Xn)
    Z = spsolve(A, sps.csc_matrix(rinv * (H.T @ D)))
    Z = Z.toarray() if sps.issparse(Z) else np.asarray(Z)

    Xa = fc["Xn"].copy()
    Xa[qb, :] = Xn + Z.reshape(nq, N)
    Xa = bed.denormalize(Xa)

    # Rebuilding psi means one multigrid Helmholtz solve per member, and it is
    # only needed when the analysis is going to be propagated: the criterion
    # scores observations of q and never looks at psi. Doing it inside the
    # objective cost 1.72 s per evaluation against 0.18, a factor of ten, and
    # would have put EXP-03 at roughly 690 hours. It is off by default and
    # switched on for the analysis that actually advances the filter.
    if rebuild_psi:
        for e in range(N):
            Xa[:, e] = bed.psi_from_q(Xa[:, e])
    return Xa



def analyse(bed, fc, r_vector, method="letkf", mask=None, extra=None):
    """Analyse a prepared forecast with a candidate radius. No integration.

    ``mask`` selects a subset of the observation lattice, which is how the
    cross-validated criterion withholds part of it without rebuilding
    anything. The analysis is performed on the **normalized** state, so a
    single isotropic observation error is correct for both fields.
    """
    cfg = bed.cfg
    idx = fc["idx"] if mask is None else fc["idx"][mask]
    y = fc["y"] if mask is None else fc["y"][mask]
    Xn = fc["Xn"]
    N = Xn.shape[1]
    noise = IsotropicDiagonal(std=cfg.obs_std, dim=idx.size)

    kwargs = dict(model=bed.model, r=r_vector)
    kwargs.update(extra or {})
    analysis = AnalysisFactory(method, **kwargs).create_analysis()

    Hm = None
    if method != "letkf":
        # Sparse, and not by preference. A dense selection operator at 74498
        # components and 4608 observations is 2.7 GB, which killed the process
        # outright; the matrix has one entry per row.
        import scipy.sparse as sps
        Hm = sps.csr_matrix(
            (np.ones(idx.size), (np.arange(idx.size), idx)),
            shape=(idx.size, bed.n))

    class _Bg:
        Xb = Xn
        ensemble_size = N

        def get_ensemble(self):
            return Xn

    class _Obs:
        H_index = idx

        def __init__(self):
            self.y = y
            self.noise = noise

        def get_observation(self):
            return y

        def get_observation_operator(self):
            return Hm

        def get_data_error_covariance(self):
            return np.eye(idx.size) * cfg.obs_std ** 2

    return bed.denormalize(analysis.perform_assimilation(_Bg(), _Obs()))


def assimilate(bed, X, x_true, r_vector, rng, method="letkf", stride=None,
               offset=0, extra=None):
    """One full cycle: forecast then analysis.

    Routes the modified-Cholesky filter to :func:`analyse_fast`, so that the
    sweep and the benchmark run the *same* analysis. They did not for a while
    and the symptom was visible in the output: the sweep reported a gain in psi
    of exactly zero, because the pyteda path leaves the psi block untouched
    while the fast path rebuilds it from the analysed q. An exact zero in a
    column that should carry noise is worth chasing.
    """
    fc = forecast(bed, X, x_true, rng, stride=stride, offset=offset)
    if method in ("enkf-modified-cholesky", "fast"):
        Xa = analyse_fast(bed, fc, r_vector, rebuild_psi=True)
    else:
        Xa = analyse(bed, fc, r_vector, method=method, extra=extra)
    return Xa, fc["x_true"], fc["xb"], fc["idx"]


class Diverged(RuntimeError):
    """The forecast could not be propagated.

    Not a bug to be avoided. A sparse network with a short radius corrects the
    observed points hard and leaves their neighbours alone, so the analysed
    field carries gradients the model cannot digest and the biharmonic term
    amplifies them until the integration fails. Sakov and Oke report the same
    and leave the diverged configurations blank in their figures: the map of
    where a scheme diverges is a result, not an accident, so it is recorded
    per cell rather than allowed to end the run.
    """


def run_cycles(bed, X0, x_true0, radius, seed, method="letkf", stride=None,
               moving=True, on_cycle=None):
    """A filter run at a fixed radius specification. Returns per-cycle rows.

    ``moving`` shifts the observation lattice by one grid point each cycle, so
    the network changes place without changing density, which is what a real
    network does and what stops any grid point from being permanently
    unobserved.
    """
    cfg = bed.cfg
    rng = np.random.default_rng(seed)
    X, x_true = X0.copy(), x_true0.copy()
    r_vec = radius if np.ndim(radius) else np.full(bed.n, float(radius))

    rows = []
    for k in range(cfg.cycles):
        try:
            Xa, x_true, xb, idx = assimilate(
                bed, X, x_true, r_vec, rng, method=method, stride=stride,
                offset=(k % (stride or cfg.obs_stride)) if moving else 0)
        except (RuntimeError, FloatingPointError) as exc:
            raise Diverged(f"cycle {k}: {exc}") from exc
        xa = Xa.mean(axis=1)
        rec = dict(cycle=k, p_obs=int(idx.size),
                   spread=float(np.mean(np.std(bed.normalize(Xa), axis=1))))
        rec.update({f"b_{a}": b for a, b in bed.errors(xb, x_true).items()})
        rec.update(bed.errors(xa, x_true))
        rows.append(rec)
        if on_cycle is not None:
            on_cycle(k, xb, xa, x_true, Xa, idx, rec)
        X = Xa
    return rows


def score(bed, rows, key="rmse"):
    """Post-burn-in mean of a per-cycle quantity."""
    import numpy as _np
    v = [r[key] for r in rows[bed.cfg.burn_in:]]
    return float(_np.mean(v)) if v else float("nan")
