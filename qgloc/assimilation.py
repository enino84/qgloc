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
``clustered`` one radius per cluster per field, where the clusters come from
              ensemble-derived features. This is the parameterization the
              experiments are built around: it lets the radius follow the flow
              (the jet is not where the quiet corners are) while keeping the
              number of estimated parameters at 2K rather than at the state
              dimension, which is the ratio that makes it identifiable.
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
                raise ValueError("clustered needs labels per field")
            self.K = sum(int(labels[k].max()) + 1 for k in bed.blocks)
        else:
            raise ValueError(f"unknown parameterization '{kind}'")

    def expand(self, theta):
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        r = np.empty(self.bed.n)
        if self.kind == "uniform":
            r[:] = theta[0]
            return r
        pos = 0
        for name, b in self.bed.blocks.items():
            if self.kind == "per_field":
                r[b] = theta[pos]
                pos += 1
            else:
                lab = self.labels[name]
                k = int(lab.max()) + 1
                r[b] = theta[pos:pos + k][lab]
                pos += k
        return r

    def label(self):
        return f"{self.kind}(K={self.K})"


# ----------------------------------------------------------------------
def assimilate(bed, X, x_true, r_vector, rng, method="letkf", stride=None,
               offset=0, extra=None):
    """One cycle: propagate, observe on the lattice, analyse.

    The analysis is performed on the **normalized** state, so a single
    isotropic observation error is correct for both fields, and the result is
    mapped back afterwards.
    """
    cfg = bed.cfg
    T = np.array([0.0, cfg.obs_freq])
    N = X.shape[1]

    X = np.stack([bed.model.propagate(X[:, e], T) for e in range(N)], axis=1)
    xb = X.mean(axis=1)
    X = xb[:, None] + cfg.inflation * (X - xb[:, None])
    x_true = bed.model.propagate(x_true, T)

    idx = bed.checkerboard(stride or cfg.obs_stride, offset=offset)
    Xn = bed.normalize(X)
    xtn = bed.normalize(x_true)
    y = xtn[idx] + cfg.obs_std * rng.standard_normal(idx.size)
    noise = IsotropicDiagonal(std=cfg.obs_std, dim=idx.size)

    kwargs = dict(model=bed.model, r=r_vector)
    kwargs.update(extra or {})
    analysis = AnalysisFactory(method, **kwargs).create_analysis()

    Hm = None
    if method != "letkf":
        # Sparse, and not by preference. A dense selection operator at
        # 74498 components and 4608 observations is 2.7 GB, which killed the
        # process outright; the matrix has one entry per row.
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

    Xa = bed.denormalize(analysis.perform_assimilation(_Bg(), _Obs()))
    return Xa, x_true, xb, idx


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
        Xa, x_true, xb, idx = assimilate(
            bed, X, x_true, r_vec, rng, method=method, stride=stride,
            offset=(k % (stride or cfg.obs_stride)) if moving else 0)
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
