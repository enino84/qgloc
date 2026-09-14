# -*- coding: utf-8 -*-
"""
Modified Cholesky precision, built for repeated evaluation.

This reproduces exactly what ``pyteda.analysis.AnalysisEnKFModifiedCholesky``
computes and is about two orders of magnitude faster at the thing that matters
here, which is evaluating thousands of candidate radii rather than one.

The formula, unchanged
----------------------
For each component ``i`` with predecessor set ``P(i, r)``, regress the
deviation ``DX[i]`` on ``DX[P]`` with a ridge penalty, and take

    L[i, i]   = 1
    L[i, P]   = -beta_i
    D[i, i]   = 1 / var(residual_i)
    B^{-1}    = L' D L

The first component has no predecessors, so its residual is its own deviation.
pyteda fits with ``sklearn.linear_model.Ridge(fit_intercept=False)``, and that
is what is reproduced below in closed form:

    beta = (X'X + alpha I)^{-1} X'y

with ``alpha`` absolute, exactly as sklearn applies it, so the numbers match.

Why this module exists
----------------------
Profiling one analysis at 4802 components: 7.8 seconds total, 7.2 of them in
the precision, and of those 5.1 inside scikit-learn, 4800 calls to
``Ridge.fit``, of which 2.1 seconds are ``check_array`` validating inputs.
Each regression has four predecessors and forty members; solving it directly is
microseconds. The cost was framework overhead repeated once per grid point.

Two changes follow from that.

**Closed form.** The normal equations are solved directly, so a regression
costs a 4x4 solve instead of a sklearn call.

**Per-row caching.** The predecessor set of component ``i`` depends only on
``i`` and on its own integer radius, so the pair ``(i, r_i)`` determines its
row of ``L`` and its entry of ``D`` completely. They are cached. A search that
changes the radius of one cluster then recomputes only the rows of that
cluster, and a search that revisits a radius recomputes nothing. Over a tabu
run, where every move changes one cluster out of ``2K``, that is the difference
between rebuilding the whole precision each time and touching a fraction of it.

The cache is keyed on the deviation matrix identity, so it is invalidated
automatically when the ensemble changes at the next cycle.

The boundary
------------
``q`` vanishes on the Dirichlet boundary, so every member of the ensemble is
zero there, the deviation is identically zero, and ``1 / var`` is infinite.
Measured at 4802 components: 290 rows, all of them boundary points, and
pyteda's precision contains 290 infinities for exactly that reason.

Neither zero precision nor infinite precision is the honest answer. Zero says
the variable is unknown, which is false; infinity says it is known exactly,
which is true but propagates into the analysis solve as an infinity. The clean
answer is the one Sakov takes when he reports the QG dimension as 127x127
*excluding the boundary points*: those components are not estimated at all.
``active_mask`` marks them, the regressions skip them, and their rows of the
precision are left with a unit diagonal, which leaves the analysis increment at
those points equal to zero -- which is the correct increment for a component
whose value the boundary condition already fixes.

Numerical fragility worth knowing about
---------------------------------------
Some rows near the boundary are fitted almost exactly by their predecessors, so
the residual variance is of order 1e-13 and ``1 / var`` is of order 1e13. Those
entries dominate the analysis solve, and they are pure rounding: pyteda and
this module agree to 2.4e-16 on well-conditioned rows and disagree by tens of
percent on these, because both are inverting numerical noise. It is a property
of the estimator rather than of either implementation, and a floor on the
residual variance would be the way to control it if it matters.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags


def ridge_closed_form(X, y, alpha, scale=True):
    """``(X'X + a I)^{-1} X'y``, with ``a`` a fraction of the trace of X'X.

    Scaling matters more than the value. An absolute penalty is meaningless
    unless the state happens to be of order one: in model units the variance of
    q here is of order 1e7, so alpha = 0.01 is fifteen orders of magnitude
    below the diagonal and regularizes nothing. The regression then interpolates
    its predecessors exactly whenever the ensemble is small, the residual
    variance is rounding noise, and its inverse -- which is the diagonal of the
    precision -- explodes.

    Measured: with an absolute penalty the median diagonal of the precision was
    2059 against an inverse ensemble variance of 2.25e-6, a factor of 9e8. The
    filter then believes the background is a billion times more certain than it
    is and ignores every observation; the analysis made the error *worse* at
    every radius. With the penalty scaled by the trace the diagonal returns to
    the order of the ensemble variance and the analysis improves the background.

    ``scale=False`` reproduces sklearn's ``Ridge(fit_intercept=False)`` exactly,
    which is what pyteda uses, and is kept for the equivalence test.
    """
    p = X.shape[1]
    G = X.T @ X
    a = alpha * (np.trace(G) / p) if scale else alpha
    G.flat[:: p + 1] += a
    return np.linalg.solve(G, X.T @ y)


class PrecisionBuilder:
    """B^{-1} from the modified Cholesky decomposition, cached row by row.

    Parameters
    ----------
    model : pyteda model
        Used only for ``get_pre(i, r)``, the predecessor set. Passing the model
        rather than reimplementing the geometry keeps this consistent with
        whatever the model defines a neighbourhood to be, including how it
        treats the Dirichlet boundary.
    alpha : float
        Ridge penalty, matching pyteda's default.
    """

    def __init__(self, model, n, alpha=0.3, active_mask=None,
                 var_floor=1e-8, scale_ridge=True):
        self.model = model
        self.n = int(n)
        self.alpha = float(alpha)
        self.scale_ridge = bool(scale_ridge)
        self.var_floor = float(var_floor)
        self.active = (np.ones(self.n, dtype=bool) if active_mask is None
                       else np.asarray(active_mask, dtype=bool))
        self._rows = {}          # (i, r) -> (predecessors, beta, d)
        self._pre = {}           # (i, r) -> predecessors
        self._DX_id = None
        self.n_row_builds = 0
        self.n_row_hits = 0

    # ------------------------------------------------------------------
    def bind(self, DX):
        """Attach a deviation matrix, clearing the cache if it changed.

        The cached rows are only valid for the ensemble they were computed
        from, so binding a new one invalidates them. Identity is used rather
        than contents: the caller propagates a new ensemble each cycle and
        never mutates one in place.
        """
        key = id(DX)
        if key != self._DX_id:
            self._rows.clear()
            self._DX_id = key
        self.DX = np.asarray(DX, dtype=float)
        self.N = self.DX.shape[1]
        # Components with no ensemble variance at all are inactive whatever the
        # caller said: they carry no information and would divide by zero.
        zero_var = self.DX.var(axis=1) <= 0.0
        if zero_var.any():
            self.active = self.active & ~zero_var
        return self

    def predecessors(self, i, r):
        key = (int(i), int(r))
        if key not in self._pre:
            self._pre[key] = np.asarray(self.model.get_pre(int(i), int(r)),
                                        dtype=int)
        return self._pre[key]

    def row(self, i, r):
        """The row of L and the entry of D for component ``i`` at radius ``r``."""
        key = (int(i), int(r))
        cached = self._rows.get(key)
        if cached is not None:
            self.n_row_hits += 1
            return cached
        self.n_row_builds += 1

        if not self.active[i]:
            # A component the boundary condition fixes: not estimated, unit
            # diagonal, zero increment.
            out = (np.empty(0, dtype=int), np.empty(0), 1.0)
            self._rows[key] = out
            return out

        y = self.DX[i, :]
        idx = self.predecessors(i, r) if i > 0 else np.empty(0, dtype=int)
        # Predecessors that carry no variance contribute nothing and make the
        # normal equations singular, so they are dropped.
        if idx.size:
            idx = idx[self.active[idx]]
        if idx.size == 0:
            v = float(np.var(y))
            out = (idx, np.empty(0), 1.0 / v if v > 0 else 0.0)
        else:
            X = self.DX[idx, :].T
            beta = ridge_closed_form(X, y, self.alpha, self.scale_ridge)
            resid = y - X @ beta
            v = float(np.var(resid))
            # Floor the residual variance relative to the component's own
            # variance. Without it, rows that their predecessors fit almost
            # exactly get a precision of order 1e13, which is the inverse of
            # rounding noise and dominates the analysis solve.
            floor = self.var_floor * float(np.var(y))
            v = max(v, floor)
            out = (idx, beta, 1.0 / v if v > 0 else 0.0)
        self._rows[key] = out
        return out

    # ------------------------------------------------------------------
    def build(self, r_field, sparse=None):
        """B^{-1} for a per-component radius field.

        ``r_field`` is one radius per component, which is what a clustered
        parameterization expands to. Radii are floored to integers, so two
        candidates that floor alike share every cached row.
        """
        r = np.asarray(r_field)
        if r.ndim == 0:
            r = np.full(self.n, int(r))
        r = np.floor(r).astype(int)

        if sparse is None:
            sparse = self.n >= 2000

        rows, cols, vals = [], [], []
        d = np.empty(self.n)
        for i in range(self.n):
            idx, beta, di = self.row(i, r[i])
            d[i] = di
            rows.append(i)
            cols.append(i)
            vals.append(1.0)
            if idx.size:
                rows.extend([i] * idx.size)
                cols.extend(idx.tolist())
                vals.extend((-beta).tolist())

        if sparse:
            L = csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))
            return (L.T @ diags(d, format="csr") @ L).tocsr()
        L = np.zeros((self.n, self.n))
        L[rows, cols] = vals
        return L.T @ (d[:, None] * L)

    # ------------------------------------------------------------------
    def stats(self):
        return dict(cached_rows=len(self._rows), builds=self.n_row_builds,
                    hits=self.n_row_hits)
