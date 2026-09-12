# -*- coding: utf-8 -*-
"""
The quasi-geostrophic testbed: model, ensemble, observations.

Three things in this module are decisions rather than conveniences, and each
was arrived at by measuring what happened without it.

**Only ``q`` is estimated.** The model integrates ``q`` and recovers ``psi``
by solving the Helmholtz problem at every step, so ``psi`` is a diagnostic and
not a prognostic variable: ``pyteda``'s ``propagate`` reads ``x0[:field_size]``
and calls ``_calc_psi(q)``, ignoring whatever ``psi`` it was handed. Measured
directly: halving ``psi`` while leaving ``q`` untouched and propagating for 20
time units gives a bit-identical result.

An analysis that corrects the ``psi`` block therefore throws that correction
away at the next propagation. The state estimated here is ``q`` alone, 2401
components rather than 4802, and ``psi`` is recomputed from the analysed ``q``.
That halves the number of regressions, halves the dimension of the linear
solve, and takes the radius vector from ``2K`` components to ``K``.

**The state is normalized per field.** The potential vorticity ``q`` has a
climatological standard deviation of order 2000 and the streamfunction ``psi``
of order 1. An observation error that is reasonable for one is meaningless for
the other, and pyteda's LETKF reads a single scalar variance off the noise
object, so a heterogeneous R silently applies the first field's error to both.
Dividing each field by its climatological spread turns that into a
non-problem: in normalized units both fields have unit spread, one isotropic
error is correct for both, and the aggregate RMSE stops being dominated by
``q`` by four orders of magnitude.

**Observations are placed on a regular lattice, not at random.** With a random
subset the spacing between observations fluctuates, so some local domains
contain several and others none, and the analysis at a point with an empty
domain is simply the background. Measured directly: at 2% random density the
analysis left ``q`` unchanged to four decimal places at every radius, which
looks exactly like a broken filter and is not one. A lattice fixes the spacing
at ``stride`` grid points, which makes the relation between observation density
and useful radius explicit rather than statistical.

**The ensemble is climatological, not perturbed.** This is the recipe of Sakov
and Oke (2008): run the model once for a long time, keep snapshots along the
way, and draw the members and the truth at random from that set. Perturbing a
state and propagating does not work here, and the reason is measurable. A
perturbation is mostly energy outside the attractor, and the hyperviscosity
removes it: starting from 30% of climatology, the background error falls to
0.068 after 120 time units, a factor of five, and no amount of raising the
perturbation amplitude compensates because the propagation eats it. Only after
about 250 units does the surviving component start to grow, reaching 0.872 at
t = 1000 -- which is exactly the 0.873 a climatological ensemble has from the
start. The long propagation is a slow way of arriving where the snapshots
already are.

**The dissipation is ten times the model default.** With ``rkh2 = 1e-12`` the
norm of ``q`` grows without saturating -- a factor of fourteen between t = 500
and t = 8000 -- so there is no stationary regime and no climatology to sample.
At ``rkh2 = 1e-11`` it settles: 1.107e4 at t = 20000 against 1.222e4 at
t = 60000. Sakov and Oke report the same, raising the dissipation by a factor
of ten and reducing the step from 1.5 to 1.25 to get a stable assimilating
system.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyteda.models import QGModel


# ----------------------------------------------------------------------
@dataclass
class QGConfig:
    """Everything that defines the testbed."""
    mrefin: int = 6              # 6 -> 97x97 per field, 7 -> 193x193
    # Sakov and Oke use 1.25 for the assimilating run, arrived at by reducing
    # it from the 1.5 that free runs allow. The same reduction is needed again
    # here and for the same reason they give: the corrections localization
    # introduces are dynamically inconsistent, and the biharmonic term
    # amplifies the short scales they create until the forecast blows up. The
    # failure is in `propagate`, not in the analysis.
    dt: float = 0.3125
    bc: str = "dirichlet"
    scheme: str = "rk4"
    rkh2: float = 1e-11          # ten times the model default; see the header
    # Climatology: one long run, snapshots along it, members drawn at random.
    spinup: float = 20000.0      # to the stationary regime, cached
    n_snapshots: int = 60
    snapshot_every: float = 250.0
    ensemble_size: int = 20
    obs_stride: int = 4          # lattice spacing; density is 1/stride^2
    obs_std: float = 0.05        # in normalized units, so 5% of each spread
    obs_freq: float = 10.0       # time between assimilation cycles
    inflation: float = 1.15
    cycles: int = 40
    burn_in: int = 10


def build_model(cfg: QGConfig) -> QGModel:
    return QGModel(mrefin=cfg.mrefin, scheme=cfg.scheme, dt=cfg.dt,
                   bc=cfg.bc, rkh2=cfg.rkh2, ic_kind="zero", verbose=False)


class Testbed:
    """Model, blocks, climatological scales and the normalization they define."""

    def __init__(self, cfg: QGConfig, snapshots: np.ndarray):
        self.cfg = cfg
        self.model = build_model(cfg)
        self.n = self.model.get_number_of_variables()
        self.blocks = dict(self.model.var_blocks)
        self.g = int(np.sqrt(self.blocks["q"].stop - self.blocks["q"].start))
        self.snapshots = np.asarray(snapshots, dtype=float)
        if self.snapshots.ndim != 2 or self.snapshots.shape[1] != self.n:
            raise ValueError(f"snapshots must be (n_snapshots, {self.n})")
        self.x0_ref = self.snapshots[0]

        # Climatological spread per field, over the whole snapshot set rather
        # than one state, since that is what "climatological" means and what
        # the normalization should be expressed in.
        self.spread = {k: float(self.snapshots[:, b].std())
                       for k, b in self.blocks.items()}
        self.scale = np.ones(self.n)
        for k, b in self.blocks.items():
            self.scale[b] = self.spread[k]

    # ------------------------------------------------------------------
    @property
    def qblock(self):
        """The prognostic block. Everything estimated lives here."""
        return self.blocks["q"]

    @property
    def nq(self):
        return self.qblock.stop - self.qblock.start

    def psi_from_q(self, x):
        """Recompute psi from q, the way the model does at every step.

        Used after an analysis so that the state handed back satisfies
        Lap psi - F psi = q, which an independently corrected psi would not.
        """
        x = np.asarray(x, dtype=float)
        out = x.copy()
        b = self.blocks["psi"]
        core = getattr(self.model, "_core", None)
        q2 = x[self.qblock].reshape(self.g, self.g)
        if core is not None and hasattr(core, "_calc_psi"):
            out[b] = np.asarray(core._calc_psi(q2)).ravel()
        return out

    def normalize(self, x):
        return np.asarray(x) / (self.scale if np.ndim(x) == 1
                                else self.scale[:, None])

    def denormalize(self, x):
        return np.asarray(x) * (self.scale if np.ndim(x) == 1
                                else self.scale[:, None])

    def _unused_perturbation(self, rng, fraction):
        """A perturbation that respects the model's own constraints.

        Two of them, and ignoring either puts the ensemble outside the space of
        valid states.

        The boundary is Dirichlet: ``psi`` vanishes on it. Adding noise there
        gives members that violate the condition, and although the integrator
        reimposes it, the ensemble starts with spurious variance concentrated
        exactly on the boundary, where local domains have fewest neighbours and
        the analysis is already most fragile. Worse for this study, a
        clustering built on ensemble features picks that variance up and
        produces a boundary cluster that corresponds to no physical regime.

        And ``q`` is not free: it is tied to ``psi`` by q = Laplacian(psi) - F
        psi. Perturbing the two independently produces states in which that
        relation does not hold, and an ensemble whose cross-field correlation
        is weaker than the model's own. Here ``psi`` is perturbed and ``q`` is
        derived from it, so the pair stays on the constraint.
        """
        p = np.zeros(self.n)
        g = self.g
        dpsi = rng.standard_normal((g, g))
        dpsi[0, :] = dpsi[-1, :] = dpsi[:, 0] = dpsi[:, -1] = 0.0
        dpsi *= fraction * self.spread["psi"]

        h = 1.0 / (g - 1)
        lap = np.zeros_like(dpsi)
        lap[1:-1, 1:-1] = (dpsi[2:, 1:-1] + dpsi[:-2, 1:-1]
                           + dpsi[1:-1, 2:] + dpsi[1:-1, :-2]
                           - 4.0 * dpsi[1:-1, 1:-1]) / h ** 2
        F = float(getattr(self.model, "f", 1600.0))
        dq = lap - F * dpsi

        p[self.blocks["psi"]] = dpsi.ravel()
        # The derived q inherits psi's amplitude through the operator, so it is
        # rescaled to the requested fraction of q's own spread; without that a
        # perturbation meant to be 5% would be set by the Helmholtz constant.
        sd = dq.std()
        if sd > 0:
            dq = dq / sd * fraction * self.spread["q"]
        p[self.blocks["q"]] = dq.ravel()
        return p

    def boundary_mask(self):
        """True on the grid points that lie on the Dirichlet boundary."""
        g = self.g
        m2 = np.zeros((g, g), dtype=bool)
        m2[0, :] = m2[-1, :] = m2[:, 0] = m2[:, -1] = True
        m = np.zeros(self.n, dtype=bool)
        for b in self.blocks.values():
            m[b] = m2.ravel()
        return m

    # ------------------------------------------------------------------
    def build_ensemble(self, seed):
        """Members and truth drawn at random from the climatological set.

        No perturbation and no per-member spin-up: every member is already a
        state of the attractor, and the dispersion between them is the
        variability the model itself produces. This is what Sakov and Oke do,
        and the measurements in the module header say why nothing else works.
        """
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        need = cfg.ensemble_size + 1
        if self.snapshots.shape[0] < need:
            raise ValueError(f"need {need} snapshots, have "
                             f"{self.snapshots.shape[0]}")
        pick = rng.choice(self.snapshots.shape[0], size=need, replace=False)
        x_true = self.snapshots[pick[0]].copy()
        X = self.snapshots[pick[1:]].T.copy()
        return X, x_true

    # ------------------------------------------------------------------
    def checkerboard(self, stride, offset=0, fields=("q",)):
        """Observed state indices on a regular lattice of the given spacing.

        A lattice rather than a random subset, so that every analysis point is
        at a known distance from its nearest observation and the relation
        between density and useful radius is a property of the design instead
        of an accident of the draw. ``offset`` shifts the lattice, which lets
        the network move between cycles without changing its density.
        """
        g = self.g
        stride = int(stride)
        # The grid side is not in general a multiple of the stride, so a plain
        # modulo lattice changes size when it is shifted: at g=25 and stride 4
        # the offsets 0 and 1 give 98 and 72 observations. Taking a fixed count
        # of evenly spaced positions instead keeps the density constant while
        # still moving the network, which is what the design needs.
        k = max(1, g // stride)
        base = (np.round(np.linspace(0, g - g / k, k)).astype(int) + offset) % g
        rows = np.zeros(g, dtype=bool); rows[np.unique(base)] = True
        mask = np.outer(rows, rows).ravel()
        idx = []
        for name in fields:
            b = self.blocks[name]
            idx.append(b.start + np.flatnonzero(mask))
        return np.sort(np.concatenate(idx))

    def density(self, stride):
        """Actual fraction of the state observed by the lattice.

        Not 1/stride**2: the grid side is not a multiple of the stride in
        general, so the realized density is computed rather than assumed.
        """
        return float(self.checkerboard(stride).size) / self.n

    # ------------------------------------------------------------------
    def errors(self, x, x_true):
        """RMSE per field in normalized units, plus the relative error.

        Normalized units are what make the two fields comparable; the relative
        error is what a reader expects to see. Both are reported because they
        answer different questions and neither can be recovered from the other
        without the climatological spread.
        """
        out = {}
        for k, b in self.blocks.items():
            d = (x[b] - x_true[b]) / self.spread[k]
            out[f"rmse_{k}"] = float(np.sqrt(np.mean(d ** 2)))
            ref = np.sqrt(np.mean((x_true[b] / self.spread[k]) ** 2))
            out[f"rel_{k}"] = float(out[f"rmse_{k}"] / ref) if ref > 0 else np.nan
        out["rmse"] = float(np.mean([out[f"rmse_{k}"] for k in self.blocks]))
        return out
