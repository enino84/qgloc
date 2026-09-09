# -*- coding: utf-8 -*-
"""
The quasi-geostrophic testbed: model, ensemble, observations.

Three things in this module are decisions rather than conveniences, and each
was arrived at by measuring what happened without it.

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

**The ensemble is built by perturbing in units of each field's own spread.** A
perturbation of 0.5 is enormous for ``psi`` and negligible for ``q``; using an
absolute number produces an ensemble whose members are indistinguishable from
one another in ``q``, a background error four thousand times smaller than
climatology, and nothing for the filter to correct.
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
    dt: float = 0.5
    bc: str = "dirichlet"
    scheme: str = "rk4"
    spinup: float = 8000.0       # from rest to a developed flow, cached
    spinup_xb: float = 60.0      # about two e-folding times (tau_L ~ 31.5)
    spinup_ens: float = 30.0
    pert_xb: float = 0.20        # as a fraction of each field's spread
    # Calibrated, not guessed: at 0.05 the ensemble spread is a fifth of the
    # background error, so the filter believes it knows five times more than it
    # does and rejects observations it should accept. At 0.3 the ratio is 1.02.
    pert_ens: float = 0.30
    ensemble_size: int = 20
    obs_stride: int = 4          # lattice spacing; density is 1/stride^2
    obs_std: float = 0.05        # in normalized units, so 5% of each spread
    obs_freq: float = 10.0       # time between assimilation cycles
    inflation: float = 1.05
    cycles: int = 40
    burn_in: int = 10


def build_model(cfg: QGConfig) -> QGModel:
    return QGModel(mrefin=cfg.mrefin, scheme=cfg.scheme, dt=cfg.dt,
                   bc=cfg.bc, ic_kind="zero", verbose=False)


class Testbed:
    """Model, blocks, climatological scales and the normalization they define."""

    def __init__(self, cfg: QGConfig, x0_ref: np.ndarray):
        self.cfg = cfg
        self.model = build_model(cfg)
        self.n = self.model.get_number_of_variables()
        self.blocks = dict(self.model.var_blocks)
        self.g = int(np.sqrt(self.blocks["q"].stop - self.blocks["q"].start))
        self.x0_ref = np.asarray(x0_ref, dtype=float)

        # Climatological spread per field, from the reference state. This is
        # what the normalization and every perturbation are expressed in.
        self.spread = {k: float(self.x0_ref[b].std()) for k, b in self.blocks.items()}
        self.scale = np.ones(self.n)
        for k, b in self.blocks.items():
            self.scale[b] = self.spread[k]

    # ------------------------------------------------------------------
    def normalize(self, x):
        return np.asarray(x) / (self.scale if np.ndim(x) == 1
                                else self.scale[:, None])

    def denormalize(self, x):
        return np.asarray(x) * (self.scale if np.ndim(x) == 1
                                else self.scale[:, None])

    def perturbation(self, rng, fraction):
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
        """Phases 2 and 3 of the recipe, plus the synchronized truth."""
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        xb = self.model.propagate(self.x0_ref + self.perturbation(rng, cfg.pert_xb),
                                  np.array([0.0, cfg.spinup_xb]))
        X = np.empty((self.n, cfg.ensemble_size))
        for k in range(cfg.ensemble_size):
            X[:, k] = self.model.propagate(
                xb + self.perturbation(rng, cfg.pert_ens),
                np.array([0.0, cfg.spinup_ens]))
        x_true = self.model.propagate(
            self.x0_ref, np.array([0.0, cfg.spinup_xb + cfg.spinup_ens]))
        return X, x_true

    # ------------------------------------------------------------------
    def checkerboard(self, stride, offset=0, fields=("q", "psi")):
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
