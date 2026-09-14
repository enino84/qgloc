# -*- coding: utf-8 -*-
"""
qgloc: adaptive localization by observation assignment.

Every model component grows a neighbourhood until it holds a few candidate
observations, and is then updated by exactly one of them, or by none. The
radius is not a parameter: it is the distance to whichever observation the
component ended up using. The choice minimizes a variational cost evaluated at
a scalar local analysis, computed from ensemble anomalies and observations
alone, and the resulting radius field fixes the predecessor sets of a
modified-Cholesky precision used for a single global analysis.

Modules
-------
``testbed``     the quasi-geostrophic model, the climatological ensemble and
                the observation lattice
``assignment``  candidates, the local cost, the searches, and the coupled
                variant that breaks separability
``precision``   the modified-Cholesky precision, with rows cached by
                (component, radius) and a trace-scaled ridge
``persist``     snapshot archive
``progress``    run logging
"""
from __future__ import annotations

__version__ = "0.2.0"

from . import assignment, persist, precision, progress
from .assignment import (AssignmentObjective, Candidates, CoupledCost,
                         CoupledObjective, LocalCost, NONE, SEARCHES,
                         build_candidates, search_annealing, search_greedy,
                         search_nearest, search_tabu)
from .persist import SnapshotWriter, load_snapshots, per_cycle_frame
from .precision import PrecisionBuilder, ridge_closed_form
from .testbed import QGConfig, Testbed, build_model

__all__ = [
    "AssignmentObjective", "Candidates", "CoupledCost", "CoupledObjective",
    "LocalCost", "NONE", "PrecisionBuilder", "QGConfig", "SEARCHES",
    "SnapshotWriter", "Testbed", "assignment", "build_candidates",
    "build_model", "load_snapshots", "per_cycle_frame", "persist",
    "precision", "progress", "ridge_closed_form", "search_annealing",
    "search_greedy", "search_nearest", "search_tabu", "__version__",
]
