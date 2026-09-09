# -*- coding: utf-8 -*-
"""
qgloc: localization radius estimation on the quasi-geostrophic model.

The radius is estimated per cluster of grid points, where the clusters come
from ensemble-derived features, so it can follow the flow while the number of
estimated parameters stays small enough to be identifiable.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import metaheuristics, persist, progress
from .assimilation import (RadiusSpec, assimilate, cluster_features,
                           cluster_labels, run_cycles, score)
from .metaheuristics import METAHEURISTICS, defaults_for, get_optimizer
from .objective import BudgetExhausted, RadiusObjective
from .persist import (SnapshotWriter, load_snapshots, per_cycle_frame)
from .testbed import QGConfig, Testbed, build_model

__all__ = [
    "BudgetExhausted", "METAHEURISTICS", "QGConfig", "RadiusObjective",
    "RadiusSpec", "SnapshotWriter", "Testbed", "assimilate", "build_model",
    "cluster_features", "cluster_labels", "defaults_for", "get_optimizer",
    "load_snapshots", "metaheuristics", "per_cycle_frame", "persist",
    "progress", "run_cycles", "score", "__version__",
]
