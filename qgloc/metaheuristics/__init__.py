# -*- coding: utf-8 -*-
"""Registry of combinatorial searches over the integer radius vector.

All share the signature

    optimize(obj, lo, hi, budget, rng, x0=None, **params) -> (theta, info)

and all operate on the same space: a vector of ``2K`` integers in
``{r_min, ..., r_max}``. Nothing here produces a real-valued solution and
nothing is rounded. The four population methods and the two trajectory methods
share one neighbourhood, so a comparison between them measures the acceptance
rule rather than the move.
"""
from __future__ import annotations

from . import discrete
from .discrete import neighbour, random_vector

OPTIMIZERS = {
    "tabu": discrete.optimize_tabu,
    "annealing": discrete.optimize_annealing,
    "genetic": discrete.optimize_genetic,
    "ant": discrete.optimize_ant,
    "fpa": discrete.optimize_fpa,
    "firefly": discrete.optimize_firefly,
    "random": discrete.optimize_random,
    "exhaustive": discrete.optimize_exhaustive,
}

LABELS = {
    "tabu": "Tabu search",
    "annealing": "Simulated annealing",
    "genetic": "Genetic algorithm",
    "ant": "Ant colony",
    "fpa": "Flower pollination",
    "firefly": "Firefly",
    "random": "Random sampling",
    "exhaustive": "Exhaustive sweep",
}

# Firefly is implemented and registered but left out of the comparison: it and
# flower pollination are both Yang's, and one of the two is enough to represent
# that family. Adding "firefly" to this list puts it back in every experiment.
METAHEURISTICS = ["tabu", "annealing", "genetic", "ant", "fpa"]
CONTROLS = ["random", "exhaustive"]

DEFAULTS = {
    "tabu": dict(tenure=8, iters=500),
    "annealing": dict(cooling=0.95),
    "genetic": dict(pop_size=15, p_mut=0.25, tournament=2),
    "ant": dict(n_ants=10, rho=0.1, alpha=1.0),
    "fpa": dict(pop_size=15, switch_p=0.8, lam=1.5, levy_scale=0.5),
    "firefly": dict(pop_size=15, beta0=0.9, absorb=1.0, zeta=0.1),
    "random": dict(),
    "exhaustive": dict(),
}


def get_optimizer(name):
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer '{name}'. Available: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name]


def defaults_for(name):
    return dict(DEFAULTS.get(name, {}))


__all__ = ["OPTIMIZERS", "LABELS", "METAHEURISTICS", "CONTROLS", "DEFAULTS",
           "get_optimizer", "defaults_for", "neighbour", "random_vector"]
