# -*- coding: utf-8 -*-
"""Registry of radius searches.

All share the signature

    optimize(obj, lo, hi, budget, rng, x0=None, **params) -> (theta, info)

Tabu is the one built for this problem: the radius is an integer per cluster,
its move changes one cluster at a time, and a single-cluster move rebuilds only
the part of the analysis that depends on it. The others are here so that the
comparison has more than one side, and random sampling and the exhaustive sweep
are here so that the comparison means something.
"""
from __future__ import annotations

from . import baselines, fpa, sa, swarm, tabu

OPTIMIZERS = {
    "tabu": tabu.optimize,
    "sa": sa.optimize,
    "fpa": fpa.optimize,
    "firefly": swarm.optimize_firefly,
    "ga": swarm.optimize_ga,
    "pso": swarm.optimize_pso,
    "de": swarm.optimize_de,
    "random": baselines.optimize_random,
    "grid": baselines.optimize_grid,
}

LABELS = {
    "tabu": "Tabu search",
    "sa": "Simulated annealing",
    "fpa": "Flower pollination",
    "firefly": "Firefly",
    "ga": "Genetic algorithm",
    "pso": "Particle swarm",
    "de": "Differential evolution",
    "random": "Random sampling",
    "grid": "Exhaustive sweep",
}

METAHEURISTICS = ["tabu", "sa", "fpa", "firefly", "ga"]
CONTROLS = ["random", "grid"]

DEFAULTS = {
    "tabu": dict(tenure=8, iters=200),
    "sa": dict(cooling=0.92, step_frac=0.2, step_decay=0.98),
    "fpa": dict(pop_size=15, switch_p=0.8, lam=1.5, gamma=0.1, levy_scale=0.01),
    "firefly": dict(pop_size=15, beta0=1.0, absorb=1.0, zeta=0.25),
    "ga": dict(pop_size=15, blx_alpha=0.5, p_mut=0.25, mut_frac=0.15),
    "pso": dict(pop_size=15, w=0.72, c1=1.49, c2=1.49),
    "de": dict(pop_size=15, F=0.6, CR=0.9),
    "random": dict(),
    "grid": dict(log_spaced=False),
}


def get_optimizer(name):
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer '{name}'. Available: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name]


def defaults_for(name):
    return dict(DEFAULTS.get(name, {}))


__all__ = ["OPTIMIZERS", "LABELS", "METAHEURISTICS", "CONTROLS", "DEFAULTS",
           "get_optimizer", "defaults_for"]
