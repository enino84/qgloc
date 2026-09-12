# -*- coding: utf-8 -*-
"""Budget-counted objectives over the radius.

Two of them. The truth-based one is the oracle: it scores a radius by the
analysis error against the true state, which no operational system can do, and
it exists to bound what any selection rule could achieve. The cross-validated
one withholds part of the observation lattice and scores against what it
withheld, which is admissible.

Both count unique evaluations and cache repeats, so two searches compared at
"equal budget" are comparable in the only sense that matters. The radius is an
integer, so candidates that floor to the same value cost one evaluation.
"""
from __future__ import annotations

import numpy as np


class BudgetExhausted(Exception):
    """Raised when a search asks for one evaluation too many."""


class RadiusObjective:
    """Wrap a scoring function with budget accounting and an integer cache."""

    n_evals = n_calls = n_cache_hits = 0
    stalled = False

    def __init__(self, evaluate, budget, r_min=1, r_max=12, call_factor=25):
        self._evaluate = evaluate
        self.budget = int(budget)
        self.r_min, self.r_max = int(r_min), int(r_max)
        self.call_factor = int(call_factor)
        self._cache = {}
        self.trace = []
        self.arg_trace = []
        self.n_evals = self.n_calls = self.n_cache_hits = 0
        self.stalled = False

    def key(self, theta):
        return tuple(np.clip(np.floor(np.atleast_1d(theta)),
                             self.r_min, self.r_max).astype(int))

    def __call__(self, theta):
        self.n_calls += 1
        if self.n_calls > self.budget * self.call_factor:
            self.stalled = True
            raise BudgetExhausted("call limit")
        k = self.key(theta)
        if k in self._cache:
            self.n_cache_hits += 1
            return self._cache[k]
        if self.n_evals >= self.budget:
            raise BudgetExhausted("budget")
        v = float(self._evaluate(np.array(k, dtype=float)))
        self._cache[k] = v
        self.n_evals += 1
        self.trace.append(v)
        self.arg_trace.append(k)
        return v

    def remaining(self):
        return 0 if self.stalled else max(0, self.budget - self.n_evals)

    def best_so_far(self):
        return min(self._cache.values()) if self._cache else float("nan")

    def best_trace(self):
        """Running minimum over unique evaluations, for a convergence plot."""
        return (np.minimum.accumulate(np.asarray(self.trace, dtype=float))
                if self.trace else np.empty(0))

    def history(self):
        """Every evaluation with its argument, for the optimization record."""
        return [dict(step=i, value=float(v), theta=list(map(int, t)))
                for i, (v, t) in enumerate(zip(self.trace, self.arg_trace))]

    def top(self, k=10):
        """The k best distinct vectors the search visited, best first.

        The winner alone does not say whether the problem is identified. If the
        runners-up are close to it the structure is pinned down; if they are
        very different at nearly the same objective, the landscape is flat and
        that is itself the result. Reporting them costs nothing and makes the
        difference visible.
        """
        items = sorted(self._cache.items(), key=lambda kv: kv[1])[:int(k)]
        return [dict(rank=i, value=float(v), theta=list(map(int, t)))
                for i, (t, v) in enumerate(items)]
