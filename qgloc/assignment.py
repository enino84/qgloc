# -*- coding: utf-8 -*-
"""
Localization as an assignment problem.

Every grid point grows a neighbourhood until it has collected a small number of
candidate observations. It is then updated by **exactly one** of them, or by
none at all. Which one is the decision variable; the radius of the point falls
out of that choice, because the radius is the distance to the observation it
ended up using.

Why this is different from choosing a radius directly
-----------------------------------------------------
A radius per grid point is a free parameter with nothing to pin it down, and
the experiments that preceded this module showed what happens: with one radius
per component the criterion has far more freedom than the data can support, and
optimizing it well makes the analysis worse. Here the parameter is a choice
among a handful of observations that are actually there. There is no radius to
invent, and the search space is the product of a few small sets rather than a
box of integers.

Distance does not decide it
---------------------------
The nearest observation is not automatically the best one. What determines how
much an observation tells you about a point is the ensemble correlation between
them, and in a flow with structure two points aligned along a current can be
far better correlated than two adjacent points sitting across a front. So the
neighbourhood keeps growing past the first observation until it has
``n_candidates`` of them, and the choice among those is made on the cost, not
on the distance.

Abstaining is a legal choice
----------------------------
If none of the candidates correlates with the point, updating it from any of
them injects noise, which is the thing localization exists to prevent. Leaving
the point at its background value is therefore one of the options, and the cost
selects it on its own: with no observation the departure term is zero and only
the residual remains.

The cost
--------
For a point ``i`` updated by a single observation ``j``, the local analysis is a
scalar Kalman update and the cost is the variational one evaluated at it,

    J_i = (x^a_i - x^b_i)^2 / var_i  +  (y_j - x^a_j)^2 / R_j

the departure from the background weighted by the background precision, plus
the residual weighted by the observation precision. No truth, nothing withheld.
Everything is computed from ensemble anomalies, so it costs a handful of dot
products over the N members; the modified-Cholesky precision is never formed
during the search, because the structure of its predecessor sets is what the
assignment is deciding. That structure is built once, at the end, from the
assignment that won.

The global score is the mean of the local costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NONE = -1          # the abstain option, stored in an assignment vector


# ----------------------------------------------------------------------
@dataclass
class Candidates:
    """For each point, the observations it may be updated by.

    ``obs`` has shape (n_points, n_candidates) and holds indices into the
    observation vector, padded with ``NONE`` where a point found fewer than
    asked for. ``dist`` holds the corresponding grid distances, which become
    the radius of the point once one of them is chosen. ``free`` lists the
    points with a real decision to make: a point with a single candidate has
    nothing to optimize and is excluded from the search entirely.
    """
    obs: np.ndarray
    dist: np.ndarray
    free: np.ndarray
    n_candidates: int
    r_max: int

    @property
    def n_points(self):
        return self.obs.shape[0]

    def count(self, i):
        return int(np.sum(self.obs[i] != NONE))

    def summary(self):
        n = np.array([self.count(i) for i in range(self.n_points)])
        return dict(n_points=int(self.n_points),
                    n_free=int(self.free.size),
                    mean_candidates=float(n.mean()),
                    n_with_none=int(np.sum(n == 0)),
                    radius_mean=float(np.mean(self.dist[self.dist > 0])),
                    radius_max=float(np.max(self.dist)))


def build_candidates(g, obs_rows, obs_cols, n_candidates=4, r_max=12):
    """Grow a square neighbourhood around every point until it has candidates.

    The radius adapts to the local density on its own: where observations are
    dense it stops early, where they are sparse it grows. That is the behaviour
    a fixed radius cannot have, and it costs no extra parameter.

    Distances are Chebyshev, matching the square neighbourhoods the model's own
    predecessor sets use.
    """
    obs_rows = np.asarray(obs_rows, dtype=int)
    obs_cols = np.asarray(obs_cols, dtype=int)
    n_obs = obs_rows.size
    n_points = g * g

    grid = np.full((g, g), NONE, dtype=int)
    grid[obs_rows, obs_cols] = np.arange(n_obs)

    obs = np.full((n_points, n_candidates), NONE, dtype=int)
    dist = np.zeros((n_points, n_candidates), dtype=int)

    for i in range(n_points):
        r0, c0 = divmod(i, g)
        found, fdist = [], []
        for r in range(0, r_max + 1):
            if len(found) >= n_candidates:
                break
            lo_r, hi_r = max(0, r0 - r), min(g - 1, r0 + r)
            lo_c, hi_c = max(0, c0 - r), min(g - 1, c0 + r)
            if r == 0:
                ring = [(r0, c0)]
            else:
                ring = []
                for c in range(lo_c, hi_c + 1):
                    if r0 - r >= 0:
                        ring.append((r0 - r, c))
                    if r0 + r <= g - 1:
                        ring.append((r0 + r, c))
                for rr in range(max(lo_r, r0 - r + 1), min(hi_r, r0 + r - 1) + 1):
                    if c0 - r >= 0:
                        ring.append((rr, c0 - r))
                    if c0 + r <= g - 1:
                        ring.append((rr, c0 + r))
            for rr, cc in ring:
                k = grid[rr, cc]
                if k != NONE and k not in found:
                    found.append(int(k))
                    fdist.append(r)
        m = min(len(found), n_candidates)
        obs[i, :m] = found[:m]
        dist[i, :m] = fdist[:m]

    # Only points with more than one candidate carry a decision.
    n_each = np.sum(obs != NONE, axis=1)
    free = np.flatnonzero(n_each > 1)
    return Candidates(obs=obs, dist=dist, free=free,
                      n_candidates=n_candidates, r_max=r_max)


# ----------------------------------------------------------------------
class LocalCost:
    """The variational cost of an assignment, from ensemble anomalies alone.

    Everything is precomputed once per cycle: the variance of each point, the
    variance at each observed site, and the covariance between each point and
    each of its own candidates. Scoring an assignment is then a lookup and a
    mean, and changing one point's choice changes exactly one term. That is
    what makes a local move cheap and the whole search affordable.
    """

    def __init__(self, DX, cand, obs_idx, y, xb, obs_var):
        self.cand = cand
        n_members = DX.shape[1]
        self.var = DX.var(axis=1)                       # background variance
        self.obs_idx = np.asarray(obs_idx, dtype=int)
        self.y = np.asarray(y, dtype=float)
        self.xb = np.asarray(xb, dtype=float)
        self.obs_var = (np.full(self.y.size, float(obs_var))
                        if np.ndim(obs_var) == 0 else np.asarray(obs_var, float))
        self.innov = self.y - self.xb[self.obs_idx]

        A = DX - DX.mean(axis=1, keepdims=True)
        denom = max(n_members - 1, 1)

        # Covariance of each point with each of its candidates, and the
        # variance at the observed sites. Computed once; the search only reads.
        n_p, n_c = cand.obs.shape
        self.cov = np.zeros((n_p, n_c))
        for c in range(n_c):
            k = cand.obs[:, c]
            ok = k != NONE
            if not np.any(ok):
                continue
            site = self.obs_idx[k[ok]]
            self.cov[ok, c] = np.einsum("ij,ij->i", A[np.flatnonzero(ok)],
                                        A[site]) / denom
        self.var_at_obs = self.var[self.obs_idx]

        # The cost of abstaining: the background is kept, so the departure term
        # vanishes and only the residual against the observation remains.
        self._cost_none = np.zeros(n_p)
        for c in range(n_c):
            k = cand.obs[:, c]
            ok = k != NONE
            self._cost_none[ok] = (self.innov[k[ok]] ** 2
                                   / self.obs_var[k[ok]])
        self._table = self._build_table()

    def _build_table(self):
        """Cost of every (point, choice) pair, including abstaining."""
        n_p, n_c = self.cand.obs.shape
        tab = np.full((n_p, n_c + 1), np.inf)
        tab[:, n_c] = self._cost_none
        for c in range(n_c):
            k = self.cand.obs[:, c]
            ok = np.flatnonzero(k != NONE)
            if ok.size == 0:
                continue
            kk = k[ok]
            s = self.var_at_obs[kk] + self.obs_var[kk]
            gain = self.cov[ok, c] / np.where(s > 0, s, 1.0)
            d = gain * self.innov[kk]                    # x^a - x^b
            resid = self.innov[kk] * (1.0 - self.var_at_obs[kk] / s)
            v = np.where(self.var[ok] > 0, self.var[ok], np.inf)
            tab[ok, c] = d ** 2 / v + resid ** 2 / self.obs_var[kk]
        return tab

    # ------------------------------------------------------------------
    @property
    def n_choices(self):
        return self.cand.n_candidates + 1

    def cost_of(self, point, choice):
        return float(self._table[point, choice])

    def score(self, assignment):
        """The global cost: the mean of the local costs."""
        rows = np.arange(self._table.shape[0])
        return float(np.mean(self._table[rows, assignment]))

    def best_greedy(self):
        """The assignment each point would make on its own.

        Not the answer: the points interact through the precision that is built
        from them at the end, so the greedy choice is a starting point and a
        baseline, not an optimum.
        """
        return np.argmin(np.where(np.isfinite(self._table), self._table, np.inf),
                         axis=1)

    def radius_field(self, assignment):
        """The radius each point ends up with, given the assignment.

        The radius is not chosen; it is the distance to whichever observation
        the point was assigned to. A point that abstains gets zero.
        """
        n_c = self.cand.n_candidates
        r = np.zeros(self.cand.n_points, dtype=int)
        for c in range(n_c):
            sel = assignment == c
            r[sel] = self.cand.dist[sel, c]
        return r

    def assigned_obs(self, assignment):
        """The observation index updating each point, or NONE where it abstains."""
        n_c = self.cand.n_candidates
        out = np.full(self.cand.n_points, NONE, dtype=int)
        for c in range(n_c):
            sel = assignment == c
            out[sel] = self.cand.obs[sel, c]
        return out


# ----------------------------------------------------------------------
# Search over assignments
#
# A move changes the choice of one point, so the global score moves by one
# term out of n_points. Recomputing the mean from scratch would be O(n) per
# move; the delta is O(1). With a score that costs 16 microseconds to rebuild
# and a delta that costs nothing, hundreds of thousands of moves per cycle are
# affordable, and the search is no longer the expensive part of the cycle.
# ----------------------------------------------------------------------
class AssignmentObjective:
    """Budget-counted objective over assignments, with incremental moves.

    ``evaluate`` scores a whole assignment. ``delta`` gives the change from
    moving one point, without touching the rest. A search should use ``delta``
    for candidate moves and ``commit`` once it accepts one.
    """

    def __init__(self, cost: "LocalCost", budget=100_000):
        self.cost = cost
        self.table = cost._table
        self.n_points, self.n_choices = self.table.shape
        self.budget = int(budget)
        self.n_evals = 0
        self.trace = []

    def evaluate(self, assignment):
        self.n_evals += 1
        rows = np.arange(self.n_points)
        v = float(np.mean(self.table[rows, assignment]))
        self.trace.append(v)
        return v

    def delta(self, assignment, point, choice):
        """Change in the global score from reassigning one point."""
        self.n_evals += 1
        old = self.table[point, assignment[point]]
        new = self.table[point, choice]
        if not np.isfinite(new):
            return np.inf
        return float(new - old) / self.n_points

    def out_of_budget(self):
        return self.n_evals >= self.budget


def search_tabu(obj, assignment=None, iters=200, tenure=12, n_sample=None,
                rng=None):
    """Tabu search over assignments: one point changes per move.

    The move is the natural one for this problem and the reason tabu fits it:
    a point's choice is a small categorical variable, and changing it moves the
    score by a single term. The tabu list forbids returning a point to a choice
    it recently left, with the usual aspiration when the move improves on the
    incumbent best.

    Points are sampled rather than scanned exhaustively: with thousands of them
    a full sweep of the neighbourhood would be n_points x n_choices evaluations
    per iteration, and sampling a few hundred gets the same behaviour at a
    fraction of the cost.
    """
    rng = np.random.default_rng() if rng is None else rng
    a = (obj.cost.best_greedy() if assignment is None
         else np.asarray(assignment).copy())
    best_a, best_f = a.copy(), obj.evaluate(a)
    cur = best_f
    tabu = {}
    n_sample = n_sample or min(256, obj.n_points)

    for it in range(int(iters)):
        if obj.out_of_budget():
            break
        pts = rng.choice(obj.n_points, size=n_sample, replace=False)
        move, best_d = None, np.inf
        for p in pts:
            for c in range(obj.n_choices):
                if c == a[p]:
                    continue
                d = obj.delta(a, p, c)
                if not np.isfinite(d):
                    continue
                is_tabu = tabu.get((int(p), int(c)), 0) > it
                if is_tabu and not (cur + d < best_f):
                    continue
                if d < best_d:
                    move, best_d = (int(p), int(c)), d
        if move is None:
            break
        p, c = move
        tabu[(p, int(a[p]))] = it + int(tenure)
        a[p] = c
        cur += best_d
        if cur < best_f:
            best_a, best_f = a.copy(), cur

    return best_a, dict(J=float(best_f), n_evals=int(obj.n_evals),
                        optimizer="tabu")


def search_annealing(obj, assignment=None, iters=20_000, T0=None,
                     cooling=0.9995, rng=None):
    """Simulated annealing on the same move.

    Identical neighbourhood to the tabu search, so what is being compared
    between them is the acceptance rule and nothing else.
    """
    rng = np.random.default_rng() if rng is None else rng
    a = (obj.cost.best_greedy() if assignment is None
         else np.asarray(assignment).copy())
    cur = obj.evaluate(a)
    best_a, best_f = a.copy(), cur

    if T0 is None:
        ds = []
        for _ in range(64):
            p = int(rng.integers(obj.n_points))
            c = int(rng.integers(obj.n_choices))
            d = obj.delta(a, p, c)
            if np.isfinite(d) and d > 0:
                ds.append(d)
        T0 = (-float(np.mean(ds)) / np.log(0.8)) if ds else 1e-6
    T = float(T0)

    for _ in range(int(iters)):
        if obj.out_of_budget() or T <= 1e-18:
            break
        p = int(rng.integers(obj.n_points))
        c = int(rng.integers(obj.n_choices))
        if c == a[p]:
            continue
        d = obj.delta(a, p, c)
        if not np.isfinite(d):
            continue
        if d <= 0 or rng.random() < np.exp(-d / T):
            a[p] = c
            cur += d
            if cur < best_f:
                best_a, best_f = a.copy(), cur
        T *= cooling

    return best_a, dict(J=float(best_f), n_evals=int(obj.n_evals), T0=float(T0),
                        optimizer="annealing")


def search_greedy(obj, **_):
    """Each point takes its own best choice, ignoring the others.

    The baseline for the search: if a metaheuristic cannot beat this, the
    interaction between points is not worth searching over.
    """
    a = obj.cost.best_greedy()
    return a, dict(J=obj.evaluate(a), n_evals=int(obj.n_evals),
                   optimizer="greedy")


def search_nearest(obj, **_):
    """Every point takes its nearest observation. The classical choice."""
    a = np.zeros(obj.n_points, dtype=int)
    return a, dict(J=obj.evaluate(a), n_evals=int(obj.n_evals),
                   optimizer="nearest")


SEARCHES = dict(tabu=search_tabu, annealing=search_annealing,
                greedy=search_greedy, nearest=search_nearest)


# ----------------------------------------------------------------------
# Coupling: what makes the problem combinatorial
#
# The cost above decomposes over components, so its optimum is reached
# component by component and no search is needed. Verified, not assumed: tabu
# at 2e5 evaluations and annealing at 1.6e4 both return the greedy solution to
# every digit.
#
# A coupling term breaks that. The physical argument is the one the divergence
# measurements pointed at: an analysis that corrects an observed point hard and
# leaves its neighbour untouched creates a gradient the quasi-geostrophic
# dynamics cannot carry, and the biharmonic term amplifies it until the
# forecast fails. Penalising roughness in the increment is the direct statement
# of that.
#
# With it, the cost of a component depends on what its neighbours chose, so
# greedy is no longer optimal and a search has something to do. A move still
# only touches the component and its four neighbours, so it stays cheap.
# ----------------------------------------------------------------------
class CoupledCost:
    """Local cost plus a penalty on roughness of the analysis increment.

        J(a) = mean_i [ J_i(a_i) ]  +  lambda * mean_{(i,k) neighbours}
                                        (dx_i - dx_k)^2 / (var_i + var_k)

    where ``dx_i`` is the increment component ``i`` receives under its own
    choice. The pair term is normalised by the background variance of the two
    components, so a region where the ensemble is genuinely uncertain is
    allowed a larger jump than one where it is not; the penalty is on
    *unjustified* roughness, not on structure.

    ``lam`` is the one free parameter, and it has a natural scale: at
    ``lam = 0`` this is exactly ``LocalCost`` and greedy is optimal; as ``lam``
    grows the assignment is pushed towards agreeing with its neighbours.
    Calibrating it is the job of an experiment, not of a default.
    """

    def __init__(self, cost: "LocalCost", g, lam=1.0):
        self.cost = cost
        self.table = cost._table
        self.g = int(g)
        self.lam = float(lam)
        self.n_points, self.n_choices = self.table.shape

        # The increment each component would receive under each choice. Stored
        # once, so the pair term is a lookup.
        self.incr = self._increments()
        self.var = cost.var
        self.pairs = self._pairs()
        # Which pairs each component belongs to, so a move can find its own
        # terms without scanning.
        self.of = [[] for _ in range(self.n_points)]
        for p, (i, k) in enumerate(self.pairs):
            self.of[i].append(p)
            self.of[k].append(p)

    def _increments(self):
        c = self.cost
        n_c = c.cand.n_candidates
        inc = np.zeros((self.n_points, n_c + 1))
        for j in range(n_c):
            k = c.cand.obs[:, j]
            ok = np.flatnonzero(k != NONE)
            if ok.size == 0:
                continue
            kk = k[ok]
            s = c.var_at_obs[kk] + c.obs_var[kk]
            inc[ok, j] = (c.cov[ok, j] / np.where(s > 0, s, 1.0)) * c.innov[kk]
        return inc                       # last column stays 0: abstaining

    def _pairs(self):
        g = self.g
        idx = np.arange(self.n_points).reshape(g, g)
        right = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
        down = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
        return np.vstack([right, down])

    # ------------------------------------------------------------------
    def pair_terms(self, assignment):
        i, k = self.pairs[:, 0], self.pairs[:, 1]
        d = self.incr[i, assignment[i]] - self.incr[k, assignment[k]]
        denom = self.var[i] + self.var[k]
        return d ** 2 / np.where(denom > 0, denom, np.inf)

    def score(self, assignment):
        rows = np.arange(self.n_points)
        local = float(np.mean(self.table[rows, assignment]))
        rough = float(np.mean(self.pair_terms(assignment)))
        return local + self.lam * rough

    def delta(self, assignment, point, choice):
        """Change in the score from reassigning one component.

        Only the component's own term and the pairs it belongs to change, which
        is four pairs in the interior. That is what keeps a local move cheap
        even though the problem is no longer separable.
        """
        if choice == assignment[point]:
            return 0.0
        new_local = self.table[point, choice]
        if not np.isfinite(new_local):
            return np.inf
        d_local = (new_local - self.table[point, assignment[point]]) / self.n_points

        ps = self.of[point]
        if not ps:
            return float(d_local)
        ps = np.asarray(ps)
        i, k = self.pairs[ps, 0], self.pairs[ps, 1]
        denom = self.var[i] + self.var[k]
        denom = np.where(denom > 0, denom, np.inf)

        old_i = self.incr[i, assignment[i]]
        old_k = self.incr[k, assignment[k]]
        before = np.sum((old_i - old_k) ** 2 / denom)

        new_i = old_i.copy()
        new_k = old_k.copy()
        new_i[i == point] = self.incr[point, choice]
        new_k[k == point] = self.incr[point, choice]
        after = np.sum((new_i - new_k) ** 2 / denom)

        d_rough = (after - before) / self.pairs.shape[0]
        return float(d_local + self.lam * d_rough)

    def best_greedy(self):
        """Still available, and now only a starting point: with coupling it is
        no longer the optimum, which is the whole reason a search is needed."""
        return self.cost.best_greedy()

    def radius_field(self, assignment):
        return self.cost.radius_field(assignment)

    def assigned_obs(self, assignment):
        return self.cost.assigned_obs(assignment)


class CoupledObjective(AssignmentObjective):
    """Budget-counted wrapper around a coupled cost."""

    def __init__(self, cost: "CoupledCost", budget=100_000):
        self.cost = cost
        self.table = cost.table
        self.n_points, self.n_choices = cost.n_points, cost.n_choices
        self.budget = int(budget)
        self.n_evals = 0
        self.trace = []

    def evaluate(self, assignment):
        self.n_evals += 1
        v = self.cost.score(assignment)
        self.trace.append(v)
        return v

    def delta(self, assignment, point, choice):
        self.n_evals += 1
        return self.cost.delta(assignment, point, choice)
