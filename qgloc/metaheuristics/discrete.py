# -*- coding: utf-8 -*-
"""
Combinatorial searches over the integer radius vector.

The decision variable is a vector of ``2K`` integers: the first ``K`` are the
radii of the clusters of ``q``, the next ``K`` those of ``psi``, each in
``{r_min, ..., r_max}``. With ``K = 4`` and ``r_max = 12`` that is 12^8, about
430 million configurations. Nothing here ever produces a real-valued solution
and nothing is rounded: a vector is born integer and stays integer. The only
place a real number appears is where a Levy draw sets *how many* components to
copy, which is discretizing a counter, not a solution.

All six searches share the same neighbourhood so that a comparison between them
measures the acceptance rule and not the move. The neighbourhood is: pick a
component at random and give it a different integer value. ``step`` controls
how far: with ``step=None`` the new value is drawn uniformly from the whole
range, otherwise from within ``step`` of the current one. Annealing shrinks
``step`` as the temperature falls, which is what makes it a tabu search with an
unrestricted but cooling neighbourhood rather than a different algorithm.

Why the clusters being non-contiguous matters here. They come from k-means on
ensemble features, not from position, so a single component controls grid
points scattered over the whole domain and the components are close to
independent of one another. That favours moves which change few components at a
time, and it is why ant colony optimization fits this problem so naturally:
its pheromone accumulates evidence about each (component, value) pair
separately, and an ant can combine the good value of cluster 3 with the good
value of cluster 7 even though they have never appeared together.
"""
from __future__ import annotations

import numpy as np

from .base import INF, make_info, out_of_budget, safe_eval


# ----------------------------------------------------------------------
def random_vector(K, r_min, r_max, rng):
    return rng.integers(r_min, r_max + 1, size=K)


def neighbour(x, rng, r_min, r_max, step=None):
    """Change one component to a different integer value."""
    y = x.copy()
    j = int(rng.integers(x.size))
    if step is None:
        choices = [v for v in range(r_min, r_max + 1) if v != y[j]]
    else:
        lo, hi = max(r_min, y[j] - step), min(r_max, y[j] + step)
        choices = [v for v in range(lo, hi + 1) if v != y[j]]
    if not choices:
        return y
    y[j] = int(rng.choice(choices))
    return y


def _bounds(lo, hi):
    lo = np.atleast_1d(np.asarray(lo))
    hi = np.atleast_1d(np.asarray(hi))
    return lo.size, int(lo[0]), int(hi[0])


def _start(x0, K, r_min, r_max, rng):
    if x0 is None:
        return random_vector(K, r_min, r_max, rng)
    return np.clip(np.rint(np.asarray(x0, dtype=float)).astype(int),
                   r_min, r_max)


# ----------------------------------------------------------------------
def optimize_tabu(obj, lo, hi, budget, rng, x0=None, tenure=8, iters=500,
                  n_candidates=None, aspiration=True, **_):
    """Tabu search: best non-tabu neighbour, with aspiration.

    The neighbourhood is scanned rather than sampled: for the chosen
    components, every admissible value is tried. A move is forbidden for
    ``tenure`` iterations unless it improves on the incumbent best.
    """
    K, r_min, r_max = _bounds(lo, hi)
    x = _start(x0, K, r_min, r_max, rng)
    f = safe_eval(obj, x)
    best_x, best_f = x.copy(), f
    tabu = {}
    n_cand = n_candidates or K

    for it in range(int(iters)):
        if out_of_budget(obj):
            break
        cand = (None, INF, None)
        for j in rng.permutation(K)[:n_cand]:
            for v in range(r_min, r_max + 1):
                if v == x[j] or out_of_budget(obj):
                    continue
                is_tabu = tabu.get((int(j), v), 0) > it
                t = x.copy()
                t[j] = v
                ft = safe_eval(obj, t)
                if is_tabu and not (aspiration and ft < best_f):
                    continue
                if ft < cand[1]:
                    cand = ((int(j), v), ft, t)
        if cand[0] is None:
            break
        (j, _), f, x = cand
        tabu[(j, int(x[j]))] = it + int(tenure)
        if f < best_f:
            best_x, best_f = x.copy(), f

    return best_x, make_info(obj, best_x, best_f,
                             dict(optimizer="tabu", tenure=int(tenure)))


def optimize_annealing(obj, lo, hi, budget, rng, x0=None, T0=None,
                       cooling=0.95, level_iters=None, step0=None,
                       calib_moves=8, **_):
    """Simulated annealing on the same neighbourhood as tabu.

    The two differ only in the acceptance rule: tabu forbids revisiting,
    annealing accepts a worsening move with probability exp(-delta/T). The step
    shrinks with temperature, so the neighbourhood starts unrestricted and ends
    at nearest values.
    """
    K, r_min, r_max = _bounds(lo, hi)
    x = _start(x0, K, r_min, r_max, rng)
    f = safe_eval(obj, x)
    best_x, best_f = x.copy(), f
    span = r_max - r_min
    step = int(step0) if step0 else span
    level_iters = level_iters or max(2 * K, 10)

    if T0 is None:
        deltas, px, pf = [], x.copy(), f
        for _ in range(calib_moves):
            if out_of_budget(obj):
                break
            c = neighbour(px, rng, r_min, r_max, step)
            cf = safe_eval(obj, c)
            if np.isfinite(cf):
                if cf > pf:
                    deltas.append(cf - pf)
                px, pf = c, cf
                if cf < best_f:
                    best_x, best_f = c.copy(), cf
        mean_delta = float(np.mean(deltas)) if deltas else max(abs(f) * 1e-2, 1e-9)
        T0 = -mean_delta / np.log(0.8)

    T = float(T0)
    n_accept = n_uphill = 0
    while not out_of_budget(obj) and T > 1e-12:
        for _ in range(level_iters):
            if out_of_budget(obj):
                break
            c = neighbour(x, rng, r_min, r_max, max(1, step))
            cf = safe_eval(obj, c)
            if cf == INF:
                break
            d = cf - f
            if d <= 0 or rng.random() < np.exp(-d / T):
                x, f = c, cf
                n_accept += 1
                n_uphill += int(d > 0)
            if f < best_f:
                best_x, best_f = x.copy(), f
        T *= cooling
        step = max(1, int(round(span * T / T0))) if T0 > 0 else 1

    return best_x, make_info(obj, best_x, best_f,
                             dict(optimizer="annealing", T0=float(T0),
                                  n_accept=n_accept, n_uphill=n_uphill))


def optimize_genetic(obj, lo, hi, budget, rng, x0=None, pop_size=15,
                     p_mut=0.25, tournament=2, elite=1, **_):
    """Real integer GA: uniform crossover on components, one-gene mutation.

    Uniform crossover rather than an interpolating one: the components are
    integers with no meaningful midpoint, and exchanging whole genes is what
    preserves the good value a cluster has found.
    """
    K, r_min, r_max = _bounds(lo, hi)
    pop = np.stack([random_vector(K, r_min, r_max, rng)
                    for _ in range(pop_size)])
    if x0 is not None:
        pop[0] = _start(x0, K, r_min, r_max, rng)
    fit = np.array([safe_eval(obj, p) for p in pop])
    g = int(np.argmin(fit))
    best_x, best_f = pop[g].copy(), float(fit[g])

    n_gen = 0
    while not out_of_budget(obj):
        n_gen += 1
        order = np.argsort(fit)
        new_pop = [pop[i].copy() for i in order[:elite]]
        new_fit = [fit[i] for i in order[:elite]]
        while len(new_pop) < pop_size and not out_of_budget(obj):
            parents = []
            for _ in range(2):
                c = rng.choice(pop_size, size=tournament, replace=False)
                parents.append(pop[c[int(np.argmin(fit[c]))]])
            mask = rng.random(K) < 0.5
            child = np.where(mask, parents[0], parents[1])
            if rng.random() < p_mut:
                child = neighbour(child, rng, r_min, r_max)
            cf = safe_eval(obj, child)
            if cf == INF:
                break
            new_pop.append(child)
            new_fit.append(cf)
            if cf < best_f:
                best_x, best_f = child.copy(), cf
        if len(new_pop) < pop_size:
            break
        pop, fit = np.stack(new_pop), np.array(new_fit)

    return best_x, make_info(obj, best_x, best_f,
                             dict(optimizer="genetic", pop_size=int(pop_size),
                                  n_gen=n_gen))


def optimize_ant(obj, lo, hi, budget, rng, x0=None, n_ants=10, rho=0.1,
                 alpha=1.0, tau0=1.0, elitist=True, **_):
    """Ant colony over (component, value) pairs.

    A pheromone table of shape (K, r_max - r_min + 1). Each ant builds a whole
    vector by choosing, for every component independently, a value with
    probability proportional to its pheromone. The best ants then reinforce the
    pairs they used, and everything evaporates.

    This is the search that fits the problem best. Because the clusters come
    from ensemble features rather than position, their radii are close to
    independent, and the pheromone accumulates evidence about each one
    separately: an ant can combine the good value of one cluster with the good
    value of another even though the two have never appeared together.
    """
    K, r_min, r_max = _bounds(lo, hi)
    n_vals = r_max - r_min + 1
    tau = np.full((K, n_vals), float(tau0))
    best_x, best_f = None, INF

    n_iter = 0
    while not out_of_budget(obj):
        n_iter += 1
        sols, vals = [], []
        for _ in range(n_ants):
            if out_of_budget(obj):
                break
            p = tau ** alpha
            p = p / p.sum(axis=1, keepdims=True)
            x = np.array([r_min + int(rng.choice(n_vals, p=p[j]))
                          for j in range(K)])
            f = safe_eval(obj, x)
            if f == INF:
                break
            sols.append(x)
            vals.append(f)
            if f < best_f:
                best_x, best_f = x.copy(), f
        if not sols:
            break
        tau *= (1.0 - rho)
        order = np.argsort(vals)
        for rank in order[:max(1, len(sols) // 3)]:
            x, f = sols[rank], vals[rank]
            w = 1.0 / (1.0 + f)
            for j in range(K):
                tau[j, x[j] - r_min] += w
        if elitist and best_x is not None:
            for j in range(K):
                tau[j, best_x[j] - r_min] += 1.0 / (1.0 + best_f)
        tau = np.clip(tau, 1e-6, 1e6)

    if best_x is None:
        best_x = random_vector(K, r_min, r_max, rng)
        best_f = safe_eval(obj, best_x)
    return best_x, make_info(obj, best_x, best_f,
                             dict(optimizer="ant", n_ants=int(n_ants),
                                  rho=float(rho), n_iter=n_iter))


def _levy_count(K, lam, rng, scale=0.5):
    """How many components a global move copies, from a Levy draw.

    This is the one place a real number appears, and it sets a counter rather
    than a solution: the heavy tail means one or two components most of the
    time and occasionally many, which is the property the Levy step is there
    for.
    """
    from math import gamma as gamma_fn
    from math import pi, sin
    sigma = (gamma_fn(1 + lam) * sin(pi * lam / 2)
             / (gamma_fn((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = rng.standard_normal() * sigma
    v = rng.standard_normal()
    step = abs(u / abs(v) ** (1 / lam))
    return int(np.clip(round(scale * step) + 1, 1, K))


def optimize_fpa(obj, lo, hi, budget, rng, x0=None, pop_size=15, switch_p=0.8,
                 lam=1.5, levy_scale=0.5, **_):
    """Discrete flower pollination.

    Both moves become copies of components, so no arithmetic on the radii is
    ever performed and nothing is rounded.

    Global pollination: a Levy draw gives a number of components, and that many
    randomly chosen components are copied from the incumbent best. The heavy
    tail gives mostly small moves with occasional large ones, which is what the
    Levy flight contributes in the continuous algorithm.

    Local pollination: two candidates are drawn, the components in which they
    differ are found, and a random subset of those is copied from one to the
    other. The move is confined to variability the population already has.

    Following the reference implementation, the global step is taken when
    ``rand > p``, so ``p = 0.8`` means global pollination one time in five.
    """
    K, r_min, r_max = _bounds(lo, hi)
    pop = np.stack([random_vector(K, r_min, r_max, rng)
                    for _ in range(pop_size)])
    if x0 is not None:
        pop[0] = _start(x0, K, r_min, r_max, rng)
    fit = np.array([safe_eval(obj, p) for p in pop])
    g = int(np.argmin(fit))
    best_x, best_f = pop[g].copy(), float(fit[g])

    n_global = n_local = n_iter = 0
    while not out_of_budget(obj):
        n_iter += 1
        for s in range(pop_size):
            if out_of_budget(obj):
                break
            if rng.random() > switch_p:
                m = _levy_count(K, lam, rng, levy_scale)
                idx = rng.choice(K, size=m, replace=False)
                cand = pop[s].copy()
                cand[idx] = best_x[idx]
                n_global += 1
            else:
                u, v = rng.choice(pop_size, size=2, replace=False)
                diff = np.flatnonzero(pop[u] != pop[v])
                cand = pop[s].copy()
                if diff.size:
                    m = 1 + int(rng.integers(diff.size))
                    idx = rng.choice(diff, size=m, replace=False)
                    cand[idx] = pop[u][idx]
                n_local += 1
            cf = safe_eval(obj, cand)
            if cf == INF:
                break
            if cf < fit[s]:
                pop[s], fit[s] = cand, cf
            if cf < best_f:
                best_x, best_f = cand.copy(), cf

    return best_x, make_info(obj, best_x, best_f,
                             dict(optimizer="fpa-discrete",
                                  pop_size=int(pop_size), n_iter=n_iter,
                                  n_global=n_global, n_local=n_local))


def optimize_firefly(obj, lo, hi, budget, rng, x0=None, pop_size=15, beta0=0.9,
                     absorb=1.0, zeta=0.1, **_):
    """Discrete firefly: attraction becomes a probability of copying.

    In the continuous algorithm a firefly moves a fraction ``beta`` of the way
    towards a brighter one, with ``beta`` decaying with distance. Here it
    copies each component of the brighter one with probability ``beta``, where
    the distance is the Hamming distance between the two vectors. The idea of
    the algorithm survives; the arithmetic, which meant nothing on small
    integers, does not.
    """
    K, r_min, r_max = _bounds(lo, hi)
    pop = np.stack([random_vector(K, r_min, r_max, rng)
                    for _ in range(pop_size)])
    if x0 is not None:
        pop[0] = _start(x0, K, r_min, r_max, rng)
    fit = np.array([safe_eval(obj, p) for p in pop])
    g = int(np.argmin(fit))
    best_x, best_f = pop[g].copy(), float(fit[g])

    n_iter = 0
    while not out_of_budget(obj):
        n_iter += 1
        moved = False
        for u in range(pop_size):
            if out_of_budget(obj):
                break
            for v in range(pop_size):
                # A converged population makes every pair fail this test, and
                # without the `moved` guard below the loop would then spin
                # without ever evaluating anything: the budget counter never
                # advances and the search never terminates.
                if fit[v] >= fit[u] or out_of_budget(obj):
                    continue
                moved = True
                ham = float(np.sum(pop[u] != pop[v])) / K
                beta = beta0 * np.exp(-absorb * ham ** 2)
                copy = rng.random(K) < beta
                cand = np.where(copy, pop[v], pop[u])
                if rng.random() < zeta:
                    cand = neighbour(cand, rng, r_min, r_max)
                cf = safe_eval(obj, cand)
                if cf == INF:
                    break
                if cf < fit[u]:
                    pop[u], fit[u] = cand, cf
                if cf < best_f:
                    best_x, best_f = cand.copy(), cf
        if not moved:
            # Fully converged: perturb the worst half rather than stall, which
            # is the discrete analogue of the random walk the continuous
            # firefly applies to the brightest individual.
            for u in np.argsort(fit)[pop_size // 2:]:
                if out_of_budget(obj):
                    break
                cand = neighbour(pop[u], rng, r_min, r_max)
                cf = safe_eval(obj, cand)
                if cf == INF:
                    break
                pop[u], fit[u] = cand, cf
                if cf < best_f:
                    best_x, best_f = cand.copy(), cf
            if out_of_budget(obj):
                break

    return best_x, make_info(obj, best_x, best_f,
                             dict(optimizer="firefly-discrete",
                                  pop_size=int(pop_size), n_iter=n_iter))


# ----------------------------------------------------------------------
# Controls. Not competitors: they calibrate the comparison. A search that does
# not beat random sampling at the same budget is doing nothing, and where the
# space is small enough the exhaustive sweep says what the answer actually is.
# ----------------------------------------------------------------------
def optimize_random(obj, lo, hi, budget, rng, x0=None, **_):
    K, r_min, r_max = _bounds(lo, hi)
    best_x, best_f = None, INF
    while not out_of_budget(obj):
        x = random_vector(K, r_min, r_max, rng)
        f = safe_eval(obj, x)
        if f == INF:
            break
        if f < best_f:
            best_x, best_f = x.copy(), f
    if best_x is None:
        best_x = random_vector(K, r_min, r_max, rng)
        best_f = obj.best_so_far()
    return best_x, make_info(obj, best_x, best_f, dict(optimizer="random"))


def optimize_exhaustive(obj, lo, hi, budget, rng, x0=None, **_):
    """Every integer vector, when the space is small enough to enumerate.

    ``info['complete']`` says whether the enumeration finished. When it did,
    the reported value is the true optimum and every other search can be scored
    as a gap to it rather than as a comparison against its rivals.
    """
    from itertools import product
    K, r_min, r_max = _bounds(lo, hi)
    total = (r_max - r_min + 1) ** K
    best_x, best_f, seen = None, INF, 0
    for combo in product(range(r_min, r_max + 1), repeat=K):
        if out_of_budget(obj):
            break
        x = np.array(combo)
        f = safe_eval(obj, x)
        if f == INF:
            break
        seen += 1
        if f < best_f:
            best_x, best_f = x.copy(), f
    if best_x is None:
        best_x = random_vector(K, r_min, r_max, rng)
        best_f = obj.best_so_far()
    return best_x, make_info(obj, best_x, best_f,
                             dict(optimizer="exhaustive", n_space=int(total),
                                  n_seen=int(seen), complete=bool(seen >= total)))
