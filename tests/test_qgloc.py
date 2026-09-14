# -*- coding: utf-8 -*-
"""Tests for the properties the results depend on."""
from __future__ import annotations

import numpy as np
import pytest

import qgloc
from qgloc import (LocalCost, NONE, PrecisionBuilder, QGConfig, Testbed,
                   build_candidates, build_model)


@pytest.fixture(scope="module")
def bed():
    cfg = QGConfig(mrefin=4, spinup=200.0, n_snapshots=12, snapshot_every=50.0,
                   ensemble_size=8, cycles=3, burn_in=1)
    model = build_model(cfg)
    n = model.get_number_of_variables()
    x = model.propagate(np.zeros(n), np.array([0.0, cfg.spinup]))
    snaps = [x.copy()]
    for _ in range(cfg.n_snapshots):
        x = model.propagate(x, np.array([0.0, cfg.snapshot_every]))
        snaps.append(x.copy())
    return Testbed(cfg, np.array(snaps))


# ----------------------------------------------------------------------
# Testbed
# ----------------------------------------------------------------------
def test_propagation_ignores_the_psi_block(bed):
    """The measurement the whole design rests on: halving psi and leaving q
    alone gives a bit-identical propagation, so psi is not prognostic and an
    analysis correcting it would be discarded."""
    x = bed.snapshots[0].copy()
    y = x.copy()
    y[bed.blocks["psi"]] *= 0.5
    T = np.array([0.0, 20.0])
    assert np.array_equal(bed.model.propagate(x, T), bed.model.propagate(y, T))


def test_psi_is_rebuilt_from_q(bed):
    x = bed.snapshots[0].copy()
    x[bed.blocks["psi"]] *= 0.5
    y = bed.psi_from_q(x)
    g, h, F = bed.g, 1.0 / (bed.g - 1), 1600.0
    psi = y[bed.blocks["psi"]].reshape(g, g)
    q = y[bed.qblock].reshape(g, g)
    lap = np.zeros_like(psi)
    lap[1:-1, 1:-1] = (psi[2:, 1:-1] + psi[:-2, 1:-1] + psi[1:-1, 2:]
                       + psi[1:-1, :-2] - 4 * psi[1:-1, 1:-1]) / h ** 2
    res = (lap - F * psi - q)[1:-1, 1:-1]
    assert np.linalg.norm(res) / np.linalg.norm(q[1:-1, 1:-1]) < 1e-8


def test_members_are_states_of_the_attractor(bed):
    """Every member is a snapshot of the long run, not a perturbed state: a
    perturbation is mostly energy off the attractor and the hyperviscosity
    removes it."""
    X, x_true = bed.build_ensemble(0)
    rows = {tuple(np.round(s[:8], 6)) for s in bed.snapshots}
    for j in range(X.shape[1]):
        assert tuple(np.round(X[:8, j], 6)) in rows
    assert tuple(np.round(x_true[:8], 6)) in rows
    for j in range(X.shape[1]):
        assert not np.allclose(X[:, j], x_true)


def test_lattice_observes_only_the_prognostic_block(bed):
    idx = bed.checkerboard(3)
    qb = bed.qblock
    assert idx.size > 0
    assert np.all((idx >= qb.start) & (idx < qb.stop))


def test_lattice_offset_preserves_density(bed):
    a = bed.checkerboard(4, offset=0)
    c = bed.checkerboard(4, offset=1)
    assert a.size == c.size
    assert not np.array_equal(a, c)


# ----------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------
def _obs(bed, stride=3):
    idx = bed.checkerboard(stride) - bed.qblock.start
    return idx // bed.g, idx % bed.g, idx


def test_every_component_finds_candidates(bed):
    r, c, _ = _obs(bed)
    cand = build_candidates(bed.g, r, c, n_candidates=4, r_max=12)
    assert cand.n_points == bed.g * bed.g
    n = np.sum(cand.obs != NONE, axis=1)
    assert n.min() >= 1


def test_candidates_are_ordered_by_distance(bed):
    r, c, _ = _obs(bed)
    cand = build_candidates(bed.g, r, c, n_candidates=4, r_max=12)
    d = cand.dist
    ok = cand.obs != NONE
    for i in range(0, cand.n_points, 37):
        dd = d[i][ok[i]]
        assert np.all(np.diff(dd) >= 0)


def test_the_radius_adapts_to_the_observation_density(bed):
    """Nothing sets the radius; it is where the neighbourhood stopped. A
    sparser network must give a larger one."""
    r1, c1, _ = _obs(bed, stride=2)
    r2, c2, _ = _obs(bed, stride=4)
    a = build_candidates(bed.g, r1, c1, 4, 12)
    b = build_candidates(bed.g, r2, c2, 4, 12)
    assert b.dist.max() >= a.dist.max()
    assert b.summary()["radius_mean"] > a.summary()["radius_mean"]


# ----------------------------------------------------------------------
# Local cost
# ----------------------------------------------------------------------
def _cost(bed, stride=3):
    X, x_true = bed.build_ensemble(1)
    qb = bed.qblock
    Q = X[qb, :]
    DX = Q - Q.mean(axis=1, keepdims=True)
    r, c, idx = _obs(bed, stride)
    cand = build_candidates(bed.g, r, c, 4, 12)
    sd = bed.spread["q"]
    y = x_true[qb][idx] + 0.05 * sd * np.random.default_rng(0).standard_normal(idx.size)
    return LocalCost(DX, cand, idx, y, Q.mean(axis=1), (0.05 * sd) ** 2), cand


def test_abstaining_is_available_and_costed(bed):
    """With no observation the background is kept, so the departure term
    vanishes and only the residual remains."""
    lc, cand = _cost(bed)
    n_c = cand.n_candidates
    assert lc.n_choices == n_c + 1
    v = lc.score(np.full(cand.n_points, n_c))
    assert np.isfinite(v) and v > 0


def test_greedy_is_the_exact_optimum(bed):
    """The cost decomposes over components, so the per-component arg-min is the
    global optimum. This is why no search is needed, and why a coupling term is
    what makes the problem combinatorial."""
    lc, cand = _cost(bed)
    g = lc.best_greedy()
    best = lc.score(g)
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.integers(0, lc.n_choices, size=cand.n_points)
        assert lc.score(a) >= best - 1e-12


def test_radius_follows_the_choice(bed):
    """The radius is not chosen: it is the distance to the assigned
    observation."""
    lc, cand = _cost(bed)
    a = lc.best_greedy()
    r = lc.radius_field(a)
    for i in range(0, cand.n_points, 53):
        if a[i] < cand.n_candidates:
            assert r[i] == cand.dist[i, a[i]]
        else:
            assert r[i] == 0


def test_coupling_breaks_separability(bed):
    """With the roughness term the per-component arg-min stops being optimal,
    which is the whole point of adding it."""
    from qgloc import CoupledCost, CoupledObjective, SEARCHES
    lc, cand = _cost(bed)
    cc = CoupledCost(lc, bed.g, lam=20.0)
    obj = CoupledObjective(cc, budget=20000)
    greedy = cc.score(cc.best_greedy())
    _, info = SEARCHES["tabu"](obj, rng=np.random.default_rng(0), iters=40)
    assert info["J"] < greedy


def test_without_coupling_the_searches_agree_with_greedy(bed):
    from qgloc import AssignmentObjective, SEARCHES
    lc, _ = _cost(bed)
    res = {}
    for name in ("greedy", "tabu", "annealing"):
        obj = AssignmentObjective(lc, budget=20000)
        _, info = SEARCHES[name](obj, rng=np.random.default_rng(0))
        res[name] = info["J"]
    assert abs(res["tabu"] - res["greedy"]) < 1e-12
    assert abs(res["annealing"] - res["greedy"]) < 1e-12


# ----------------------------------------------------------------------
# Precision
# ----------------------------------------------------------------------
def test_ridge_must_be_scaled_to_do_anything(bed):
    """An absolute penalty is orders of magnitude below the diagonal of X'X in
    model units. Scaling by the trace makes the estimator invariant to the
    units of the state, which is what the whole filter turned out to depend
    on."""
    from qgloc import ridge_closed_form
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.1 * rng.standard_normal(30)
    b1 = ridge_closed_form(X, y, 0.3, scale=True)
    c = 1000.0
    b2 = ridge_closed_form(c * X, c * y, 0.3, scale=True)
    assert np.allclose(b1, b2, rtol=1e-8)
    # unscaled, the same change of units changes the estimate
    u1 = ridge_closed_form(X, y, 0.3, scale=False)
    u2 = ridge_closed_form(c * X, c * y, 0.3, scale=False)
    assert not np.allclose(u1, u2, rtol=1e-3)


def test_precision_is_positive_definite(bed):
    X, _ = bed.build_ensemble(1)
    qb = bed.qblock
    Q = X[qb, :]
    DX = Q - Q.mean(axis=1, keepdims=True)
    pb = PrecisionBuilder(bed.model, bed.nq, alpha=0.3,
                          active_mask=~bed.boundary_mask()[qb]).bind(DX)
    for r in (1, 3):
        B = pb.build(np.full(bed.nq, float(r)), sparse=False)
        assert np.isfinite(B).all()
        assert np.linalg.eigvalsh(0.5 * (B + B.T)).min() > 0


def test_precision_magnitude_matches_the_ensemble(bed):
    """The failure that hid for a day: with an unscaled ridge the diagonal of
    the precision came out a factor of 1e9 above the inverse ensemble variance,
    so the filter ignored every observation."""
    X, _ = bed.build_ensemble(2)
    qb = bed.qblock
    Q = X[qb, :]
    DX = Q - Q.mean(axis=1, keepdims=True)
    act = ~bed.boundary_mask()[qb]
    pb = PrecisionBuilder(bed.model, bed.nq, alpha=0.3, active_mask=act,
                          var_floor=0.0).bind(DX)
    B = pb.build(np.full(bed.nq, 2.0), sparse=True)
    d = np.median(B.diagonal()[act])
    inv_var = np.median(1.0 / np.maximum(DX.var(axis=1)[act], 1e-300))
    assert 1e-3 < d / inv_var < 1e3


def test_rows_are_cached_per_component_and_radius(bed):
    X, _ = bed.build_ensemble(2)
    qb = bed.qblock
    DX = X[qb, :] - X[qb, :].mean(axis=1, keepdims=True)
    pb = PrecisionBuilder(bed.model, bed.nq, alpha=0.3).bind(DX)
    pb.build(np.full(bed.nq, 3.0))
    first = pb.stats()["builds"]
    pb.build(np.full(bed.nq, 3.0))
    assert pb.stats()["builds"] == first
    r = np.full(bed.nq, 3.0)
    r[:50] = 5.0
    pb.build(r)
    assert pb.stats()["builds"] <= first + 50


def test_cache_is_invalidated_when_the_ensemble_changes(bed):
    X1, _ = bed.build_ensemble(3)
    X2, _ = bed.build_ensemble(4)
    qb = bed.qblock
    pb = PrecisionBuilder(bed.model, bed.nq, alpha=0.3)
    B1 = pb.bind(X1[qb] - X1[qb].mean(axis=1, keepdims=True)).build(np.full(bed.nq, 2.0), sparse=False)
    B2 = pb.bind(X2[qb] - X2[qb].mean(axis=1, keepdims=True)).build(np.full(bed.nq, 2.0), sparse=False)
    assert not np.allclose(B1, B2)
