# -*- coding: utf-8 -*-
"""Tests for the properties the results depend on."""
from __future__ import annotations

import numpy as np
import pytest

import qgloc
from qgloc import QGConfig, RadiusObjective, RadiusSpec, Testbed, build_model
from qgloc.metaheuristics import METAHEURISTICS, get_optimizer


@pytest.fixture(scope="module")
def bed():
    cfg = QGConfig(mrefin=4, spinup=200.0, ensemble_size=8, cycles=3, burn_in=1)
    model = build_model(cfg)
    x0 = model.propagate(np.zeros(model.get_number_of_variables()),
                         np.array([0.0, cfg.spinup]))
    return Testbed(cfg, x0)


def test_the_two_fields_differ_by_orders_of_magnitude(bed):
    """If they did not, normalizing per field would be unnecessary and the
    whole design would be over-engineering."""
    assert bed.spread["q"] / bed.spread["psi"] > 100


def test_normalization_gives_both_fields_unit_scale(bed):
    z = bed.normalize(bed.x0_ref)
    for k, b in bed.blocks.items():
        assert 0.5 < float(z[b].std()) < 2.0


def test_normalize_and_denormalize_round_trip(bed):
    x = bed.x0_ref
    assert np.allclose(bed.denormalize(bed.normalize(x)), x)
    X = np.column_stack([x, 2 * x])
    assert np.allclose(bed.denormalize(bed.normalize(X)), X)


def test_lattice_density_matches_the_stride(bed):
    """Density is 1/stride^2 by construction, which is what makes the relation
    between density and useful radius a design choice rather than an accident
    of a random draw."""
    for stride in (2, 3, 4):
        idx = bed.checkerboard(stride)
        assert abs(idx.size / bed.n - bed.density(stride)) < 1e-12
        # and denser strides really are denser
    assert bed.density(2) > bed.density(3) > bed.density(4)


def test_lattice_covers_both_fields(bed):
    idx = bed.checkerboard(3)
    for k, b in bed.blocks.items():
        assert np.any((idx >= b.start) & (idx < b.stop))


def test_lattice_offset_moves_the_network_without_changing_density(bed):
    a = bed.checkerboard(4, offset=0)
    c = bed.checkerboard(4, offset=1)
    assert a.size == c.size
    assert not np.array_equal(a, c)


def test_perturbation_scales_with_each_field(bed):
    """An absolute perturbation is enormous for psi and negligible for q, which
    produces an ensemble with no usable background error at all."""
    p = bed.perturbation(np.random.default_rng(0), 0.1)
    for k, b in bed.blocks.items():
        assert 0.05 < float(p[b].std()) / bed.spread[k] < 0.2


def test_radius_spec_expands_to_one_value_per_component(bed):
    for kind, K, labels in (("uniform", 1, None), ("per_field", 2, None)):
        spec = RadiusSpec(bed, kind=kind)
        r = spec.expand(np.full(spec.K, 3.0))
        assert r.shape == (bed.n,)
        assert np.allclose(r, 3.0)


def test_clustered_spec_gives_one_radius_per_cluster(bed):
    labels = {k: np.arange(bed.blocks[k].stop - bed.blocks[k].start) % 3
              for k in bed.blocks}
    spec = RadiusSpec(bed, kind="clustered", labels=labels)
    assert spec.K == 6
    theta = np.arange(1, 7, dtype=float)
    r = spec.expand(theta)
    assert r.shape == (bed.n,)
    assert set(np.unique(r)) == set(theta)


def test_parameterizations_are_nested(bed):
    """A uniform radius must be reachable by every richer parameterization, or
    a comparison between them measures the code path and not the idea."""
    labels = {k: np.arange(bed.blocks[k].stop - bed.blocks[k].start) % 3
              for k in bed.blocks}
    uni = RadiusSpec(bed, kind="uniform").expand([4.0])
    per = RadiusSpec(bed, kind="per_field").expand([4.0, 4.0])
    clu = RadiusSpec(bed, kind="clustered", labels=labels).expand(np.full(6, 4.0))
    assert np.allclose(uni, per) and np.allclose(uni, clu)


def test_cluster_labels_cover_every_point(bed):
    X, _ = bed.build_ensemble(1)
    lab = qgloc.cluster_labels(X, bed.blocks["q"], bed.g, K=3)
    assert lab.size == bed.blocks["q"].stop - bed.blocks["q"].start
    assert len(np.unique(lab)) <= 3


def test_objective_counts_budget_and_caches_integers(bed):
    calls = {"n": 0}

    def ev(theta):
        calls["n"] += 1
        return float(np.sum(theta))

    obj = RadiusObjective(ev, budget=3, r_min=1, r_max=12)
    obj(np.array([2.0])); obj(np.array([5.0]))
    assert obj.n_evals == 2
    obj(np.array([2.4]))          # floors to 2, already seen
    assert obj.n_evals == 2 and obj.n_cache_hits == 1
    obj(np.array([7.0]))
    with pytest.raises(qgloc.BudgetExhausted):
        obj(np.array([9.0]))


def test_objective_records_the_search(bed):
    obj = RadiusObjective(lambda t: float(np.sum(t)), budget=5)
    for v in (3.0, 1.0, 2.0):
        obj(np.array([v]))
    assert len(obj.history()) == 3
    assert list(obj.best_trace()) == [3.0, 1.0, 1.0]


@pytest.mark.parametrize("name", METAHEURISTICS + ["random"])
def test_every_search_respects_the_budget(bed, name):
    budget = 20
    obj = RadiusObjective(lambda t: float(np.sum((t - 4.0) ** 2)), budget=budget)
    theta, info = get_optimizer(name)(
        obj, np.full(3, 1.0), np.full(3, 12.0), budget,
        np.random.default_rng(0), **qgloc.defaults_for(name))
    assert obj.n_evals <= budget
    assert np.all((np.asarray(theta) >= 1) & (np.asarray(theta) <= 12))


@pytest.mark.parametrize("name", METAHEURISTICS)
def test_every_search_improves_on_its_start(bed, name):
    obj = RadiusObjective(lambda t: float(np.sum((t - 5.0) ** 2)), budget=60)
    theta, info = get_optimizer(name)(
        obj, np.full(3, 1.0), np.full(3, 12.0), 60,
        np.random.default_rng(1), **qgloc.defaults_for(name))
    assert info["J"] <= obj.trace[0] + 1e-9


def test_tabu_moves_one_component_at_a_time(bed):
    """The move is what makes tabu the right search here: one cluster changes,
    so only the part of the analysis that depends on it is affected."""
    seen = []

    def ev(theta):
        seen.append(np.asarray(theta, dtype=int).copy())
        return float(np.sum((np.asarray(theta) - 6.0) ** 2))

    obj = RadiusObjective(ev, budget=40)
    get_optimizer("tabu")(obj, np.full(4, 1.0), np.full(4, 8.0), 40,
                          np.random.default_rng(0), iters=5)
    diffs = [int(np.sum(a != b)) for a, b in zip(seen[1:], seen[:-1])]
    assert min(diffs) == 1


def test_perturbation_vanishes_on_the_dirichlet_boundary(bed):
    """psi is zero on the boundary; a perturbation that is not breaks the
    condition and plants spurious variance exactly where the local analysis
    has fewest neighbours."""
    p = bed.perturbation(np.random.default_rng(0), 0.1)
    edge = bed.boundary_mask()
    b = bed.blocks["psi"]
    psi_edge = p[b][edge[b]]
    assert np.allclose(psi_edge, 0.0)
    assert np.abs(p[b][~edge[b]]).max() > 0


def test_perturbation_respects_the_helmholtz_relation(bed):
    """q is not free: perturbing it independently of psi produces states off
    the constraint and an ensemble whose cross-field correlation is weaker
    than the model's own."""
    rng = np.random.default_rng(1)
    X = np.column_stack([bed.x0_ref + bed.perturbation(rng, 0.1)
                         for _ in range(8)])
    A = X - X.mean(axis=1, keepdims=True)
    q = A[bed.blocks["q"]]
    ps = A[bed.blocks["psi"]]
    c = [np.corrcoef(q[j], ps[j])[0, 1]
         for j in range(0, q.shape[0], 53)]
    assert np.nanmean(np.abs(c)) > 0.8


def test_the_ensemble_is_calibrated(bed):
    """Spread and background error should be of the same size. Far below one
    means the filter trusts a background it has no right to trust, and no
    amount of tuning the radius fixes that."""
    X, x_true = bed.build_ensemble(3)
    Xn, xtn = bed.normalize(X), bed.normalize(x_true)
    spread = float(np.mean(np.std(Xn, axis=1)))
    error = float(np.sqrt(np.mean((Xn.mean(axis=1) - xtn) ** 2)))
    assert 0.4 < spread / error < 2.5
