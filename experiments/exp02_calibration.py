# -*- coding: utf-8 -*-
"""
EXP-02-CALIBRATION

Fix the search parameters once, before anything is compared.

A comparison between metaheuristics is worthless if one of them was tuned and
the others were given whatever the literature's default happened to be. Every
search here therefore gets the same treatment: a small grid over its own
parameters, scored on the same objective, on cycles that no reported experiment
uses.

Two rules make the calibration honest.

**Held-out cycles.** The forecasts used here are drawn with seeds disjoint from
those of EXP-01 and EXP-03, so no search is tuned on the data it is later
reported on.

**The admissible criterion, not the truth.** Parameters are chosen by the value
of the cross-validated objective they attain, never by the analysis error. A
parameter set picked by looking at the truth would smuggle the oracle into a
method that claims not to need it.

Convergence is recorded, not just the final value. ``traces.csv`` holds every
evaluation of every run --- search, parameter set, problem, repeat, step, the
value and the running minimum --- so the convergence curves can be drawn with
their mean and standard deviation across problems and repeats, and so a
parameter set that reaches the same objective in a third of the evaluations can
be told from one that merely gets there.

The winner of each grid is written to ``results/frozen_params.json``, which
EXP-03 reads. It runs after EXP-01 because the regime sweep says which range of
radii is actually useful, and calibrating over a range wider than that would
spend the grid on values no run will visit. Re-running this file changes every later experiment
without touching their code, and the paper can state that no constant in it was
chosen by hand.
"""
from __future__ import annotations

import itertools
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from common import (CACHE_ROOT, ExperimentContext, get_scale, latex_table,
                    make_testbed, method_allowed, parse_cli, RESULTS_ROOT,
                    setup_matplotlib, shard_cells)
from qgloc import (RadiusObjective, RadiusSpec, analyse_fast, cluster_auto,
                   forecast, get_optimizer)
from qgloc.metaheuristics import LABELS, METAHEURISTICS
from qgloc.progress import Progress, log

EXP_ID = "EXP-02-CALIBRATION"
DESCRIPTION = ("Search parameters calibrated once, on held-out cycles, against "
               "the admissible criterion.")

FROZEN = os.path.join(RESULTS_ROOT, "frozen_params.json")


def grids(scale):
    """The parameter grid of each search.

    Kept small and of comparable size across searches: a grid of forty
    combinations for one method and four for another would itself be a form of
    favouritism.
    """
    return {
        "tabu": dict(tenure=[4, 8, 16], iters=[200, 500]),
        "annealing": dict(cooling=[0.90, 0.95, 0.99],
                          level_iters=[None, 20]),
        "genetic": dict(pop_size=[10, 15, 25], p_mut=[0.15, 0.25, 0.40]),
        "ant": dict(n_ants=[6, 10, 20], rho=[0.05, 0.1, 0.3]),
        "fpa": dict(pop_size=[10, 15, 25], switch_p=[0.6, 0.8],
                    levy_scale=[0.25, 0.5]),
        "firefly": dict(pop_size=[10, 15, 25], beta0=[0.6, 0.9],
                        zeta=[0.05, 0.2]),
    }


def combos(grid):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def build_problems(bed, scale, n_problems, seed_base=770_000):
    """A few frozen optimization problems: forecast, clusters, split.

    Frozen on purpose. Every parameter set is scored on exactly the same
    problems, so the comparison is between parameters and not between the
    forecasts they happened to be handed.
    """
    problems = []
    i = 0
    while len(problems) < n_problems and i < 4 * n_problems:
        seed = seed_base + 101 * i
        i += 1
        X, x_true = bed.build_ensemble(seed)
        rng = np.random.default_rng(seed)
        try:
            fc = forecast(bed, X, x_true, rng, stride=scale.stride)
        except RuntimeError:
            # A forecast that cannot be propagated is skipped rather than
            # allowed to end the calibration; the next seed is tried.
            continue

        lab, _ = cluster_auto(fc["Xf"], bed.qblock, bed.g,
                              K_range=scale.K_range, seed=seed)
        spec = RadiusSpec(bed, kind="clustered", labels={"q": lab})

        n_obs = fc["idx"].size
        n_hold = max(4, int(round(scale.cv_fraction * n_obs)))
        hold = np.sort(rng.choice(n_obs, size=n_hold, replace=False))
        keep = np.setdiff1d(np.arange(n_obs), hold)

        def make_cv(fc=fc, spec=spec, keep=keep, hold=hold):
            def cv(theta):
                xa = analyse_fast(bed, fc, spec.expand(theta),
                                  mask=keep).mean(axis=1)
                res = fc["y"][hold] - bed.normalize(xa)[fc["idx"][hold]]
                return float(res @ res) / res.size
            return cv

        problems.append(dict(cv=make_cv(), K=spec.K, seed=seed))
    return problems


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(
        EXP_ID, DESCRIPTION, scale,
        extra_config=dict(budget=scale.budget,
                          n_problems=scale.calib_problems,
                          n_repeats=scale.calib_repeats))

    bed = make_testbed(scale)
    problems = build_problems(bed, scale, scale.calib_problems)
    log(f"{len(problems)} held-out problems, K = "
        f"{[p['K'] for p in problems]}", EXP_ID)

    G = grids(scale)
    cells = shard_cells([(s, cfg, pi, rep)
                         for s in METAHEURISTICS if method_allowed(LABELS[s])
                         for cfg in combos(G[s])
                         for pi in range(len(problems))
                         for rep in range(scale.calib_repeats)])

    rows, traces = [], []
    prog = Progress(len(cells), label="cells", exp_id=EXP_ID, every=10,
                    heartbeat_s=60.0)
    for search, cfg, pi, rep in cells:
        prob = problems[pi]
        obj = RadiusObjective(prob["cv"], budget=scale.budget, r_min=1,
                              r_max=scale.r_max)
        t0 = time.perf_counter()
        _, info = get_optimizer(search)(
            obj, np.full(prob["K"], 1), np.full(prob["K"], scale.r_max),
            scale.budget, np.random.default_rng(4_000 + 13 * rep + 7 * pi),
            **cfg)
        rows.append(dict(exp_id=EXP_ID, search=search,
                         label=LABELS[search], problem=pi, repeat=rep,
                         K=prob["K"], J=float(info["J"]),
                         n_evals=int(obj.n_evals),
                         seconds=time.perf_counter() - t0,
                         params=json.dumps(cfg),
                         **{f"p_{k}": v for k, v in cfg.items()}))

        # The whole trace, not just the final value. Convergence is half of
        # what a calibration is for: a parameter set that reaches the same
        # objective in a third of the evaluations is the better one, and
        # nothing in the summary would show it.
        best = obj.best_trace()
        for step, (raw, run_min) in enumerate(zip(obj.trace, best)):
            traces.append(dict(search=search, label=LABELS[search],
                               params=json.dumps(cfg), problem=pi, repeat=rep,
                               step=step, value=float(raw),
                               best_so_far=float(run_min)))
        prog.step(f"{LABELS[search]:<20s} {cfg} problem={pi}")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    tr = pd.DataFrame(traces)
    ctx.save_table(tr, "traces.csv")
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(df)))
        return

    # Normalize within each problem: a problem with a naturally larger
    # objective would otherwise dominate the average.
    df["J_rel"] = df.groupby("problem")["J"].transform(
        lambda s: s / s.min() if s.min() > 0 else s)

    # Evaluations needed to get within 1% of the value the run ended at: the
    # convergence number that goes next to the final objective.
    def to_within(g, tol=0.01):
        b = g.sort_values("step")["best_so_far"].values
        if b.size == 0:
            return np.nan
        target = b[-1] * (1.0 + tol) if b[-1] > 0 else b[-1]
        idx = np.flatnonzero(b <= target)
        return float(idx[0] + 1) if idx.size else float(b.size)

    conv = (tr.groupby(["search", "params", "problem", "repeat"])
            .apply(to_within, include_groups=False)
            .rename("evals_to_1pct").reset_index())
    df = df.merge(conv, on=["search", "params", "problem", "repeat"],
                  how="left")

    summary = (df.groupby(["search", "label", "params"])
               .agg(J_rel=("J_rel", "mean"), J=("J", "mean"),
                    J_sd=("J", "std"),
                    evals_to_1pct=("evals_to_1pct", "mean"),
                    evals_to_1pct_sd=("evals_to_1pct", "std"),
                    seconds=("seconds", "mean"), n=("J", "count"))
               .reset_index().sort_values(["search", "J_rel"]))
    ctx.save_table(summary, "summary.csv")
    print()
    for s, g in summary.groupby("search"):
        print(f"\n{LABELS[s]}")
        print(g[["params", "J_rel", "J", "seconds"]].head(5).to_string(index=False))

    frozen = {}
    for s, g in summary.groupby("search"):
        best = g.sort_values("J_rel").iloc[0]
        frozen[s] = json.loads(best["params"])
        log(f"frozen {LABELS[s]}: {frozen[s]}  (J_rel {best['J_rel']:.4f})",
            EXP_ID)

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    payload = dict(scale=scale.name, written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   params=frozen)
    with open(FROZEN, "w") as fh:
        json.dump(payload, fh, indent=2)
    log(f"wrote {FROZEN}", EXP_ID)
    ctx.save_json(payload, "frozen_params.json")

    # ---- figure 1: attained objective per parameter set ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (s, g) in zip(axes.ravel(), summary.groupby("search")):
        g = g.sort_values("J_rel")
        ax.barh(range(len(g)), g["J_rel"], color="tab:blue")
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([p[:38] for p in g["params"]], fontsize=5)
        ax.set_xlabel("attained $J$ / best of problem")
        ax.set_title(LABELS[s], fontsize=10)
        ax.invert_yaxis()
    fig.suptitle("EXP-03: parameter calibration on held-out problems, "
                 "scored on the admissible criterion")
    fig.tight_layout()
    ctx.save_fig(fig, "calibration.png")

    # ---- figure 2: convergence, mean and standard deviation ----
    # Normalized per problem so that curves from problems with different
    # objective levels can be averaged at all.
    tr = tr.merge(tr.groupby("problem")["best_so_far"].min()
                  .rename("J_floor"), on="problem")
    tr["norm"] = tr["best_so_far"] / tr["J_floor"].where(tr["J_floor"] > 0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for ax, (s, g) in zip(axes.ravel(), tr.groupby("search")):
        best_params = summary[summary["search"] == s].iloc[0]["params"]
        for prm, gg in g.groupby("params"):
            stat = gg.groupby("step")["norm"].agg(["mean", "std", "count"])
            stat = stat[stat["count"] >= 2]
            if stat.empty:
                continue
            is_best = (prm == best_params)
            ax.plot(stat.index, stat["mean"], lw=1.6 if is_best else 0.7,
                    color="tab:red" if is_best else "0.6",
                    label=prm[:30] if is_best else None, zorder=3 if is_best else 1)
            if is_best:
                ax.fill_between(stat.index, stat["mean"] - stat["std"],
                                stat["mean"] + stat["std"], color="tab:red",
                                alpha=0.18, zorder=2)
        ax.set_title(LABELS[s], fontsize=10)
        ax.set_ylabel("best $J$ / best of problem")
        ax.set_xlabel("objective evaluations")
        ax.legend(fontsize=6)
    fig.suptitle("EXP-03: convergence per parameter set. Red is the selected "
                 "one, with one standard deviation across problems and repeats")
    fig.tight_layout()
    ctx.save_fig(fig, "convergence.png")

    # ---- figure 3: all searches, selected parameters, side by side ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for s, g in tr.groupby("search"):
        best_params = summary[summary["search"] == s].iloc[0]["params"]
        gg = g[g["params"] == best_params]
        stat = gg.groupby("step")["norm"].agg(["mean", "std", "count"])
        stat = stat[stat["count"] >= 2]
        if stat.empty:
            continue
        ax.plot(stat.index, stat["mean"], lw=1.4, label=LABELS[s])
        ax.fill_between(stat.index, stat["mean"] - stat["std"],
                        stat["mean"] + stat["std"], alpha=0.15)
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("best $J$ / best of problem")
    ax.set_title("Convergence of each search at its calibrated parameters "
                 "(mean $\\pm$ 1 s.d.)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    ctx.save_fig(fig, "convergence_all.png")

    ctx.save_latex(latex_table(
        summary.groupby("search").head(1)[["label", "params", "J_rel",
                                           "evals_to_1pct"]],
        "Calibrated search parameters, with the evaluations each needs to come "
        "within 1\\% of its final value.", "tab:calibration"),
        "calibration.tex")

    ctx.finish(summary=dict(n_cells=len(df), frozen=frozen))


if __name__ == "__main__":
    main(parse_cli())
