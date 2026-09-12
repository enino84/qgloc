# -*- coding: utf-8 -*-
"""
EXP-03-CYCLED

The radius estimated inside every assimilation cycle.

At each cycle the forecast ensemble is formed, its features are computed, the
grid points of each field are grouped by silhouette, part of the observation
lattice is withheld, and the vector of ``2K`` integer radii is optimized
against what was withheld. The winner is used for that cycle's analysis and the
ensemble is propagated to the next one, so the radius the filter runs at is
estimated and never tuned.

What makes this affordable
--------------------------
Row separability: row ``i`` of the Cholesky factor depends on ``r_i`` and on
nothing else, so a move that changes one cluster recomputes only that cluster's
rows. With the rows cached, one objective evaluation costs 0.45 s at
n = 4802 against 2.84 s through the reference implementation, and of that only
0.064 s is the precision -- the rest is the linear solve, which the cache does
not touch. A search of a hundred evaluations is then 45 s per cycle.

What is recorded
----------------
Everything needed to redraw a figure without rerunning anything, per cycle:

* the analysis and background means, and the truth;
* RMSE and relative error per field for both, plus ensemble spread;
* the cluster labels of each field and the K that silhouette chose;
* the winning radius vector and its expansion over the domain;
* **every evaluation the search made** -- step, the integer vector, and its
  objective value.

The last item is the point of the experiment as much as the error curves are:
it is what shows how the cross-validated objective was minimized inside a
single cycle, and how that differs between metaheuristics.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from common import (ExperimentContext, get_scale, latex_table, make_testbed,
                    method_allowed, parse_cli, setup_matplotlib, shard_cells)
from common import load_frozen
from qgloc import (RadiusObjective, RadiusSpec, SnapshotWriter, analyse_fast,
                   cluster_auto, defaults_for, forecast, get_optimizer)
from qgloc.metaheuristics import CONTROLS, LABELS, METAHEURISTICS
from qgloc.progress import Progress

EXP_ID = "EXP-03-CYCLED"
DESCRIPTION = ("The localization radius estimated by metaheuristic search "
               "inside every assimilation cycle, against a criterion that "
               "uses observations only.")


def split_lattice(n_obs, fraction, rng):
    """Withhold a fraction of the lattice, once per cycle.

    Once, not per candidate: if the partition moved during the search, the
    candidates would be ranked on different noise realizations, and the
    ranking is what the search consumes.
    """
    n_hold = max(4, int(round(fraction * n_obs)))
    hold = np.sort(rng.choice(n_obs, size=n_hold, replace=False))
    keep = np.setdiff1d(np.arange(n_obs), hold)
    return keep, hold


def run_one(bed, X0, x_true0, search, scale, seed, criterion="cv",
            on_cycle=None):
    """One filter run with the radius re-estimated at every cycle."""
    cfg = bed.cfg
    rng = np.random.default_rng(seed)
    X, x_true = X0.copy(), x_true0.copy()
    params = load_frozen(search)
    rows, history, top = [], [], []
    theta_prev = None

    for k in range(cfg.cycles):
        t0 = time.perf_counter()
        fc = forecast(bed, X, x_true, rng, stride=scale.stride,
                      offset=k % scale.stride)
        x_true = fc["x_true"]
        xb = fc["xb"]

        # Clusters, and how many of them, from the ensemble alone.
        # Only q is clustered, because only q is estimated: psi is recomputed
        # from the analysed q at every step.
        lab, info = cluster_auto(fc["Xf"], bed.qblock, bed.g,
                                 K_range=scale.K_range, seed=seed + k)
        labels = {"q": lab}
        cinfo = {"q": info}
        spec = RadiusSpec(bed, kind="clustered", labels=labels)

        keep, hold = split_lattice(fc["idx"].size, scale.cv_fraction, rng)
        xtn = fc["xtn"]

        def cv(theta):
            xa = analyse_fast(bed, fc, spec.expand(theta), mask=keep).mean(axis=1)
            res = fc["y"][hold] - bed.normalize(xa)[fc["idx"][hold]]
            return float(res @ res) / res.size

        def oracle(theta):
            xa = analyse_fast(bed, fc, spec.expand(theta)).mean(axis=1)
            return float(np.sqrt(np.mean((bed.normalize(xa) - xtn) ** 2)))

        objective = cv if criterion == "cv" else oracle
        obj = RadiusObjective(objective, budget=scale.budget, r_min=1,
                              r_max=scale.r_max)
        theta, info = get_optimizer(search)(
            obj, np.full(spec.K, 1), np.full(spec.K, scale.r_max),
            scale.budget, np.random.default_rng(seed * 131 + k),
            x0=theta_prev if scale.warm_start else None, **params)
        theta = np.asarray(theta, dtype=int)
        # Carry the solution forward only when the parameterization has not
        # changed size. K is chosen by silhouette each cycle, so the number of
        # clusters can move, and a warm start of the wrong length is worse
        # than none.
        theta_prev = theta if theta.size == spec.K else None

        r_vec = spec.expand(theta)
        Xa = analyse_fast(bed, fc, r_vec)
        xa = Xa.mean(axis=1)
        elapsed = time.perf_counter() - t0

        rec = dict(cycle=k, K_total=int(spec.K),
                   K_q=int(cinfo["q"]["K"]),
                   silhouette_q=float(cinfo["q"]["silhouette"]),
                   J=float(info["J"]), n_evals=int(obj.n_evals),
                   n_cache_hits=int(obj.n_cache_hits),
                   radius_mean=float(np.mean(r_vec)),
                   radius_std=float(np.std(r_vec)),
                   theta=str(theta.tolist()), p_obs=int(fc["idx"].size),
                   n_hold=int(hold.size), seconds=elapsed,
                   spread=float(np.mean(np.std(bed.normalize(Xa), axis=1))))
        rec.update({f"b_{a}": b for a, b in bed.errors(xb, x_true).items()})
        rec.update(bed.errors(xa, x_true))
        # The error the oracle would have achieved on the same forecast, so
        # the gap between what the criterion picks and what was available is
        # available per cycle rather than only in aggregate.
        rec["rmse_at_theta_oracle"] = oracle(theta)
        rows.append(rec)

        for h in obj.history():
            history.append(dict(cycle=k, step=h["step"], value=h["value"],
                                theta=str(h["theta"])))
        for t in obj.top(scale.n_top):
            top.append(dict(cycle=k, rank=t["rank"], value=t["value"],
                            theta=str(t["theta"])))

        if on_cycle is not None:
            on_cycle(k, fc, xb, xa, Xa, x_true, labels, r_vec, rec, obj)
        X = Xa

    return rows, history, top


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(
        EXP_ID, DESCRIPTION, scale,
        extra_config=dict(searches=METAHEURISTICS + CONTROLS,
                          K_range=list(scale.K_range),
                          budget=scale.budget, stride=scale.stride,
                          cv_fraction=scale.cv_fraction,
                          ensemble_sizes=list(scale.ensemble_sizes)))

    searches = [s for s in METAHEURISTICS + CONTROLS
                if method_allowed(LABELS[s])]
    cells = shard_cells([(N, s, crit, run)
                         for N in scale.ensemble_sizes
                         for s in searches
                         for crit in ("cv", "oracle")
                         for run in range(scale.runs)])

    snaps = SnapshotWriter()
    rows, hist_rows, top_rows = [], [], []
    prog = Progress(len(cells), label="runs", exp_id=EXP_ID, every=1,
                    heartbeat_s=60.0)

    beds = {}
    for N, search, crit, run in cells:
        if N not in beds:
            beds[N] = make_testbed(scale, ensemble_size=N)
        bed = beds[N]
        X0, xt0 = bed.build_ensemble(scale.seeds[run])

        # Trajectory snapshots for every search, not just one: the radius
        # field each method settles on is a figure in itself, and six maps side
        # by side answer whether they find the same structure. At 3.8 MB per
        # combination the whole set is a few tens of megabytes.
        keep_snaps = (crit == "cv" and run == 0
                      and N == scale.ensemble_sizes[0])

        def on_cycle(k, fc, xb, xa, Xa, x_true, labels, r_vec, rec, obj,
                     _N=N, _s=search, _c=crit):
            if not keep_snaps:
                return
            if k in scale.snapshot_cycles:
                snaps.add(tag=f"N{_N}/{_s}/{_c}/cycle{k}",
                          Xb=fc["Xf"], Xa=Xa, x_true=x_true,
                          obs_idx=fc["idx"], r=r_vec,
                          cycle=k, N=_N, search=_s, criterion=_c,
                          rmse=rec["rmse"])
                # The resulting precision, so the covariance structure the
                # winning radius implies can be drawn without rebuilding it.
                if k in scale.precision_cycles and bed._pb is not None:
                    B = bed._pb.build(r_vec, sparse=True).tocoo()
                    snaps.add(tag=f"precision/N{_N}/{_s}/{_c}/cycle{k}",
                              Xb=np.vstack([B.row, B.col]).astype(np.int32),
                              y=B.data.astype(np.float32),
                              r=r_vec, cycle=k, N=_N, search=_s,
                              criterion=_c, nnz=int(B.nnz), n=int(bed.n))
            snaps.add_trajectory(
                tag=f"fields/N{_N}/{_s}/{_c}/cycle{k}",
                xb_mean=xb[None, :], xa_mean=xa[None, :],
                x_true=x_true[None, :],
                metrics=dict(radius=r_vec,
                             labels_q=labels["q"].astype(float)),
                cycle=k, N=_N, search=_s, criterion=_c)

        t0 = time.time()
        cyc, hist, tops = run_one(bed, X0, xt0, search, scale,
                            seed=9000 + 37 * run, criterion=crit,
                            on_cycle=on_cycle)
        for rec in cyc:
            rows.append(dict(exp_id=EXP_ID, N=N, search=LABELS[search],
                             is_control=search in CONTROLS, criterion=crit,
                             run=run, **rec))
        for h in hist:
            hist_rows.append(dict(exp_id=EXP_ID, N=N, search=LABELS[search],
                                  criterion=crit, run=run, **h))
        for t in tops:
            top_rows.append(dict(exp_id=EXP_ID, N=N, search=LABELS[search],
                                 criterion=crit, run=run, **t))
        cut = scale.burn_in
        prog.step(f"N={N} {LABELS[search]:<20s} {crit:<6s} run={run} "
                  f"rmse={np.mean([r['rmse'] for r in cyc[cut:]]):.4f} "
                  f"({time.time()-t0:.0f}s)")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    if hist_rows:
        ctx.save_table(pd.DataFrame(hist_rows), "search_history.csv")
    if top_rows:
        ctx.save_table(pd.DataFrame(top_rows), "top_structures.csv")
    ctx.save_snapshots(snaps)
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(cells)))
        return

    post = df[df["cycle"] >= scale.burn_in]
    summary = (post.groupby(["N", "criterion", "search", "is_control"])
               .agg(rmse=("rmse", "mean"), rmse_q=("rmse_q", "mean"),
                    rmse_psi=("rmse_psi", "mean"),
                    b_rmse=("b_rmse", "mean"),
                    J=("J", "mean"), K=("K_total", "mean"),
                    radius=("radius_mean", "mean"),
                    radius_std=("radius_std", "mean"),
                    evals=("n_evals", "mean"), seconds=("seconds", "mean"),
                    spread=("spread", "mean"), n=("rmse", "count"))
               .reset_index().sort_values(["N", "criterion", "rmse"]))
    summary["gain_pct"] = 100 * (1 - summary["rmse"] / summary["b_rmse"])
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.to_string(index=False))

    # ---------------- figures ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

    sub = post[(post["criterion"] == "cv") & (post["N"] == scale.ensemble_sizes[0])]
    for s, g in sub.groupby("search"):
        c = g.groupby("cycle")["rmse"].mean()
        axes[0, 0].plot(c.index, c.values, lw=1, label=s)
    axes[0, 0].set_xlabel("assimilation cycle")
    axes[0, 0].set_ylabel("analysis RMSE (normalized)")
    axes[0, 0].set_title("(a) error over the run, by search")
    axes[0, 0].legend(fontsize=6)

    for s, g in sub.groupby("search"):
        c = g.groupby("cycle")["radius_mean"].mean()
        axes[0, 1].plot(c.index, c.values, lw=1, label=s)
    axes[0, 1].set_xlabel("assimilation cycle")
    axes[0, 1].set_ylabel("mean estimated radius")
    axes[0, 1].set_title("(b) the radius the searches settle on")

    if hist_rows:
        hdf = pd.DataFrame(hist_rows)
        one = hdf[(hdf["criterion"] == "cv") &
                  (hdf["N"] == scale.ensemble_sizes[0]) &
                  (hdf["cycle"] == scale.burn_in)]
        for s, g in one.groupby("search"):
            v = g.sort_values("step")["value"].values
            axes[1, 0].plot(np.minimum.accumulate(v), lw=1, label=s)
        axes[1, 0].set_xlabel("objective evaluations")
        axes[1, 0].set_ylabel("best $J$ so far")
        axes[1, 0].set_title(f"(c) inside one cycle (k={scale.burn_in}): "
                             "how each search minimized $J$")
        axes[1, 0].legend(fontsize=6)

    piv = (post.groupby(["search", "criterion"])["rmse"].mean().unstack())
    if not piv.empty:
        o = piv.sort_values("cv") if "cv" in piv else piv
        x = np.arange(len(o))
        w = 0.38
        for i, c in enumerate(o.columns):
            axes[1, 1].bar(x + i * w, o[c].values, w, label=c)
        axes[1, 1].set_xticks(x + w / 2)
        axes[1, 1].set_xticklabels(o.index, rotation=30, ha="right", fontsize=6)
        axes[1, 1].set_ylabel("analysis RMSE")
        axes[1, 1].set_title("(d) admissible criterion against the oracle")
        axes[1, 1].legend(fontsize=7)

    fig.suptitle(f"EXP-02: radius estimated every cycle, QG {bed.g}x{bed.g}, "
                 f"{scale.cycles} cycles")
    fig.tight_layout()
    ctx.save_fig(fig, "cycled.png")

    ctx.save_latex(latex_table(summary.drop(columns=["is_control"]),
                               "Radius estimated inside every assimilation "
                               "cycle.", "tab:cycled"), "cycled.tex")

    adm = summary[(summary["criterion"] == "cv") & (~summary["is_control"])]
    ctx.finish(summary=dict(
        n_cells=len(cells),
        best=(adm.sort_values("rmse").iloc[0].to_dict() if not adm.empty else {})))


if __name__ == "__main__":
    main(parse_cli())
