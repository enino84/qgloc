# -*- coding: utf-8 -*-
"""
EXP-01-REGIME

Where does the localization radius become a real decision?

Before any radius can be estimated, the configuration has to be one in which
the choice matters. Two regimes make it not matter, and both were measured
rather than assumed.

If every grid point is observed, each point has its own observation and never
needs remote information: the analysis error rises monotonically with the
radius, the optimum sits at the smallest admissible value, and the problem is
one-sided. If observations are too sparse for the radius in use, the local
domain of most points contains no observation at all and the analysis is the
background: measured at 2% random density, the analysis left the vorticity
unchanged to four decimal places at every radius, which looks like a broken
filter and is not one.

The regime worth studying is between the two, and this experiment locates it by
sweeping the observation lattice spacing against the radius. The quantity to
watch is not the smallest error but whether the optimal radius moves into the
interior of the admissible range: an optimum at the boundary means the sweep
has not yet found a genuine trade-off.

Everything is stored per cycle and per field, since the two fields have
different scales and an aggregate would only measure the vorticity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (ExperimentContext, cached_ensemble, get_scale, latex_table,
                    make_testbed, parse_cli, setup_matplotlib, shard_cells)
from qgloc import SnapshotWriter, run_cycles, score
from qgloc.progress import Progress

EXP_ID = "EXP-01-REGIME"
DESCRIPTION = ("Observation lattice spacing against localization radius: "
               "locating the regime in which the radius is a real decision.")


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(EXP_ID, DESCRIPTION, scale,
                            extra_config=dict(strides=list(scale.strides),
                                              radii=list(scale.radii),
                                              method=scale.methods[0]))
    bed = make_testbed(scale)
    print(f"  n={bed.n}  grid {bed.g}x{bed.g}  "
          f"spread q={bed.spread['q']:.4g} psi={bed.spread['psi']:.4g}")

    cells = shard_cells([(st, r, s) for st in scale.strides
                         for r in scale.radii
                         for s in range(scale.runs)])
    snaps = SnapshotWriter()
    rows = []
    prog = Progress(len(cells), label="runs", exp_id=EXP_ID, every=2,
                    heartbeat_s=60.0)
    for stride, r, s in cells:
        X0, xt0 = cached_ensemble(bed, scale.seeds[s])
        keep = (s == 0 and r == scale.radii[len(scale.radii) // 2])

        def on_cycle(k, xb, xa, x_true, Xa, idx, rec, _st=stride, _r=r):
            if keep and k in scale.snapshot_cycles:
                snaps.add_trajectory(
                    tag=f"stride{_st}/r{_r}/cycle{k}",
                    xb_mean=xb[None, :], xa_mean=xa[None, :],
                    x_true=x_true[None, :],
                    metrics={a: np.array([b]) for a, b in rec.items()
                             if isinstance(b, (int, float))},
                    stride=_st, radius=_r, cycle=k, n_obs=int(idx.size))

        cyc = run_cycles(bed, X0, xt0, float(r), 7000 + 13 * s,
                         method=scale.methods[0], stride=stride,
                         on_cycle=on_cycle)
        for rec in cyc:
            rows.append(dict(exp_id=EXP_ID, method=scale.methods[0],
                             stride=stride, radius=r, run=s,
                             density=bed.density(stride), **rec))
        prog.step(f"stride={stride} r={r} run={s} "
                  f"q={score(bed, cyc, 'rmse_q'):.4f} "
                  f"psi={score(bed, cyc, 'rmse_psi'):.4f}")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    ctx.save_snapshots(snaps)
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(cells)))
        return

    post = df[df["cycle"] >= scale.burn_in]
    summary = (post.groupby(["stride", "radius"])
               .agg(density=("density", "first"), p_obs=("p_obs", "first"),
                    rmse=("rmse", "mean"), rmse_q=("rmse_q", "mean"),
                    rmse_psi=("rmse_psi", "mean"),
                    rel_q=("rel_q", "mean"), rel_psi=("rel_psi", "mean"),
                    b_rmse_q=("b_rmse_q", "mean"),
                    b_rmse_psi=("b_rmse_psi", "mean"),
                    spread=("spread", "mean"), n=("rmse", "count"))
               .reset_index())
    summary["gain_q"] = 100 * (1 - summary["rmse_q"] / summary["b_rmse_q"])
    summary["gain_psi"] = 100 * (1 - summary["rmse_psi"] / summary["b_rmse_psi"])
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.to_string(index=False))

    # Where is the optimum, and is it interior?
    best = (summary.loc[summary.groupby("stride")["rmse"].idxmin()]
            [["stride", "density", "radius", "rmse", "gain_q", "gain_psi"]])
    best["interior"] = (~best["radius"].isin([min(scale.radii),
                                              max(scale.radii)])).astype(int)
    ctx.save_table(best, "optimum_by_density.csv")
    print()
    print(best.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for st, g in summary.groupby("stride"):
        lab = f"stride {st} ({100*bed.density(st):.0f}%)"
        axes[0].plot(g["radius"], g["rmse_q"], "o-", label=lab)
        axes[1].plot(g["radius"], g["rmse_psi"], "s-", label=lab)
    axes[0].set_xlabel("localization radius")
    axes[0].set_ylabel(r"normalized RMSE in $q$")
    axes[0].set_title("(a) potential vorticity")
    axes[1].set_xlabel("localization radius")
    axes[1].set_ylabel(r"normalized RMSE in $\psi$")
    axes[1].set_title(r"(b) streamfunction")
    axes[1].legend(fontsize=6)

    axes[2].plot(best["density"] * 100, best["radius"], "ko-")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("observation density (%)")
    axes[2].set_ylabel("optimal radius")
    axes[2].set_title("(c) the optimum against density")

    fig.suptitle(f"EXP-01: QG {bed.g}x{bed.g}, N={scale.ensemble_size}, "
                 f"{scale.cycles} cycles")
    fig.tight_layout()
    ctx.save_fig(fig, "regime.png")

    ctx.save_latex(latex_table(best, "Optimal radius against observation "
                                     "density.", "tab:regime"), "regime.tex")
    ctx.finish(summary=dict(n_cells=len(cells),
                            interior_optima=int(best["interior"].sum()),
                            n_densities=int(len(best))))


if __name__ == "__main__":
    main(parse_cli())
