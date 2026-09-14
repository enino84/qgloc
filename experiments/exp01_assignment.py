# -*- coding: utf-8 -*-
"""
EXP-01-ASSIGNMENT

The assignment in the loop, against fixed radii.

Everything measured about the assignment so far was taken on a single forecast:
it says what the method does, not whether it helps. This experiment puts it
inside a running filter and compares it, cycle by cycle, against the thing it
has to beat, which is a uniform radius chosen well.

Each cycle: propagate, grow a neighbourhood around every component until it has
``kappa`` candidate observations, take the choice that minimizes the local
variational cost, read the radius field off that assignment, build the
modified-Cholesky precision from it, and analyse. The baselines run the same
cycles with a fixed radius everywhere.

The ensemble size is an axis, not a setting. The radius the assignment settles
on is whatever the rule "grow until kappa observations are inside" produces,
and that radius has to be one the ensemble can support. Measured at N = 12: the
assignment lands on a mean radius of 2.9 and diverges, while fixed radius 1
survives and fixed radius 3 does not. The same assignment at a larger ensemble
is a different proposition, which is why N is swept rather than chosen.

A note on the ridge. Every error measurement taken before the penalty was
scaled by the trace of X'X is void: an absolute penalty is fifteen orders of
magnitude below the diagonal in model units, the regressions interpolate their
predecessors exactly, and the diagonal of the precision comes out a factor of
1e9 too large. The filter then ignores the observations and the analysis makes
the error worse at every radius. With the penalty scaled it converges; that is
the configuration used here.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

from common import (ExperimentContext, get_scale, latex_table, make_testbed,
                    method_allowed, parse_cli, setup_matplotlib, shard_cells)
from qgloc.assignment import (LocalCost, build_candidates, CoupledCost,
                              CoupledObjective, SEARCHES)
from qgloc.precision import PrecisionBuilder
from qgloc.progress import Progress

EXP_ID = "EXP-01-ASSIGNMENT"
DESCRIPTION = ("The observation assignment run inside the filter, against "
               "fixed uniform radii.")


def analyse_q(bed, pb, Q, qm, idx, y, r_field, obs_std, rng):
    """One analysis on the prognostic block, given a radius per component."""
    nq = bed.nq
    N = Q.shape[1]
    B = pb.build(np.maximum(r_field, 1).astype(float), sparse=True)
    p = idx.size
    H = sps.csr_matrix((np.ones(p), (np.arange(p), idx)), shape=(p, nq))
    rinv = 1.0 / obs_std ** 2
    A = (B + rinv * (H.T @ H)).tocsc()
    D = (y[:, None] + obs_std * rng.standard_normal((p, N))) - (H @ Q)
    Z = spsolve(A, sps.csc_matrix(rinv * (H.T @ D)))
    Z = Z.toarray() if sps.issparse(Z) else np.asarray(Z)
    return Q + Z.reshape(nq, N), B


def run_arm(bed, X0, xt0, arm, scale, seed, lam=0.0, search="greedy"):
    """One filter run. ``arm`` is either 'assignment' or 'uniform<r>'."""
    cfg = bed.cfg
    qb, nq = bed.qblock, bed.nq
    act = ~bed.boundary_mask()[qb]
    sd = bed.spread["q"]
    obs_std = scale.obs_frac * sd
    T = np.array([0.0, cfg.obs_freq])
    rng = np.random.default_rng(seed)

    X, xt = X0.copy(), xt0.copy()
    rows = []
    for k in range(cfg.cycles):
        t0 = time.perf_counter()
        Xf = np.stack([bed.model.propagate(X[:, e], T)
                       for e in range(X.shape[1])], axis=1)
        xbm = Xf.mean(axis=1)
        Xf = xbm[:, None] + cfg.inflation * (Xf - xbm[:, None])
        xt = bed.model.propagate(xt, T)

        Q = Xf[qb, :]
        qm = Q.mean(axis=1)
        qt = xt[qb]
        DX = Q - Q.mean(axis=1, keepdims=True)

        idx = bed.checkerboard(scale.stride, offset=k % scale.stride,
                               fields=("q",)) - qb.start
        y = qt[idx] + obs_std * rng.standard_normal(idx.size)

        if arm == "assignment":
            cand = build_candidates(bed.g, idx // bed.g, idx % bed.g,
                                    n_candidates=scale.kappa,
                                    r_max=scale.r_max)
            lc = LocalCost(DX, cand, idx, y, qm, obs_std ** 2)
            if lam > 0:
                cc = CoupledCost(lc, bed.g, lam=lam)
                o = CoupledObjective(cc, budget=scale.budget)
                a, info = SEARCHES[search](o, rng=np.random.default_rng(seed + k))
                r_field = cc.radius_field(a)
                J = info["J"]
                n_abstain = int(np.sum(a == scale.kappa))
            else:
                a = lc.best_greedy()
                r_field = lc.radius_field(a)
                J = lc.score(a)
                n_abstain = int(np.sum(a == scale.kappa))
            n_used = len(set(int(v) for v in lc.assigned_obs(a) if v >= 0))
        else:
            r = int(arm.replace("uniform", ""))
            r_field = np.full(nq, r, dtype=int)
            J, n_abstain, n_used = np.nan, 0, idx.size

        pb = PrecisionBuilder(bed.model, nq, alpha=cfg.ridge_alpha,
                              active_mask=act, var_floor=0.0).bind(DX)
        Qa, B = analyse_q(bed, pb, Q, qm, idx, y, r_field, obs_std, rng)

        eb = float(np.sqrt(np.mean((qm - qt) ** 2)))
        ea = float(np.sqrt(np.mean((Qa.mean(axis=1) - qt) ** 2)))
        rows.append(dict(
            cycle=k, arm=arm, rmse_bg=eb, rmse=ea,
            rel_bg=eb / sd, rel=ea / sd,
            gain_pct=100.0 * (1.0 - ea / eb) if eb > 0 else np.nan,
            spread=float(np.mean(np.std(Qa, axis=1))),
            spread_over_error=float(np.mean(np.std(Qa, axis=1)) / ea) if ea > 0 else np.nan,
            radius_mean=float(np.mean(r_field)), radius_max=int(np.max(r_field)),
            radius_std=float(np.std(r_field)),
            J=float(J) if J == J else np.nan,
            n_abstain=n_abstain, n_obs_used=n_used, p_obs=int(idx.size),
            nnz_pct=100.0 * B.nnz / nq ** 2,
            seconds=time.perf_counter() - t0))
        X = Xf.copy()
        X[qb, :] = Qa
    return rows


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(
        EXP_ID, DESCRIPTION, scale,
        extra_config=dict(kappa=scale.kappa, stride=scale.stride,
                          ridge_alpha=scale.ridge_alpha,
                          obs_frac=scale.obs_frac,
                          arms=["assignment"] + [f"uniform{r}"
                                                 for r in scale.baseline_radii]))
    arms = ["assignment"] + [f"uniform{r}" for r in scale.baseline_radii]
    cells = shard_cells([(N, arm, run) for N in scale.assignment_sizes
                         for arm in arms
                         for run in range(scale.runs)
                         if method_allowed(arm)])

    rows = []
    beds = {}
    prog = Progress(len(cells), label="runs", exp_id=EXP_ID, every=1,
                    heartbeat_s=60.0)
    for N, arm, run in cells:
        if N not in beds:
            beds[N] = make_testbed(scale, ensemble_size=N)
            b = beds[N]
            print(f"  N={N}: nq={b.nq}  grid {b.g}x{b.g}  "
                  f"pool={b.snapshots.shape[0]}  "
                  f"ridge={b.cfg.ridge_alpha} (trace-scaled)")
        bed = beds[N]
        X0, xt0 = bed.build_ensemble(scale.seeds[run])
        t0 = time.time()
        cyc = run_arm(bed, X0, xt0, arm, scale, seed=7000 + 31 * run)
        for rec in cyc:
            rows.append(dict(exp_id=EXP_ID, N=N, run=run, **rec))
        cut = scale.burn_in
        prog.step(f"N={N} {arm:<12s} run={run} "
                  f"rel={np.mean([r['rel'] for r in cyc[cut:]]):.4f} "
                  f"({time.time()-t0:.0f}s)")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(cells)))
        return

    post = df[df["cycle"] >= scale.burn_in]
    summary = (post.groupby(["N", "arm"])
               .agg(rel=("rel", "mean"), rel_final=("rel", "last"),
                    rmse=("rmse", "mean"), gain_pct=("gain_pct", "mean"),
                    spread_over_error=("spread_over_error", "mean"),
                    radius_mean=("radius_mean", "mean"),
                    radius_max=("radius_max", "max"),
                    radius_std=("radius_std", "mean"),
                    nnz_pct=("nnz_pct", "mean"), seconds=("seconds", "mean"),
                    n=("rel", "count"))
               .reset_index().sort_values(["N", "rel"]))
    # Each ensemble size has its own baseline: the best fixed radius at that N,
    # not the best overall. Comparing against a baseline tuned at a different
    # ensemble size would flatter or punish the assignment for the wrong reason.
    base = (summary[summary["arm"].str.startswith("uniform")]
            .groupby("N")["rel"].min().rename("rel_best_uniform"))
    summary = summary.join(base, on="N")
    summary["vs_best_uniform_pct"] = 100.0 * (
        1.0 - summary["rel"] / summary["rel_best_uniform"])
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.to_string(index=False))

    sizes = list(scale.assignment_sizes)
    fig, axes = plt.subplots(2, len(sizes), figsize=(5.0 * len(sizes), 8.0),
                             squeeze=False)
    for col, N in enumerate(sizes):
        sub = df[df["N"] == N]
        for arm, gdf in sub.groupby("arm"):
            style = "-o" if arm == "assignment" else "-"
            lw = 1.8 if arm == "assignment" else 1.0
            c = gdf.groupby("cycle")["rel"].mean()
            axes[0][col].plot(c.index, c.values, style, lw=lw, ms=3, label=arm)
            sp = gdf.groupby("cycle")["spread_over_error"].mean()
            axes[1][col].plot(sp.index, sp.values, style, lw=lw, ms=3, label=arm)
        axes[0][col].set_yscale("log")
        axes[0][col].set_title(f"N = {N}: analysis error", fontsize=11)
        axes[0][col].set_xlabel("assimilation cycle")
        axes[0][col].set_ylabel("RMSE / climatological spread")
        axes[0][col].legend(fontsize=7)
        axes[1][col].axhline(1.0, color="k", lw=0.8, ls=":")
        axes[1][col].set_title(f"N = {N}: spread over error", fontsize=11)
        axes[1][col].set_xlabel("assimilation cycle")
        axes[1][col].set_ylabel("spread / error")

    fig.suptitle(f"EXP-01: assignment against fixed radii, "
                 f"{scale.cycles} cycles")
    fig.tight_layout()
    ctx.save_fig(fig, "assignment.png")
    ctx.save_latex(latex_table(summary, "The assignment in the loop against "
                                        "fixed radii.", "tab:assignment"),
                   "assignment.tex")

    # A second figure: how the radius the assignment settles on moves with the
    # ensemble size, which is the question the sweep exists to answer.
    a = df[df["arm"] == "assignment"]
    if not a.empty:
        fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.2))
        for N, g in a.groupby("N"):
            c = g.groupby("cycle")
            m = c["radius_mean"].mean()
            ax2[0].plot(m.index, m.values, "-o", ms=3, label=f"N = {N}")
            ax2[0].fill_between(m.index, m - c["radius_std"].mean(),
                                m + c["radius_std"].mean(), alpha=0.15)
        for r in scale.baseline_radii:
            ax2[0].axhline(r, color="0.6", lw=0.7, ls=":")
        ax2[0].set_xlabel("assimilation cycle")
        ax2[0].set_ylabel("radius")
        ax2[0].set_title("the radius the assignment chooses", fontsize=11)
        ax2[0].legend(fontsize=8)

        g = (summary[summary["arm"] == "assignment"]
             .sort_values("N"))
        ax2[1].plot(g["N"], g["vs_best_uniform_pct"], "-o")
        ax2[1].axhline(0.0, color="k", lw=0.8, ls=":")
        ax2[1].set_xlabel("ensemble size")
        ax2[1].set_ylabel("gain over the best fixed radius (%)")
        ax2[1].set_title("does it pay, and from what size on?", fontsize=11)
        fig2.tight_layout()
        ctx.save_fig(fig2, "radius_vs_ensemble.png")

    best = summary.iloc[0]
    per_N = {int(r["N"]): float(r["vs_best_uniform_pct"])
             for _, r in summary[summary["arm"] == "assignment"].iterrows()}
    ctx.finish(summary=dict(
        n_cells=len(cells), best_arm=str(best["arm"]),
        best_rel=float(best["rel"]),
        assignment_vs_best_uniform_pct=per_N))


if __name__ == "__main__":
    main(parse_cli())
