# -*- coding: utf-8 -*-
"""
EXP-00-SETUP

Look at the testbed before trusting anything computed on it.

Every failure encountered while building this suite was a setup failure rather
than a method failure: an ensemble whose members were indistinguishable, an
observation lattice too sparse for the radius, perturbations in the wrong units
for a field. None of them announced itself; they all looked like a method that
did not work. This experiment exists so that they announce themselves.

It runs before the sweep, costs a minute, and writes pictures and numbers for
the three objects everything else is built on: the reference state after the
spin-up, the initial ensemble, and the observation lattice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (ExperimentContext, cached_ensemble, get_scale, latex_table,
                    make_testbed, parse_cli, setup_matplotlib)
from qgloc import SnapshotWriter

EXP_ID = "EXP-00-SETUP"
DESCRIPTION = ("Diagnostics of the reference state, the initial ensemble and "
               "the observation lattice, before anything is measured on them.")


def field(bed, x, name):
    b = bed.blocks[name]
    return np.asarray(x)[b].reshape(bed.g, bed.g)


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(EXP_ID, DESCRIPTION, scale)

    bed = make_testbed(scale)
    x0 = bed.x0_ref
    print(f"  n={bed.n}  grid {bed.g}x{bed.g}")
    for k in bed.blocks:
        print(f"  {k}: spread={bed.spread[k]:.5g}  "
              f"range=[{x0[bed.blocks[k]].min():.4g}, {x0[bed.blocks[k]].max():.4g}]")

    X, x_true = cached_ensemble(bed, scale.seeds[0])
    Xn = bed.normalize(X)
    xn_true = bed.normalize(x_true)
    xb = X.mean(axis=1)

    # ---- numbers a reader should check before anything else ----
    rows = []
    for k, b in bed.blocks.items():
        spread = float(np.mean(np.std(Xn[b], axis=1)))
        err = float(np.sqrt(np.mean((bed.normalize(xb)[b] - xn_true[b]) ** 2)))
        rows.append(dict(
            field=k, climatological_spread=bed.spread[k],
            ensemble_spread=spread, background_error=err,
            spread_over_error=spread / err if err > 0 else np.nan,
            error_over_climatology=err))
    diag = pd.DataFrame(rows)
    ctx.save_table(diag, "diagnostics.csv")
    print()
    print(diag.to_string(index=False))
    print()
    print("  background error is in normalized units, so 1.0 means the ensemble")
    print("  knows nothing beyond climatology and 0.0 means it knows the truth;")
    print("  spread/error near 1 is a calibrated ensemble, far below 1 is")
    print("  under-dispersed and the filter will need inflation.")

    # ---- lattice ----
    lat = []
    for st in scale.strides:
        idx = bed.checkerboard(st)
        k = max(1, bed.g // st)
        lat.append(dict(stride=st, n_obs=int(idx.size), density=bed.density(st),
                        spacing=bed.g / k, min_useful_radius=bed.g / k / 2))
    lattice = pd.DataFrame(lat)
    ctx.save_table(lattice, "lattice.csv")
    print()
    print(lattice.to_string(index=False))

    snaps = SnapshotWriter()
    snaps.add(tag="reference", Xb=X, x_true=x_true, obs_idx=bed.checkerboard(4))
    ctx.save_snapshots(snaps, name="setup")

    # ---- figure 1: the reference state and the truth ----
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (name, cmap) in enumerate((("q", "twilight_shifted"),
                                        ("psi", "RdBu_r"))):
        ref = field(bed, x0, name)
        tru = field(bed, x_true, name)
        bgm = field(bed, xb, name)
        v = np.percentile(np.abs(ref), 99.0)
        for col, (img, ttl) in enumerate((
                (ref, f"reference after spin-up $t={scale.spinup:.0f}$"),
                (tru, "truth at cycle 0"),
                (bgm, "ensemble mean at cycle 0"))):
            im = axes[row, col].imshow(img, cmap=cmap, vmin=-v, vmax=v,
                                       origin="lower", extent=[0, 1, 0, 1],
                                       interpolation="bicubic")
            sym = "q" if name == "q" else r"\psi"
            axes[row, col].set_title(f"${sym}$: {ttl}", fontsize=9)
            axes[row, col].set_xticks([0, 0.5, 1])
            axes[row, col].set_yticks([0, 0.5, 1])
            fig.colorbar(im, ax=axes[row, col], fraction=0.046)
    fig.suptitle(f"EXP-00: reference state, {bed.g}x{bed.g}, "
                 f"$(x,y)\\in[0,1]^2$", fontsize=12)
    fig.tight_layout()
    ctx.save_fig(fig, "reference_state.png")

    # ---- figure 2: ensemble members and spread ----
    n_show = min(4, X.shape[1])
    fig, axes = plt.subplots(2, n_show + 2, figsize=(3.1 * (n_show + 2), 6.4))
    for row, (name, cmap) in enumerate((("q", "twilight_shifted"),
                                        ("psi", "RdBu_r"))):
        b = bed.blocks[name]
        anom = (Xn[b] - Xn[b].mean(axis=1, keepdims=True))
        v = np.percentile(np.abs(anom), 99)
        for j in range(n_show):
            im = axes[row, j].imshow(anom[:, j].reshape(bed.g, bed.g), cmap=cmap,
                                     vmin=-v, vmax=v, origin="lower",
                                     extent=[0, 1, 0, 1], interpolation="bicubic")
            sym = "q" if name == "q" else r"\psi"
            axes[row, j].set_title(f"${sym}$ member {j} anomaly", fontsize=8)
        sd = anom.std(axis=1).reshape(bed.g, bed.g)
        im = axes[row, n_show].imshow(sd, cmap="viridis", origin="lower",
                                      extent=[0, 1, 0, 1])
        axes[row, n_show].set_title("ensemble spread", fontsize=8)
        fig.colorbar(im, ax=axes[row, n_show], fraction=0.046)
        err = (bed.normalize(xb)[b] - xn_true[b]).reshape(bed.g, bed.g)
        im = axes[row, n_show + 1].imshow(np.abs(err), cmap="magma",
                                          origin="lower", extent=[0, 1, 0, 1])
        axes[row, n_show + 1].set_title("|background error|", fontsize=8)
        fig.colorbar(im, ax=axes[row, n_show + 1], fraction=0.046)
        for a in axes[row]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"EXP-00: initial ensemble, N={scale.ensemble_size}, "
                 "normalized units (spread should look like the error)",
                 fontsize=12)
    fig.tight_layout()
    ctx.save_fig(fig, "initial_ensemble.png")

    # ---- figure 3: the observation lattices ----
    fig, axes = plt.subplots(1, len(scale.strides),
                             figsize=(2.8 * len(scale.strides), 3.0))
    axes = np.atleast_1d(axes)
    for a, st in zip(axes, scale.strides):
        idx = bed.checkerboard(st)
        b = bed.blocks["psi"]
        mask = np.zeros(bed.g * bed.g)
        sel = idx[(idx >= b.start) & (idx < b.stop)] - b.start
        mask[sel] = 1.0
        a.imshow(mask.reshape(bed.g, bed.g), cmap="Greys", origin="lower",
                 extent=[0, 1, 0, 1])
        a.set_title(f"stride {st}\n{100*bed.density(st):.1f}%", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("EXP-00: observation lattices (one field shown)", fontsize=11)
    fig.tight_layout()
    ctx.save_fig(fig, "lattices.png")

    ctx.save_latex(latex_table(lattice, "Observation lattices.", "tab:lattice"),
                   "lattice.tex")
    ctx.finish(summary=dict(
        n=int(bed.n), grid=int(bed.g),
        spread=bed.spread,
        background_error={r["field"]: r["background_error"] for r in rows},
        spread_over_error={r["field"]: r["spread_over_error"] for r in rows}))


if __name__ == "__main__":
    main(parse_cli())
