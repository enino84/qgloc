# -*- coding: utf-8 -*-
"""Shared infrastructure: scales, output directories, manifests, sharding.

Two things are cached to disk because they are expensive and depend on nothing
else: the reference state after the long spin-up, and the initial ensemble of
each run. A second execution starts in seconds.
"""
from __future__ import annotations

import json, os, platform, sys, time, warnings
from dataclasses import asdict, dataclass, replace

import numpy as np
warnings.filterwarnings("ignore")

import qgloc
from qgloc.progress import Progress, banner, env_summary, fmt_eta, log
from qgloc.testbed import QGConfig, Testbed, build_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get("RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
CACHE_ROOT = os.path.join(RESULTS_ROOT, "cache")

SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_COUNT = max(1, int(os.environ.get("SHARD_COUNT", "1")))
METHOD_FILTER = [t.strip().lower() for t in os.environ.get("METHODS", "").split(",") if t.strip()]


def method_allowed(label):
    return not METHOD_FILTER or any(t in str(label).lower() for t in METHOD_FILTER)


def shard_suffix():
    return "" if SHARD_COUNT == 1 else f"_shard{SHARD_INDEX}"


def shard_cells(cells):
    if SHARD_COUNT == 1:
        return cells
    return [c for i, c in enumerate(cells) if i % SHARD_COUNT == SHARD_INDEX]


@dataclass
class Scale:
    """Everything an experiment varies, in one place.

    Only what the assignment method needs. The fields of the earlier
    radius-and-clusters approach are gone along with the code that used them.
    """
    name: str
    # model and climatology
    mrefin: int = 5
    spinup: float = 20000.0
    n_snapshots: int = 200
    snapshot_every: float = 250.0
    # filter
    ensemble_size: int = 40
    cycles: int = 60
    burn_in: int = 15
    obs_freq: float = 20.0
    runs: int = 2
    # observations: a lattice of this spacing, shifted each cycle
    stride: int = 4
    obs_frac: float = 0.05          # observation error, fraction of the spread
    # the assignment
    kappa: int = 4                  # candidate observations per component
    r_max: int = 8                  # ceiling on how far a neighbourhood grows
    ridge_alpha: float = 0.3        # trace-scaled; see FINDINGS
    # arms
    # Up to 7 because the assignment itself produces radii up to 7: a baseline
    # that was never allowed to try them would make the comparison unfair in
    # the method's favour.
    baseline_radii: tuple = (1, 2, 3, 4, 5, 6, 7)
    # The ensemble size is an axis: the radius the assignment settles on has to
    # be one the ensemble can support. At N = 12 it lands near radius 3 and
    # diverges, but so does a fixed radius 3 at that size.
    assignment_sizes: tuple = (40, 80, 160)
    snapshot_cycles: tuple = (0, 15, 30, 45, 59)

    @property
    def seeds(self):
        return [5000 + 17 * i for i in range(self.runs)]


SCALES = {
    "smoke": Scale(name="smoke", mrefin=5, spinup=2000.0, n_snapshots=60,
                   snapshot_every=100.0, ensemble_size=24, cycles=10,
                   burn_in=4, runs=1, assignment_sizes=(12, 24),
                   baseline_radii=(1, 2, 3),
                   snapshot_cycles=(0, 9)),
    "quick": Scale(name="quick", mrefin=5, spinup=8000.0, n_snapshots=120,
                   snapshot_every=200.0, cycles=25, burn_in=8,
                   assignment_sizes=(24, 40), snapshot_cycles=(0, 12, 24)),
    "paper": Scale(name="paper"),
}


def get_scale(name=None):
    name = name or os.environ.get("SCALE", "smoke")
    if name not in SCALES:
        raise ValueError(f"unknown scale '{name}'. Available: {sorted(SCALES)}")
    return SCALES[name]


def parse_cli(argv=None):
    import argparse
    global SHARD_INDEX, SHARD_COUNT, METHOD_FILTER
    ap = argparse.ArgumentParser(description="Run one QG localization experiment.")
    ap.add_argument("scale", nargs="?", default=None)
    ap.add_argument("--methods", "-m", default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--shards", type=int, default=None)
    a = ap.parse_args(argv)
    if a.methods is not None:
        METHOD_FILTER = [t.strip().lower() for t in a.methods.split(",") if t.strip()]
    if a.shard is not None:
        SHARD_INDEX = a.shard
    if a.shards is not None:
        SHARD_COUNT = max(1, a.shards)
    return a.scale


def config_for(scale, **over):
    cfg = QGConfig(mrefin=scale.mrefin, spinup=scale.spinup,
                   n_snapshots=scale.n_snapshots,
                   snapshot_every=scale.snapshot_every,
                   ensemble_size=scale.ensemble_size, cycles=scale.cycles,
                   burn_in=scale.burn_in, obs_freq=scale.obs_freq,
                   ridge_alpha=scale.ridge_alpha)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def climatology(cfg):
    """The long run and its snapshots, cached.

    One integration to the stationary regime, then snapshots along it. This is
    the only expensive part of the setup and it depends on nothing else, so it
    is written to disk and reused by every run, every method and every shard.
    Run it once before launching shards, or they will all compute it at the
    same time.
    """
    os.makedirs(CACHE_ROOT, exist_ok=True)
    path = os.path.join(
        CACHE_ROOT,
        f"clim_m{cfg.mrefin}_t{int(cfg.spinup)}_n{cfg.n_snapshots}"
        f"_e{int(cfg.snapshot_every)}.npy")
    if os.path.exists(path):
        return np.load(path)

    model = build_model(cfg)
    n = model.get_number_of_variables()
    t0 = time.time()
    x = model.propagate(np.zeros(n), np.array([0.0, float(cfg.spinup)]))
    log(f"spin-up to t={cfg.spinup:.0f} in {time.time()-t0:.0f}s")
    snaps = [x.astype(np.float32)]
    for k in range(cfg.n_snapshots):
        x = model.propagate(x, np.array([0.0, float(cfg.snapshot_every)]))
        snaps.append(x.astype(np.float32))
    S = np.array(snaps)
    np.save(path, S)
    log(f"{len(S)} snapshots to t={cfg.spinup + cfg.snapshot_every*cfg.n_snapshots:.0f} "
        f"in {time.time()-t0:.0f}s -> {os.path.basename(path)}")
    return S


def make_testbed(scale, **over):
    """Build the testbed, with a snapshot pool large enough for the ensemble.

    Members and truth are drawn without replacement, so the pool has to exceed
    the largest ensemble the caller will ask for. Sweeping N upward without
    growing the pool is a quiet way to end up with runs that share most of
    their members.
    """
    cfg = config_for(scale, **over)
    need = cfg.ensemble_size + 1
    if cfg.n_snapshots + 1 < need:
        cfg.n_snapshots = int(2 * need)
    return Testbed(cfg, climatology(cfg))


def cached_ensemble(bed, seed):
    """Members and truth drawn from the cached climatology.

    Nothing to cache separately: drawing from snapshots already in memory is
    free, and caching it would only risk the draw and the snapshot file
    getting out of step.
    """
    return bed.build_ensemble(seed)


class ExperimentContext:
    def __init__(self, exp_id, description, scale, extra_config=None):
        self.exp_id, self.description, self.scale = exp_id, description, scale
        self.extra_config = dict(extra_config or {})
        self.dir = os.path.join(RESULTS_ROOT, f"{exp_id}_{scale.name}")
        self.fig_dir = os.path.join(self.dir, "figures")
        self.tab_dir = os.path.join(self.dir, "tables")
        for d in (self.dir, self.fig_dir, self.tab_dir):
            os.makedirs(d, exist_ok=True)
        self.t0 = time.time()
        banner(f"{exp_id}   [scale={scale.name}]", [description, f"output: {self.dir}"])
        env_summary()

    def path(self, name):
        return os.path.join(self.dir, name)

    @property
    def is_shard(self):
        return SHARD_COUNT > 1

    def save_table(self, df, name):
        if self.is_shard:
            stem, ext = os.path.splitext(name)
            name = f"{stem}{shard_suffix()}{ext}"
        p = os.path.join(self.dir, name)
        df.to_csv(p, index=False)
        log(f"wrote {name}  ({len(df)} rows)", self.exp_id)
        return p

    def save_json(self, obj, name):
        if self.is_shard:
            stem, ext = os.path.splitext(name)
            name = f"{stem}{shard_suffix()}{ext}"
        p = os.path.join(self.dir, name)
        with open(p, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
        log(f"wrote {name}", self.exp_id)
        return p

    def save_snapshots(self, writer, name="snapshots"):
        if writer is None or len(writer) == 0:
            return None
        stem = f"{name}{shard_suffix()}"
        p = writer.write(self.path(f"{stem}.npz"), self.path(f"{stem}_index.csv"))
        if p:
            log(f"wrote {stem}.npz  ({len(writer)} records, "
                f"{os.path.getsize(p)/1e6:.1f} MB)", self.exp_id)
        return p

    def save_fig(self, fig, name, dpi=130):
        if self.is_shard:
            return None
        p = os.path.join(self.fig_dir, name)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        log(f"wrote figures/{name}", self.exp_id)
        return p

    def save_latex(self, text, name):
        if self.is_shard:
            return None
        p = os.path.join(self.tab_dir, name)
        with open(p, "w") as fh:
            fh.write(text)
        log(f"wrote tables/{name}", self.exp_id)
        return p

    def finish(self, summary=None):
        import pyteda
        man = dict(exp_id=self.exp_id, description=self.description,
                   model="QGModel", scale=asdict(self.scale),
                   extra_config=self.extra_config,
                   shard=dict(index=SHARD_INDEX, count=SHARD_COUNT),
                   elapsed_s=round(time.time() - self.t0, 2),
                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   versions=dict(qgloc=qgloc.__version__,
                                 pyteda=getattr(pyteda, "__version__", "unknown"),
                                 numpy=np.__version__,
                                 python=sys.version.split()[0],
                                 platform=platform.platform()))
        if summary is not None:
            man["summary"] = summary
        with open(self.path(f"manifest{shard_suffix()}.json"), "w") as fh:
            json.dump(man, fh, indent=2, default=str)
        log(f"DONE in {fmt_eta(man['elapsed_s'])}", self.exp_id)


def latex_table(df, caption, label, float_fmt="%.4f"):
    return (f"% generated by the qgloc suite\n\\begin{{table}}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"{df.to_latex(index=False, escape=True, float_format=float_fmt)}"
            f"\\end{{table}}\n")


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.titlesize": 9,
                         "legend.fontsize": 8, "mathtext.fontset": "cm"})
    return plt
