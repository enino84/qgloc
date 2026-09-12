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
    name: str
    mrefin: int = 6
    spinup: float = 20000.0
    # A large pool so that runs with different seeds really are independent:
    # drawing 41 members out of 60 leaves two runs sharing most of their
    # ensemble, and calling those independent would understate every spread in
    # the tables. 200 snapshots at 250 units apart is 50000 units of
    # integration, about ten minutes at mrefin 6, paid once.
    n_snapshots: int = 200
    snapshot_every: float = 250.0
    ensemble_size: int = 40
    # The ensemble size is an axis, not a setting. It is what makes the optimal
    # radius move: fewer members means more sampling noise and a tighter
    # radius, more members supports a longer one. If the sweep does not
    # reproduce that, it is not measuring what it claims to.
    ensemble_sizes: tuple = (20, 40)
    cycles: int = 60
    burn_in: int = 15
    # Sakov-Oke assimilate every fourth step, which at dt=1.25 is 5.0. Here it
    # is 20: with tau_L about 31.5 the background error then grows by a real
    # factor between cycles, so the radius is a consequential choice rather
    # than a detail, while staying below the regime where the filter loses the
    # state. Whether it does stay below is checked in EXP-01.
    obs_freq: float = 20.0
    runs: int = 3
    # observation lattice: density is 1/stride^2
    strides: tuple = (2, 4, 8)
    radii: tuple = (1, 2, 3, 4, 6, 8, 10, 12)
    r_max: int = 12
    K_list: tuple = (1, 2, 3, 4, 6)
    # K is chosen by silhouette inside each cycle rather than fixed, so what
    # the scale sets is the range to search over.
    K_range: tuple = (2, 8)
    budget: int = 120
    stride: int = 4
    cv_fraction: float = 0.30
    warm_start: bool = True
    # Calibration (EXP-03). Held-out problems, disjoint from everything the
    # benchmark reports.
    calib_problems: int = 4
    calib_repeats: int = 3
    # How many of the best distinct vectors each search visited are kept, and
    # at which cycles the resulting precision matrix is stored.
    n_top: int = 10
    precision_cycles: tuple = (15, 45)
    budgets: tuple = (60, 150)
    n_repeats: int = 3
    # Independent 70/30 partitions averaged by the cross-validated criterion.
    # One partition already removes the circularity; more cut the variance,
    # because with a single split the arg-min moves with whichever 30% was
    # drawn.
    cv_fraction: float = 0.3
    methods: tuple = ("letkf", "enkf-modified-cholesky")
    # Spread over the whole run, including one before the burn-in and one
    # at the end. The burn-in only affects the averages in the summary:
    # metrics.csv keeps every cycle, so how many to discard can be decided
    # afterwards without rerunning anything.
    snapshot_cycles: tuple = (0, 15, 30, 45, 59)

    @property
    def seeds(self):
        return [5000 + 17 * i for i in range(self.runs)]


SCALES = {
    "smoke": Scale(name="smoke", mrefin=5, spinup=2000.0, n_snapshots=30,
                   snapshot_every=100.0, ensemble_size=12,
                   ensemble_sizes=(8, 12),
                   cycles=10, burn_in=4, runs=1, strides=(2, 4),
                   radii=(1, 2, 4), K_list=(1, 2), budgets=(20,),
                   n_repeats=1, snapshot_cycles=(0, 7),
                   budget=25, K_range=(2, 4),
                   calib_problems=2, calib_repeats=1,
                   n_top=5, precision_cycles=(5,),
                   methods=("enkf-modified-cholesky",)),
    "quick": Scale(name="quick", mrefin=5, spinup=8000.0, n_snapshots=80,
                   snapshot_every=200.0, cycles=25, burn_in=8,
                   ensemble_sizes=(12, 24), budget=60, K_range=(2, 6),
                   calib_problems=3, calib_repeats=2,
                   runs=2, strides=(2, 4, 6), radii=(1, 2, 4, 6, 8),
                   K_list=(1, 2, 4), budgets=(40, 100), n_repeats=2, snapshot_cycles=(0, 10, 19),
                   methods=("enkf-modified-cholesky",)),
    # 60 cells at about 22 min each with EnKF-MC: roughly 22 hours on one
    # core, under three with eight shards. The radius grid is six values
    # rather than eight and the runs two rather than three, which is where the
    # trimming was done: the sweep needs breadth in density more than
    # resolution in radius, and the cycle-to-cycle spread within a run already
    # gives most of what a third run would.
    # mrefin 5 (49x49 per field, 4802 components), not 6. The cost that
    # decides this is the analysis, not the integration: one objective
    # evaluation is 0.45 s here against about 1.8 s at mrefin 6, and EXP-02
    # makes hundreds of them per assimilation cycle. The flow at 49x49 keeps
    # the heterogeneity the study needs -- an active region and quiet corners,
    # two fields with different correlation scales -- and psi is visually
    # almost unchanged from 193x193, though q does lose its finer filaments.
    "paper": Scale(name="paper", mrefin=5, ensemble_size=40, runs=2,
                   radii=(1, 2, 3, 4, 6, 8),
                   methods=("enkf-modified-cholesky",)),
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
                   burn_in=scale.burn_in, obs_freq=scale.obs_freq)
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
    cfg = config_for(scale, **over)
    return Testbed(cfg, climatology(cfg))


def cached_ensemble(bed, seed):
    """Members and truth drawn from the cached climatology.

    Nothing to cache separately: drawing from snapshots already in memory is
    free, and caching it would only risk the draw and the snapshot file
    getting out of step.
    """
    return bed.build_ensemble(seed)


def load_frozen(search):
    """Calibrated parameters if EXP-03 has run, defaults otherwise.

    Reading them from disk rather than hard-coding means a re-calibration
    propagates to every experiment without touching their code, and that the
    paper can say no constant in it was chosen by hand.
    """
    from qgloc.metaheuristics import defaults_for
    params = defaults_for(search)
    path = os.path.join(RESULTS_ROOT, "frozen_params.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                blob = json.load(fh)
            got = blob.get("params", {}).get(search)
            if got:
                params.update(got)
                log(f"using calibrated parameters for {search}: {got}")
        except Exception as exc:
            log(f"could not read {path}: {exc}", level="WARN")
    return params


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
