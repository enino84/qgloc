# Experiment specification

Model and testbed shared by all three experiments.

| | |
|---|---|
| Model | 1.5-layer quasi-geostrophic (Sakov–Oke), `pyteda` |
| Resolution | `mrefin=5`, $49\times49$ per field, vector length $n = 4802$ |
| Fields | $q$ (potential vorticity), $\psi$ (streamfunction) |
| Estimated | **$q$ only**, $2401$ components: $\psi$ is a diagnostic the model recomputes from $q$ at every step |
| Integrator | RK4, `dt = 0.3125`, Dirichlet boundary |
| Dissipation | `rkh2 = 1e-11` (ten times the model default) |
| Climatology | spin-up from rest to `t = 20000`, then **200 snapshots** every 250 units |
| Ensemble | drawn at random from that pool (members and truth) |
| Assimilation | every **20** time units, **60 cycles** (1200 time units), 15 discarded as burn-in |
| Inflation | 1.15 |
| Filter | `enkf-modified-cholesky` |
| Observations | regular lattice on **`q`**, noise 5% of its spread, lattice shifted each cycle at constant density |
| Cost | **0.18 s** per objective evaluation (measured: cached precision, `q`-only state) |

Everything expensive is cached under `results/cache/` and survives the
container.

---

## EXP-00-SETUP, does the testbed do what it claims?

Measures nothing about the method. It exists because every failure encountered
while building this suite was a setup failure that looked like a method
failure.

**Runs:** builds the climatology, draws one ensemble, enumerates the lattices.
About one minute.

**Checks:**

- climatology is stationary (the norm of `q` no longer drifts)
- ensemble is calibrated: spread / background error near 1, in both fields
- lattice densities are what they claim

**Saves:**

| file | content |
|---|---|
| `diagnostics.csv` | per field: climatological spread, ensemble spread, background error, their ratio |
| `lattice.csv` | per stride: number of observations, realized density, spacing, minimum useful radius |
| `setup.npz` | the full ensemble and the truth |
| `figures/reference_state.png` | reference, truth, ensemble mean, both fields |
| `figures/initial_ensemble.png` | member anomalies, spread, background error |
| `figures/lattices.png` | the five observation patterns |

**Afterwards you can compute:** nothing new, this is the gate. If
`spread_over_error` is far from 1, stop and fix it before running anything else.

---

## EXP-01-ASSIGNMENT, the assignment in the loop

The method of `qgloc/assignment.py` run inside a filter, against the thing it
has to beat.

Each cycle: propagate, grow a neighbourhood around every component until it
holds $\kappa = 4$ candidate observations, take the choice minimizing the local
variational cost, read the radius field off that assignment, build the
modified-Cholesky precision from it, analyse. The baseline arms run the same
cycles at a fixed radius.

**The ensemble size is an axis.** The radius the assignment settles on is
whatever "grow until $\kappa$ candidates" produces, and it has to be one the
ensemble can support. At $N = 12$ the assignment lands near radius 3 and
diverges, but so does a *fixed* radius 3 at that size, so the small-ensemble
result is about the pairing of radius and $N$ rather than about the method.

| axis | values | count |
|---|---|---|
| ensemble size | 40, 80, 160 | 3 |
| arm | assignment, plus uniform radius 1 through 7 | 8 |
| independent runs | 2 seeds | 2 |

The baseline goes up to 7 because the assignment itself produces radii up to 7.
A baseline that was never allowed to try them would make the comparison unfair
in the method's favour.

**48 cells x 60 cycles.** The baseline runs up to radius 7 because the
assignment produces radii up to 7, and a baseline not allowed to try the same
values would make the comparison meaningless.

**Measures, one row per cycle:** analysis and background RMSE in model units
and as a fraction of the climatological spread, the gain over the background,
ensemble spread and spread over error, the mean, max and spread of the radius
field, the attained $J$, how many components abstained, how many observations
ended up used, the non-zero fraction of $B^{-1}$, and seconds.

**Saves:** `metrics.csv` (every cycle), `summary.csv` with
`vs_best_uniform_pct` computed against the best fixed radius *at that ensemble
size*, `figures/assignment.png` (error and calibration per $N$) and
`figures/radius_vs_ensemble.png` (the radius chosen, and whether the method
pays as $N$ grows).

## Order of execution

```bash
make exp00 SCALE=paper     # the climatology cache and the setup diagnostics
make exp01 SCALE=paper     # the assignment in the loop
```

---

## Status

- EXP-00 and EXP-01: run end to end at smoke scale.
- EXP-02 and EXP-03: written, **not yet validated end to end**. Run
  `make exp02 SCALE=smoke` and `make exp03 SCALE=smoke` before anything long.
- A real defect was found and fixed at the end: the cached precision builder
  existed but was never wired into the analysis path, so every objective
  evaluation went through the slow per-row `sklearn` fitting. One evaluation
  was about 10 s; it is now **0.36 s**. Every cost figure above assumes the
  fixed path.
