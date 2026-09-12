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

## EXP-01-REGIME, where is the radius a real decision?

**Fixed uniform radius, no optimization.** Sweeps the observation density
against the radius.

**Grid:**

| axis | values | count |
|---|---|---|
| lattice stride | 2, 4, 8 (≈24%, 6%, 1.5% density) | 3 |
| uniform radius | 1, 2, 3, 4, 6, 8 | 6 |
| independent runs | 2 seeds | 2 |

**36 cells x 60 cycles = 2160 analyses, about 7 minutes.**

**Measures, one row per cycle:** `rmse_q`, `rmse_psi`, `rel_q`, `rel_psi` for
the analysis; the same four for the background (`b_` prefix); ensemble spread;
number of observations.

**The question it answers:** per density, is the optimal radius **interior** to
the range or pinned at a boundary? An optimum at the edge means the
configuration has not reached the regime where the radius trades one error
against another. It also confirms whether the filter survives `obs_freq = 20`.

**Saves:**

| file | content |
|---|---|
| `metrics.csv` | one row per cycle per cell |
| `summary.csv` | post-burn-in means, with gain over background per field |
| `optimum_by_density.csv` | best radius per density, and whether it is interior |
| `snapshots.npz` | analysis mean, background mean and truth at 5 cycles |
| `figures/regime.png` | error vs radius per density; optimum vs density |

**Afterwards you can compute:** RMSE vs cycle curves; error vs radius, one
curve per density; analysis gain over background; spread vs error (calibration
over the run); the useful radius range, which sets `r_max` for EXP-02.

---

## EXP-02-CALIBRATION, fix the search parameters once

A comparison between metaheuristics is worthless if one was tuned and the
others got literature defaults. Every search gets the same treatment: a grid
over its own parameters, scored on the same objective, on **held-out problems**
whose seeds are disjoint from everything the benchmark reports, and scored by
the **admissible criterion**, never by the truth, which would smuggle the
oracle into a method that claims not to need it.

**Grid sizes are comparable across searches** (6 to 12 combinations each) so
that no method gets more chances than another.

| search | parameters swept |
|---|---|
| Tabu | `tenure` ∈ {4, 8, 16}, `iters` ∈ {200, 500} |
| Annealing | `cooling` ∈ {0.90, 0.95, 0.99}, `level_iters` ∈ {None, 20} |
| Genetic | `pop_size` ∈ {10, 15, 25}, `p_mut` ∈ {0.15, 0.25, 0.40} |
| Ant colony | `n_ants` ∈ {6, 10, 20}, `rho` ∈ {0.05, 0.1, 0.3} |
| Flower pollination | `pop_size` ∈ {10, 15, 25}, `switch_p` ∈ {0.6, 0.8}, `levy_scale` ∈ {0.25, 0.5} |

**42 combinations x 4 problems x 3 repeats = 504 cells, about 3 h.**

**Saves:**

| file | content |
|---|---|
| `metrics.csv` | one row per cell: attained `J`, evaluations, seconds, the parameters |
| `traces.csv` | **every evaluation of every run**: step, value, running minimum |
| `summary.csv` | per parameter set: mean `J`, its s.d., and `evals_to_1pct`, how many evaluations to come within 1% of the final value |
| `frozen_params.json` | the winner per search, read automatically by EXP-02 |
| `figures/calibration.png` | attained objective per parameter set |
| `figures/convergence.png` | **convergence curves per parameter set**, selected one in red with ±1 s.d. band |
| `figures/convergence_all.png` | all searches at their calibrated parameters, mean ±1 s.d. |

**Afterwards you can compute:** convergence curves with mean and standard
deviation across problems and repeats; evaluations to reach a given tolerance;
whether a parameter set that ends at the same objective gets there faster;
sensitivity of each search to each of its parameters.

## EXP-03-CYCLED, the radius estimated inside every cycle

At **each** assimilation cycle:

1. form the forecast ensemble
2. compute ensemble features per field, log background variance, ensemble
   correlation at lags 1 and 2
3. cluster $q$, choosing **$K$ by silhouette** over $2\dots8$
4. withhold 30% of the observation lattice
5. optimize the vector of **$K$ integers** against the remaining 70%
6. use the winner for that cycle's analysis, propagate, next cycle

The decision variable is a vector of integers in $\{1,\dots,8\}$, length $K$.
Nothing is rounded.

### Algorithms

| search | move |
|---|---|
| **Tabu** | best non-tabu neighbour; forbids a (component, value) pair for `tenure` iterations; aspiration |
| **Simulated annealing** | same neighbourhood, Metropolis acceptance, step shrinking with temperature |
| **Genetic** | uniform crossover on whole components, one-component mutation, tournament |
| **Ant colony** | pheromone over (component, value) pairs; each ant chooses a value per component independently |
| **Flower pollination** | global: a Lévy draw gives *how many* components to copy from the best. Local: copy components where two candidates differ |
| *Random sampling* | control, same budget |
| *Exhaustive sweep* | control, where the space is enumerable |

All five share one neighbourhood, change one component to a different integer -
so the comparison measures the acceptance rule, not the move.

### Grid

| axis | values | count |
|---|---|---|
| arm | 5 metaheuristics + random sampling + **uniform-radius baseline** | 7 |
| criterion | `cv` (admissible) / `oracle` (uses the truth) | 2 |
| ensemble size | 20, 40 | 2 |
| independent runs | 2 seeds | 2 |

**56 runs x 60 cycles x 120 evaluations, about 28 h**, or 3.5 h with 8 shards.

**Measures, one row per cycle:**

- everything EXP-01 measures
- `K_q`, `K_psi`, `silhouette_q`, `silhouette_psi`, what the clustering chose
- `J` attained, `n_evals`, `n_cache_hits`, `seconds`
- `theta`, the winning integer vector
- `radius_mean`, `radius_std` over the domain
- `rmse_at_theta_oracle`, the error the oracle would have reached on the
  *same* forecast, so the gap between what the criterion picks and what was
  available is known per cycle rather than only in aggregate

**Saves:**

| file | content | size |
|---|---|---|
| `metrics.csv` | one row per cycle per run | 0.9 MB |
| `search_history.csv` | **every evaluation of every search**: cycle, step, integer vector, `J` | 18 MB |
| `top_structures.csv` | the **10 best distinct vectors** each search visited, per cycle, with their `J` | 2 MB |
| `summary.csv` | post-burn-in means by ensemble size, criterion and search | small |
| `snapshots.npz` → trajectory tier | **every cycle, for all five searches**: analysis and background means, truth, the radius field over the domain, cluster labels | 23 MB |
| `snapshots.npz` → ensemble tier | full forecast and analysis ensembles at 5 cycles | 4 MB |
| `snapshots.npz` → precision tier | the resulting $\mathbf{B}^{-1}$ at cycles 15 and 45, sparse (row, col, value), per search | 17 MB |
| `figures/cycled.png` | error over the run by search; radius over the run; **J inside one cycle per search**; admissible vs oracle |  |

**Total on disk for the whole suite at paper scale: about 80 MB.** Peak RAM per
process is 200–400 MB, so eight shards fit in roughly 3 GB.

### What you can draw from `search_history.csv`

This is the file the experiment exists for. Per cycle and per search it holds
the full trace, so you can plot:

- **how `J` was minimized inside a single assimilation cycle**, one curve per
  metaheuristic, which is the comparison the paper is about
- convergence speed: evaluations to reach within x% of the final value
- the path through the integer space: how each component's value settled
- variability between cycles: is the search doing the same thing at cycle 5 and
  at cycle 35?

### Metrics and figures available afterwards

**Time series over the assimilation cycles**, from `metrics.csv`:

- RMSE and relative error per field, analysis and background, one curve per
  search
- ensemble spread against error, is the filter calibrated over the run?
- the mean estimated radius per cycle, and its spatial spread
- the K silhouette chose per cycle, and whether it is stable
- `rmse_at_theta_oracle` against `rmse`: the gap, per cycle, between what the
  admissible criterion picked and what was available on that same forecast

**The optimization itself**, from `search_history.csv`:

- how $J$ was minimized **inside a single assimilation cycle**, one curve per
  metaheuristic, which is the comparison the paper is about
- evaluations to reach within a tolerance of the final value
- the path through the integer space: how each component settled
- is the search doing the same thing at cycle 5 and at cycle 35?

**The structures found**, from `top_structures.csv`:

- the 10 best vectors per cycle: are the runners-up close to the winner, or
  very different at nearly the same $J$? The second case means a flat landscape
  and is itself a result
- agreement between searches: do they converge on the same structure?

**Fields and maps**, from the snapshot archive:

- analysis, background and truth at any cycle, for both fields
- the **radius field over the domain**, per cycle and per search, five maps
  side by side answer whether the methods find the same structure
- cluster labels over the domain: does the grouping follow the flow?
- the implied covariance structure, from the stored $\mathbf{B}^{-1}$

**Aggregates**, from `summary.csv`:

- gain of the clustered radius over a uniform one (K=1 is nested inside it)
- admissible criterion against the oracle
- cost per search at equal evaluations *and* at equal wall time, which differ
  because a move that changes one cluster is cheaper than one that changes the
  whole vector

---

## Order of execution

```bash
make exp00 SCALE=paper     # 1 min    gate
make exp01 SCALE=paper     # ~7 min   sets the density and the useful radius range
make exp02 SCALE=paper     # ~3 h     calibrates and freezes the search parameters
make exp03 SCALE=paper     # ~20 h, or 2.5 with 8 shards
```

EXP-01 determines two things EXP-02 needs: which lattice stride puts the
optimum in the interior, and what range of radii is actually useful (if it is
1…6 rather than 1…12, the search space shrinks by a factor of 4 per component).

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
