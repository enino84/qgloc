# Adaptive localization by observation assignment

The localization radius of an ensemble Kalman filter is normally tuned once
against a known truth and then held fixed. This repository replaces it.

Every model component grows a neighbourhood until it holds a few candidate
observations, and is then updated by **exactly one** of them, or by none. The
radius is not a parameter: it is the distance to whichever observation the
component ended up using. The choice minimizes a variational cost evaluated at
a scalar local analysis, computed from ensemble anomalies and observations
alone, with no reference trajectory and nothing withheld. The resulting radius
field fixes the predecessor sets of a modified-Cholesky precision, assembled
once per cycle for a single global analysis.

`FINDINGS.md` has the measurements, with the figures inline.
`EXPERIMENTS.md` is the specification of what each experiment runs and records.

## The model

The 1.5-layer quasi-geostrophic system of Sakov and Oke (2008), on the unit
square:

$$
q_t = -\psi_x - \varepsilon\, J(\psi, q) - A\,\Delta^{3}\psi + 2\pi\sin(2\pi y),
\qquad
q = \Delta\psi - F\psi
$$

with $J(\psi, q) = \psi_x q_y - \psi_y q_x$ the Jacobian (the nonlinear
advection) and $\Delta = \partial^2/\partial x^2 + \partial^2/\partial y^2$.
Here $q$ is the potential vorticity and $\psi$ the streamfunction. The last
term is the forcing, a double gyre antisymmetric about $y = 1/2$.

Integration is carried on $q$; $\psi$ is recovered at each step by solving the
Helmholtz problem $\Delta\psi - F\psi = q$ with a multigrid solver.

**Only $q$ is estimated.** Because $\psi$ is recomputed from $q$ at every step,
an analysis that corrects $\psi$ throws that correction away at the next
propagation. Measured directly: halving $\psi$ while leaving $q$ untouched and
propagating for 20 time units gives a bit-identical result, and `pyteda`'s
`propagate` reads `x0[:field_size]` and calls `_calc_psi(q)`. The state
estimated here is therefore $q$ alone, $2401$ components, observations are taken
on $q$, and $\psi$ is rebuilt from the analysed $q$ so that the pair satisfies
the model's own constraint. This halves the number of regressions, halves the
dimension of the solve, takes the radius vector from $2K$ to $K$ components,
and takes one objective evaluation from 0.36 s to **0.18 s**.

Boundary conditions are Dirichlet, $\psi = \Delta\psi = \Delta^2\psi = 0$ on
the four edges of the unit square $(x,y) \in [0,1]^2$.

### Configuration

| symbol | code | value | what it is |
|---|---|---|---|
| $F$ | `f` | $1600$ | inverse Rossby radius squared |
| $\varepsilon$ | `r` | $10^{-5}$ | nonlinearity, the strength of the Jacobian |
| $A$ | `rkh2` | $\mathbf{10^{-11}}$ | biharmonic dissipation |
| | `rkh` | $-5\times10^{-8}$ | Laplacian friction (negative, as in the original) |
| | `rkb` | $3\times10^{-4}$ | bottom friction |
| | `dt` | $0.3125$ | time step, RK4 |
| | `mrefin` | $5$ | grid refinement, $49\times49$ per field |
| | `lx` | $1.0$ | domain is the unit square |
| $n$ | | $4802$ | vector length, $q$ and $\psi$ stacked |
| $n_q$ | | $\mathbf{2401}$ | **state actually estimated: $q$ alone** |

$A = 10^{-11}$ is **ten times the package default**, and that is not a detail.
At $10^{-12}$ the norm of $q$ grows without saturating, a factor of fourteen
between $t = 500$ and $t = 8000$, so there is no stationary regime and nothing
to sample a climatology from. At $10^{-11}$ it settles: $1.107\times10^{4}$ at
$t = 20000$ against $1.222\times10^{4}$ at $t = 60000$. Sakov and Oke report exactly this, raising the
dissipation by ten and reducing the step from 1.5 to 1.25 to obtain a stable
assimilating system. The same reduction was needed again here, and further: at
`dt = 1.25` the forecast blows up when the network is sparse and the radius
short, because the analysis corrects the observed points hard and leaves their
neighbours alone, and the biharmonic term amplifies the gradients that creates.
The failure is in `propagate`, not in the analysis. `dt = 0.3125` survives
almost all of the sweep.

Configurations that still diverge are **recorded and not avoided**. Which
combinations of observation density and radius a scheme cannot survive is a
result: `divergence.csv` holds the map, and Sakov and Oke leave the same cells
blank in their figures. A diverged cell no longer ends the run.

Everything else is the reference value.

The reference implementation is `qg-leapfrog-stability`; `pyteda` reproduces it
to four digits on the same configuration, including a boundary asymmetry that
belongs to Sakov's original code and not to the port.

### Timescales

The leading Lyapunov exponent is $\sigma_1 \approx 0.0318$ per time unit, so the
e-folding time is $\tau_L \approx 31.5$ units. Observations every 20 units are
about $0.6\,\tau_L$ apart: the background error grows by a real factor between cycles, which
makes the radius a consequential choice, while staying below the regime where
the filter loses the state.

### Assimilation setup

| | |
|---|---|
| Climatology | spin-up from rest to `t = 20000`, then 200 snapshots every 250 units |
| Ensemble | 20 or 40 members drawn at random from that pool, truth drawn from it too |
| Cycles | 60, every 20 time units (1200 units per run), 15 discarded as burn-in |
| Inflation | 1.15 |
| Filter | EnKF with modified Cholesky precision |
| Observations | regular lattice on **`q`**, stride 2, 4 or 8, shifted each cycle at constant density |
| Observation error | 5% of each field's climatological spread |
| Withheld for the criterion | 30% of the lattice, resampled each cycle |

## Three decisions that are not conveniences

Each was arrived at by measuring what happened without it.

**The state is normalized per field.** The climatological spread of `q` is of
order 2000 and that of `psi` of order 1. An observation error reasonable for
one is meaningless for the other, and pyteda's LETKF reads a single scalar
variance off the noise object, so a heterogeneous `R` silently applies the
first field's error to both. Working in normalized units makes one isotropic
error correct for both fields and stops the aggregate RMSE from being four
orders of magnitude dominated by `q`.

**Observations sit on a regular lattice.** With a random subset the spacing
fluctuates, so some local domains hold several observations and others none,
and the analysis at a point with an empty domain is just the background.
Measured directly at 2% random density: the analysis left `q` unchanged to four
decimal places at every radius. That looks like a broken filter and is not one.
A lattice fixes the spacing, which makes the relation between observation
density and useful radius a property of the design. The lattice is built from a
fixed count of evenly spaced rows rather than a modulo rule, because the grid
side is not a multiple of the stride and a modulo lattice changes size when it
is shifted.

**The ensemble is climatological, not perturbed.** This is the recipe of Sakov
and Oke (2008): one long run, snapshots along it, members and truth drawn from
that set. Perturbing a state and propagating does not work, and the reason was
measured rather than assumed: a perturbation is mostly energy off the
attractor and the hyperviscosity removes it. Starting from 30% of climatology
the background error falls to 0.068 after 120 time units, a factor of five, and
raising the amplitude does not compensate because the propagation eats it.
Only after about 250 units does the surviving component grow, reaching 0.872 at
t = 1000, exactly what a climatological ensemble has from the start. The
snapshots give directly what the long propagation slowly arrives at, and they
give it calibrated: spread over background error comes out near 0.5 in both
fields, with no amplitude to tune.

**The dissipation is ten times the model default.** With `rkh2 = 1e-12` the
norm of `q` grows without saturating, a factor of fourteen between t = 500 and
t = 8000, so there is no stationary regime and no climatology to sample. At
`rkh2 = 1e-11` it settles: 1.107e4 at t = 20000 against 1.222e4 at t = 60000.
Sakov and Oke report the same, raising the dissipation by ten and reducing the
step from 1.5 to 1.25 to get a stable assimilating system. Chasing this was the
single largest detour in building the suite; it is in the 2008 paper.

**A boundary asymmetry that is in the original.** The integration is carried on
`q`, which vanishes on all four boundaries. `psi` comes from the Helmholtz
solver and inherits what the solver gives it: two edges are exactly zero and
two are not. The reference implementation `qg-leapfrog-stability` gives the
same numbers to four digits, so this is Sakov's code and not the port. It is
documented and tested on `q`, where the condition is actually imposed.

## Layout

```
qgloc/
  testbed.py     the model, the climatological ensemble, the lattice
  assignment.py  candidates, the local cost, the searches, the coupled variant
  precision.py   modified-Cholesky precision, rows cached, ridge trace-scaled
  persist.py     snapshot archive
experiments/
  exp00_setup.py       diagnostics of the testbed, before anything is measured
  exp01_assignment.py  the assignment in the loop against fixed radii
tests/           18 tests, each guarding a claim
figures/         the figures in FINDINGS.md
paper/           assignment.tex
```

## Running

```bash
make build                 # builds the image and runs the tests
make smoke                 # a few minutes end to end
make exp00 SCALE=paper     # the climatology cache and the setup diagnostics
make exp01 SCALE=paper     # the assignment in the loop
```

Run `exp00` before launching shards: it writes the climatology cache, and
without it every shard computes the same spin-up at once.

```bash

# the paper sweep, split across eight containers
for i in 0 1 2 3 4 5 6 7; do
  SHARD_INDEX=$i SHARD_COUNT=8 SCALE=paper \
    docker compose run -d --name qgloc-s$i shard
done
docker logs -f qgloc-s0
```

## Experiments

| id | question |
|---|---|

The paper configuration is `mrefin=5` (49x49 per field, 4802 components),
20 and 40 ensemble members, 60 cycles at 20 time units, three lattice spacings
(2, 4, 8) and six radii, with `enkf-modified-cholesky` as the filter.

The resolution is set by the analysis, not by the integration. One objective
evaluation is 0.36 s here against about 1.4 s at 97x97, and `EXP-02` makes
hundreds of them per assimilation cycle. At 49x49 the flow keeps the
heterogeneity the study needs, and `psi` is visually almost unchanged from
193x193, though `q` does lose its finer filaments.

`EXP-01` reports, per density, whether the optimal radius is **interior**. An
optimum at the edge of the admissible range means the configuration has not
reached the regime where the radius trades one error against another, and no
estimation experiment run there would mean anything.

## The precision estimator

For each component $i$ with predecessor set $P(i, r_i)$, the deviations are
regressed on those of the predecessors with a ridge penalty,

$$
\boldsymbol{\beta}_i
= \big(\mathbf{X}_P^{\top}\mathbf{X}_P + \alpha \mathbf{I}\big)^{-1}
  \mathbf{X}_P^{\top}\, \Delta\mathbf{X}_i,
\qquad
\mathbf{X}_P = \Delta\mathbf{X}_{P(i,r_i)}^{\top},
$$

and the factors are assembled as

$$
\mathbf{L}_{ii} = 1, \quad
\mathbf{L}_{i,P(i,r_i)} = -\boldsymbol{\beta}_i^{\top}, \quad
\mathbf{D}_{ii} = \mathrm{var}\big(\Delta\mathbf{X}_i
                  - \mathbf{X}_P\boldsymbol{\beta}_i\big)^{-1},
\qquad
\hat{\mathbf{B}}^{-1}(\mathbf{r}) = \mathbf{L}^{\top}\mathbf{D}\,\mathbf{L}.
$$

Two properties of this form drive the whole design.

**Row separability.** Row $i$ of $\mathbf{L}$ and entry $\mathbf{D}_{ii}$
depend on $r_i$ and on nothing else, so changing one component of the radius
vector leaves every other row untouched. That is what the row cache exploits.

**Positive definiteness is structural.** $\hat{\mathbf{B}}^{-1}$ is
positive definite for any $\mathbf{r}$, because it has the form
$\mathbf{L}^{\top}\mathbf{D}\mathbf{L}$ with $\mathbf{D}$ diagonal and
positive. Truncating an already-formed precision does not preserve this: masking
entries beyond a given distance produced an indefinite matrix in 44 of 44 cases
tested, with minimum eigenvalues as negative as $-2.8\times10^{4}$.

The analysis solves

$$
\big(\hat{\mathbf{B}}^{-1} + \mathbf{H}^{\top}\mathbf{R}^{-1}\mathbf{H}\big)\,
\delta\mathbf{x} = \mathbf{H}^{\top}\mathbf{R}^{-1}(\mathbf{y} - \mathbf{H}\mathbf{x}^b).
$$

## What is stored

Nothing is aggregated away. `metrics.csv` keeps **every** cycle, so how many to
discard as burn-in is a decision taken afterwards without rerunning anything,
and any curve in the paper can be drawn from it: RMSE and relative error per
field for both the analysis and the background, ensemble spread, the chosen K
and its silhouette, the winning radius vector, and the error the oracle would
have reached on the same forecast.

The search itself is recorded three ways. `search_history.csv` holds every
evaluation of every search with its integer vector and objective, so the
minimization inside a single assimilation cycle can be plotted, one curve per
metaheuristic. `top_structures.csv` holds the ten best distinct vectors each
search visited per cycle, which says whether the runners-up are close to the
winner or scattered at nearly the same objective. And `traces.csv` in the
calibration gives convergence curves with mean and standard deviation across
problems and repeats.

The snapshot archive has three tiers. The trajectory tier keeps the analysis
and background means, the truth, the radius field over the domain and the
cluster labels, every cycle, for all five searches. The ensemble tier keeps
full ensembles at five cycles. The precision tier keeps the resulting sparse
`B^-1` at two cycles per search, so the implied covariance structure can be
drawn without rebuilding it.

See `EXPERIMENTS.md` for the file-by-file inventory with sizes.

## Reproducibility

Every run writes a `manifest.json` with the configuration, seeds, package
versions and wall time. The image pins one BLAS thread and `PYTHONHASHSEED=0`;
without those, runs are reproducible only up to thread scheduling.

## A caution about the model

The norm of `q` grows slowly and does not saturate, and the integration
eventually becomes unstable, near `t = 21000` at `dt = 1`, which is documented
in the reference implementation. Every window used here is far below that, but
the drift is real, so the norm of the truth is recorded per cycle and a result
should never be read without checking it.
