# Localization radius on the quasi-geostrophic model

The localization radius of an ensemble Kalman filter is normally tuned against
a known truth in a synthetic experiment and then held fixed. This repository
studies where that choice actually matters, and whether a radius that varies in
space can be estimated rather than tuned.

The model is the 1.5-layer quasi-geostrophic system of Sakov and Oke, as
implemented in `pyteda`. It was chosen because Lorenz-96 cannot answer the
question: with a uniform grid, uniform forcing and identical observations
everywhere, no grid point has any reason to want a different radius from its
neighbour, so a negative result there says nothing about the idea. The QG model
has two fields with genuinely different correlation scales — the potential
vorticity `q` carries fine filaments while the streamfunction `psi` solves a
Helmholtz problem on `q` and is smooth — and a flow that is not homogeneous:
an active jet, quiet corners, and eddies shedding in between.

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

**Perturbations are in units of each field's own spread.** An absolute
perturbation is enormous for `psi` and negligible for `q`, and produces an
ensemble whose members are indistinguishable in `q`, a background error
thousands of times below climatology, and nothing for the filter to correct.

## Layout

```
qgloc/
  testbed.py        model, ensemble recipe, lattice, per-field errors
  assimilation.py   the cycle, the radius parameterizations, the clustering
  objective.py      budget-counted objectives, with the search recorded
  persist.py        two-tier snapshot archive
  metaheuristics/   tabu, SA, FPA, firefly, GA + random and exhaustive controls
experiments/        one script per experiment, stable identifiers
scripts/            runner
tests/              25 tests, each guarding a claim
```

## Running

```bash
make build                    # builds the image and runs the tests
make smoke                    # a few minutes end to end

# the paper sweep, split across eight containers
for i in 0 1 2 3 4 5 6 7; do
  SHARD_INDEX=$i SHARD_COUNT=8 SCALE=paper \
    docker compose run -d --name qgloc-s$i shard
done
docker logs -f qgloc-s0
```

The paper sweep is 60 cells at roughly 22 minutes each: about 22 hours on one
core, under three with eight shards. Each shard writes `metrics_shard<i>.csv`
into the same directory; concatenate them afterwards.

Running it on a single container instead is `SCALE=paper docker compose up -d
experiments`.

The long spin-up and the initial ensembles are cached under `results/cache`,
which is a mounted volume, so a second run starts in seconds. Deleting that
directory forces them to be recomputed.

## Experiments

| id | question |
|---|---|
| `EXP-01-REGIME` | Sweeping lattice spacing against radius: where does the radius become a two-sided decision rather than a boundary optimum? |

The paper configuration is `mrefin=6` (97x97 per field, 18818 components), 40
ensemble members, 40 cycles at 10 time units, five lattice spacings from 2 to 8
and six radii from 1 to 12, with `enkf-modified-cholesky` as the filter. The
resolution was taken down from 193x193 because the analysis, not the
integration, is what costs: at that size one LETKF cycle is 78 seconds against
17 here, and a dense selection operator is 2.7 GB.

`EXP-01` reports, per density, whether the optimal radius is **interior**. An
optimum at the edge of the admissible range means the configuration has not
reached the regime where the radius trades one error against another, and no
estimation experiment run there would mean anything.

## The parameterizations

`RadiusSpec` maps a parameter vector to a per-component radius array, which
pyteda accepts directly.

`uniform` is one number. `per_field` is two, and has the clearest physical
justification. `clustered` is `K` per field, where the clusters come from
ensemble-derived features — log background variance and the ensemble
correlation at lags one and two — so the radius can follow the flow while the
number of estimated parameters stays at `2K`. All three are nested: a uniform
radius is reachable by each, which is what makes a comparison between them a
comparison of ideas rather than of code paths.

## What is stored

Per cycle and per field: RMSE and relative error for the analysis and the
background, ensemble spread, and the observation count. Nothing is aggregated
away, so any curve in the paper can be drawn from `metrics.csv`.

The snapshot archive has two tiers. The trajectory tier keeps the analysis and
background means with every per-cycle metric, cheap enough to keep for every
run. The ensemble tier keeps full ensembles at a few cycles, which is the only
thing a covariance figure can be rebuilt from.

The search is recorded too: `RadiusObjective.history()` returns every
evaluation with its argument, and `best_trace()` the running minimum, so the
refinement of the radii can be plotted rather than described.

## Reproducibility

Every run writes a `manifest.json` with the configuration, seeds, package
versions and wall time. The image pins one BLAS thread and `PYTHONHASHSEED=0`;
without those, runs are reproducible only up to thread scheduling.

## A caution about the model

The norm of `q` grows slowly and does not saturate, and the integration
eventually becomes unstable — near `t = 21000` at `dt = 1`, which is documented
in the reference implementation. Every window used here is far below that, but
the drift is real, so the norm of the truth is recorded per cycle and a result
should never be read without checking it.
