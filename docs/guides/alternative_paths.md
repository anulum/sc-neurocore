# SPDX-License-Identifier: AGPL-3.0-or-later
# Alternative Paths

SC-NeuroCore already contains research-tier and experimental modules, but that
alone is not enough to evaluate a novel mathematical or physical idea safely.
The safe pattern is to keep the stable implementation intact and wire the new
idea in as an explicit, opt-in alternative path.

## What This Is For

Use an alternative path when all of the following are true:

- the baseline implementation is already functional and should remain the default
- the candidate implementation is interesting, but not yet trusted enough to
  replace the baseline
- you want side-by-side benchmarking and output comparison before any promotion

This is the right place for new math perspectives, research physical models, or
algorithmic rewrites that need empirical validation before they touch the
stable path.

## Safety Rules

- default to baseline execution
- require an explicit config switch to enable the candidate path
- keep fail-open fallback available while the candidate matures
- compare outputs against the baseline before trusting performance wins
- document expected gains and expected failure modes up front

## API

The harness lives in `sc_neurocore.experimental`:

```python
from sc_neurocore.experimental import (
    AlternativePathCase,
    AlternativePathConfig,
    AlternativePathMode,
    AlternativePathRoute,
)
```

### Minimal Example

```python
from sc_neurocore.experimental import (
    AlternativePathConfig,
    AlternativePathMode,
    AlternativePathRoute,
)


def baseline_step(x: float) -> float:
    return x * 2.0


def candidate_step(x: float) -> float:
    return x * 2.0


route = AlternativePathRoute(
    name="physics.alt-step",
    baseline=baseline_step,
    candidate=candidate_step,
    summary="Candidate transport operator",
    expected_behavior="Should match the baseline within numerical tolerance",
)

result = route.run(
    AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
    0.25,
)

print(result.returned_path)       # shadow-baseline
print(result.comparison.matched)  # True
print(result.to_report())         # JSON-friendly summary
```

### Batch Evaluation

For repeated comparison and simple benchmarking, use named cases:

```python
from sc_neurocore.experimental import (
    AlternativePathCase,
    AlternativePathConfig,
    AlternativePathMode,
    build_demo_registry,
)

registry = build_demo_registry()
summary = registry.evaluate(
    "demo.affine-sigmoid",
    [
        AlternativePathCase("small", args=([0.0, 1.0, -1.0],)),
        AlternativePathCase("biased", args=([2.0, -2.0],), kwargs={"bias": 0.25}),
    ],
    AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
)

print(summary.matched_cases)
print(summary.to_report())
```

## Execution Modes

- `baseline`: stable implementation only
- `shadow`: run baseline, run candidate in parallel order, compare outputs, but
  return the baseline result
- `candidate`: return the candidate result; if `fail_open=True`, fall back to
  the baseline on candidate failure

## What To Document For Each Route

Every alternative path should document:

- `summary`: what the candidate is trying to improve
- `expected_behavior`: what should stay invariant or improve
- validation target: what benchmark, scientific observable, or acceptance test
  will decide whether it is worth keeping

Good examples:

- lower latency at equal numerical fidelity
- same firing statistics with reduced bitstream length
- improved stability on a known pathological regime
- better agreement with a chosen reference dataset

Bad examples:

- "more advanced"
- "new math"
- "should be better somehow"

## Recommended Promotion Flow

1. add the candidate under an explicit experimental import path
2. wire it behind `AlternativePathConfig`
3. run in `shadow` mode first
4. record comparison and benchmark reports
5. promote only after the candidate is both faster or more useful and still
   acceptable against the baseline

## Built-In Demo Route

The package currently includes one safe demo route:

- `demo.affine-sigmoid`: loop baseline vs vectorised NumPy candidate

This route is intentionally simple. Its purpose is not scientific novelty. Its
purpose is to prove the experimental lane can be wired, run in `shadow` mode,
benchmarked, and reported without touching the stable codepath.

## Current Intention

This harness is intentionally generic. It does not force any specific
mathematical direction. It creates a safe lane where new models can be tested,
benchmarked, and compared without damaging the currently functional codebase.

## First Real Physics Route

The first non-demo route is:

- `physics.heat.cosine-mode`

This route compares:

- baseline: Monte Carlo Feynman-Kac evolution from `sc_neurocore.physics.heat`
- candidate: exact Neumann cosine-mode solution for cases where the initial
  observable is a single cosine eigenmode

The baseline uses reflected Brownian Feynman-Kac paths with per-step variance
`2 * diffusivity * dt` and exact triangle-wave boundary folding on
`[0, length]`. That boundary model is part of the route contract: replacing it
with clipping or a fixed lattice walk changes the PDE being benchmarked.

Why this route is safe:

- it is mathematically explicit and domain-limited
- it does not claim to replace the general heat solver
- it is ideal for `shadow` validation because the candidate is exact for the
  chosen family while the baseline remains the production/reference path

## Second Real Physics Route

The second non-demo route is:

- `physics.oscillator.harmonic-symplectic`

This route compares:

- baseline: existing `RK4Solver` on the harmonic oscillator
- candidate: existing `StormerVerlet` symplectic solver on the same system

Why this route is safe:

- it uses production solver infrastructure already present in the codebase
- it stays on a strict Hamiltonian reference problem where the symplectic claim is
  actually relevant
- it does not force a symplectic method into dissipative neuron paths where the
  audit overreached

What to expect:

- on bounded horizons, the symplectic candidate should remain close to the RK4
  baseline
- on longer runs, the symplectic candidate should keep relative energy drift
  small, which is the point of the route
- this is a validation lane for Hamiltonian-like problems, not a blanket
  replacement policy for all ODEs in SC-NeuroCore

The maintained symplectic solvers now fail closed on non-finite time/state,
non-positive timesteps, odd-length or non-1-D Hamiltonian state vectors, and RHS
outputs that do not match the state shape. The route additionally rejects the
zero-energy oscillator state because relative energy drift is undefined there.

## Third Real Route

The third route is solver-focused rather than a broad physics claim:

- `solver.lif.subthreshold-exact`

This route compares:

- baseline: existing `RK4Solver` integration of the constant-current LIF ODE
- candidate: existing `ExactLIFSolver` analytical solution

Why this route is safe:

- the regime is tightly bounded to constant subthreshold current
- the candidate is analytical, so the acceptance criterion is hard rather than
  subjective
- it does not try to replace general neuron simulation, event logic, or
  time-varying input paths

What to expect:

- the candidate should match the RK4 baseline within tight tolerance
- the candidate should remain below threshold for the supplied cases
- `predicted_spike_time` should stay `null` for the built-in cases

## First Memory Route

The first bounded memory route is:

- `memory.delayed-recall.shared-state`

This route compares:

- baseline: a local trace-only delayed-recall loop built on real
  `StochasticLIFNeuron` units
- candidate: the same local spiking baseline plus a small non-local shared
  latent state that writes from and feeds back into the neurons during the
  delay and recall phases

Why this route is safe:

- it is explicitly phenomenological rather than a claim about validated brain
  quantum memory
- it stays inside `sc_neurocore.experimental` and does not touch the stable
  neuron or network path
- it tests a narrow engineering question:
  can a non-local shared state improve delayed cue recall over a purely local
  baseline?

What to expect:

- on short delays, the shared-state candidate should be no worse than the local
  baseline
- on longer delays, the candidate should show a material recall gain rather
  than merely reproducing the baseline
- the route is *quantum-inspired* only at the architectural level; it does not
  claim to model ATP, Posner molecules, CISS, or validated in-vivo
  non-locality

Built-in cases:

- `short_delay`
- `long_delay`

## Fourth Real Route

The fourth route keeps the secondary-audit symplectic idea in a bounded,
explicitly non-production lane:

- `physics.kuramoto.noiseless-symplectic-lift`

This route compares:

- baseline: the current noiseless first-order Kuramoto Euler update
- candidate: a symplectic XY-model lift integrated with `StormerVerlet`

Why this route is safe:

- it does **not** replace the production Kuramoto solver
- it is limited to the bounded noiseless phase-only regime
- it documents the mathematical lift explicitly instead of pretending the
  current noisy or forced production path is Hamiltonian

What to expect:

- on short horizons, the candidate should stay close to the Euler baseline in
  phase and order parameter
- this route is a research probe for the Hamiltonian limit, not a blanket
  promotion target for `engine/src/scpn/kuramoto.rs` or `L8_PhaseFieldLayer`
- if the lifted candidate ever becomes useful, it still needs separate
  justification before touching the stable path

## CLI Runner

Use the small runner tool to execute built-in routes and write JSON reports:

```bash
python tools/run_experimental_path.py --list-routes
python tools/run_experimental_path.py --route demo.affine-sigmoid
python tools/run_experimental_path.py --route physics.heat.cosine-mode
python tools/run_experimental_path.py --route physics.oscillator.harmonic-symplectic
python tools/run_experimental_path.py --route physics.kuramoto.noiseless-symplectic-lift
python tools/run_experimental_path.py --route solver.lif.subthreshold-exact
```

By default, reports are written under `benchmarks/results/` with an
`experimental_<route>.json` name.

## How To Read The JSON Report

Each run writes one JSON object with route-level and case-level fields.

Top-level fields:

- `route_name`: which experimental path was evaluated
- `mode`: `baseline`, `shadow`, or `candidate`
- `total_cases`: how many built-in or supplied cases ran
- `matched_cases`: how many cases stayed within the configured tolerances
- `candidate_failures`: how many cases raised and needed fallback or failed
- `median_baseline_runtime_ns`: median baseline runtime across cases
- `median_candidate_runtime_ns`: median candidate runtime across cases

Case-level fields:

- `returned_path`: whether the route returned baseline, shadow-baseline,
  candidate, or fallback-baseline
- `candidate_error`: populated only if the candidate raised
- `comparison.matched`: whether that case passed the tolerance check
- `comparison.max_abs_diff`: worst absolute difference seen in comparable leaves
- `comparison.max_rel_diff`: worst relative difference seen in comparable leaves

Practical reading order:

1. check `candidate_failures`
2. check `matched_cases == total_cases`
3. inspect the largest `max_abs_diff` and `max_rel_diff`
4. only then compare `median_*_runtime_ns`

Do not treat a faster candidate as promotable if it is failing cases or only
passing because the tolerances are loose.

## Comparing Reports Over Time

For one route, compare JSON reports by holding the route, case set, and
tolerances fixed.

Good comparison:

- same route
- same cases
- same `absolute_tolerance` and `relative_tolerance`
- same hardware or a clearly documented hardware change

Bad comparison:

- different case sets
- different tolerances
- one run in `shadow` and one in `candidate`
- one run on the local CPU and another on a different machine without noting it

Promotion should require both:

- consistent case matching
- a reason to prefer the candidate, such as lower runtime or better invariant
  preservation on the route's stated target

## Promotion Validation

Use the validator to turn report review into an explicit gate:

```bash
env PYTHONPATH=src ./.venv/bin/python tools/validate_experimental_reports.py \
  --require-mode shadow \
  --max-abs-diff 0.01 \
  --max-rel-diff 0.01
```

This fails if any report has:

- candidate failures
- unmatched cases
- wrong execution mode for the requested gate
- absolute or relative drift above the supplied caps

For a quick human summary of the current route set, see
`benchmarks/results/experimental_INDEX.md`.

## Repeated Suite Runs

To accumulate evidence instead of relying on one-off runs, use the suite runner:

```bash
env PYTHONPATH=src ./.venv/bin/python tools/run_experimental_suite.py \
  --repetitions 3 \
  --mode shadow \
  --max-abs-diff 0.01 \
  --max-rel-diff 0.01
```

This writes a timestamped directory under `benchmarks/results/experimental_runs/`
containing:

- one JSON report per route per repetition
- `suite_summary.json`
- `suite_summary.md`

The suite exits non-zero if any repetition fails the configured gate.

For a production-evidence batch that excludes the demo route:

```bash
env PYTHONPATH=src ./.venv/bin/python tools/run_experimental_suite.py \
  --repetitions 3 \
  --mode shadow \
  --max-abs-diff 0.01 \
  --max-rel-diff 0.01 \
  --real-only
```

## Comparing Two Suite Runs

To compare two suite directories on their common routes:

```bash
env PYTHONPATH=src ./.venv/bin/python tools/compare_experimental_suites.py \
  benchmarks/results/experimental_runs/<baseline_dir> \
  benchmarks/results/experimental_runs/<candidate_dir>
```

This reports:

- common routes with delta in max absolute and relative drift
- routes present only in one suite

It does not pretend non-overlapping route sets are directly comparable.
