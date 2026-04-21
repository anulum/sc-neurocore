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

This is the right place for new math perspectives, frontier physical models, or
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

Why this route is safe:

- it is mathematically explicit and domain-limited
- it does not claim to replace the general heat solver
- it is ideal for `shadow` validation because the candidate is exact for the
  chosen family while the baseline remains the production/reference path

## CLI Runner

Use the small runner tool to execute built-in routes and write JSON reports:

```bash
python tools/run_experimental_path.py --list-routes
python tools/run_experimental_path.py --route demo.affine-sigmoid
python tools/run_experimental_path.py --route physics.heat.cosine-mode
```

By default, reports are written under `benchmarks/results/` with an
`experimental_<route>.json` name.
