# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Experimental report index

# Experimental Report Index

This index tracks the current safe alternative-path reports.

## Current Reports

| Route | Report | Cases | Matched | Candidate failures | Max abs diff | Max rel diff | Median baseline runtime | Median candidate runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `physics.heat.cosine-mode` | `experimental_physics_heat_cosine_mode.json` | 2 | 2 | 0 | 0.0012876970 | 0.0035300391 | 105,036,675 ns | 6,674 ns |
| `physics.oscillator.harmonic-symplectic` | `experimental_physics_oscillator_harmonic_symplectic.json` | 2 | 2 | 0 | 0.0000110030 | 0.0001249999 | 12,238,873 ns | 47,247,295 ns |
| `physics.kuramoto.noiseless-symplectic-lift` | `experimental_physics_kuramoto_noiseless_symplectic_lift.json` | 3 | 3 | 0 | 0.0000502599 | 0.0006753018 | 350,993 ns | 1,283,736 ns |
| `solver.lif.subthreshold-exact` | `experimental_solver_lif_subthreshold_exact.json` | 2 | 2 | 0 | 0.0000000000 | 0.0000000000 | 48,318,187 ns | 13,452 ns |

## Reading Order

1. Candidate failures must stay at `0`.
2. Matched cases must equal total cases.
3. Diff limits must remain inside the chosen promotion gate.
4. Runtime is only meaningful after the first three checks pass.

## Promotion Gate

Use the validator before treating a route as promotable:

```bash
env PYTHONPATH=src ./.venv/bin/python tools/validate_experimental_reports.py \
  --require-mode shadow \
  --max-abs-diff 0.01 \
  --max-rel-diff 0.01
```

The current four reports pass that gate.

## Repeated Runs

To generate a fresh batch of reports with validation attached:

```bash
env PYTHONPATH=src ./.venv/bin/python tools/run_experimental_suite.py \
  --repetitions 3 \
  --mode shadow \
  --max-abs-diff 0.01 \
  --max-rel-diff 0.01
```

This writes a timestamped directory under `benchmarks/results/experimental_runs/`
with per-run reports plus a suite summary.

For a production-evidence run without the demo route, add `--real-only`.

## Comparing Suite Directories

To compare two suite runs on their common routes:

```bash
env PYTHONPATH=src ./.venv/bin/python tools/compare_experimental_suites.py \
  benchmarks/results/experimental_runs/<baseline_dir> \
  benchmarks/results/experimental_runs/<candidate_dir>
```
