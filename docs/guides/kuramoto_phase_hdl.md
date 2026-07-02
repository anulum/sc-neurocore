<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Research-stage Kuramoto HDL path -->

# Kuramoto Phase HDL

SC-NeuroCore now includes a research-stage Kuramoto HDL emitter for
bounded synthesis experiments.

## Boundary

This path is intentionally narrower than the production software solvers.

Current behaviour:

- noiseless only
- all-to-all scalar coupling only
- fixed-point phase registers
- LUT-based sine approximation
- explicit `step_en` driven phase advance

It does not cover:

- stochastic noise
- SSGF geometry coupling
- PGBO coupling
- field-pressure terms
- parity with the production Rust solver

That makes it a safe additive hardware exploration scaffold, not a
replacement for the stable Kuramoto implementations.

## API

```python
from sc_neurocore.hdl_gen import KuramotoEmitter, VerilogGenerator

emitter = KuramotoEmitter(
    module_name="kuramoto_phase_top",
    n_oscillators=4,
    omegas=[0.8, 1.0, 1.1, 1.3],
    initial_phases=[0.0, 0.3, 0.6, 0.9],
    coupling=0.12,
    dt=5e-3,
)
rtl = emitter.generate()
```

Or through the existing generator surface:

```python
gen = VerilogGenerator(module_name="kuramoto_phase_top")
rtl = gen.emit_kuramoto_phase(
    n_oscillators=4,
    omegas=[0.8, 1.0, 1.1, 1.3],
    initial_phases=[0.0, 0.3, 0.6, 0.9],
    coupling=0.12,
    dt=5e-3,
)
```

## Generated interface

The emitted module currently exposes:

- `clk`
- `rst_n`
- `step_en`
- `update_done`
- `phase_bus`

`phase_bus` is the concatenated fixed-point phase-register state for all
oscillators in the emitted core.

The Python reference helpers `fixed_point_step()` and `fixed_state_to_float()`
only accept canonical fixed-point phase vectors with exactly one integer entry
per oscillator and `0 <= phase < 2π_fixed`. This mirrors the emitted RTL reset
and one-step wrap invariant and rejects booleans, floats, negative phases, and
modulus-or-larger values instead of silently coercing them.

## Numerical model

The emitted HDL performs one bounded fixed-point Kuramoto phase step:

1. compute wrapped pairwise phase differences
2. approximate `sin(Δθ)` with a generated LUT
3. accumulate all-to-all coupling
4. multiply by fixed-point `K / N`
5. update each phase by `dt * (ω + coupling_term)`
6. wrap phase back into `[0, 2π)`

This keeps the structure recognisably Kuramoto-like while staying simple
enough for synthesis experiments.

## Verification level

Current verification is limited to:

- structural assertions in Python tests
- configuration validation in Python
- HDL syntax smoke compile under `iverilog`
- no-warning `iverilog` compilation for larger generated sine LUTs that
  require more than 8 address bits
- behavioural `iverilog`/`vvp` simulation for the no-coupling one-oscillator
  fixed-point phase step and `update_done` pulse
- behavioural `iverilog`/`vvp` simulation for a two-oscillator coupled
  fixed-point step through the generated sine-LUT and coupling path
- deterministic fixed-point reference helpers on the emitter are used by the
  HDL simulations, keeping expected values tied to the emitted arithmetic
- bounded fixed-point error summaries against the float Kuramoto Euler step,
  including a higher-coupling four-oscillator regression case
- a committed deterministic evidence report at
  `benchmarks/results/kuramoto_rtl_fixed_point_error.json`, regenerated with
  `tools/run_kuramoto_rtl_error_report.py`, gates higher-coupling,
  low-coupling, and no-coupling fixed-point error thresholds against both the
  float Euler reference and the maintained Rust `KuramotoSolver` baseline in
  the bounded noiseless all-to-all scalar-coupling regime
- the same report now includes explicit phase-regime metadata plus
  wrap-boundary and near-antiphase cases so circular-error handling is covered
  outside nominal phase layouts

It is not yet:

- a timing-closed FPGA implementation report
- exhaustive fixed-point error characterisation across all oscillator counts,
  coupling strengths, frequencies, and phase layouts
- parity evidence for SSGF geometry, PGBO, field-pressure, or noise extensions

## Recommendation

Use this emitter only for research-stage HDL exploration. Keep production
Kuramoto work on the current software solvers until a stronger parity and
hardware-measurement harness exists.
