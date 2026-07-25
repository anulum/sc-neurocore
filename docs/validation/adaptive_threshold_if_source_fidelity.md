# Adaptive-Threshold IF — Source Fidelity

This page records exactly which parts of the primary literature the
maintained `AdaptiveThresholdIFNeuron` implements, which parts it does not,
and the executed evidence behind every claim.

## Primary-source boundary

Two primary sources define the model's identity:

- **Mihalas & Niebur (2009)**, *A generalized linear integrate-and-fire
  neural model produces diverse spiking behaviors*, Neural Computation
  21(3), 704–718, [DOI 10.1162/neco.2008.12-07-680](https://doi.org/10.1162/neco.2008.12-07-680).
  The source threshold equation is
  `dΘ/dt = a(V - E_L) - b(Θ - Θ∞)`. Setting the voltage coupling to zero
  (`a = 0`) yields the maintained threshold decay
  `dθ/dt = -(θ - θ_rest)/τ_θ`.
- **Platkiewicz & Brette (2010)**, *A threshold equation for action
  potential initiation*, PLoS Computational Biology 6(7), e1000850,
  [DOI 10.1371/journal.pcbi.1000850](https://doi.org/10.1371/journal.pcbi.1000850).
  The source derives that "the spike threshold increases by a fixed amount
  after each spike" — the maintained post-spike shift
  `θ ← θ + Δθ`.

The model is therefore an **explicitly composite reduced adaptive-threshold
LIF**, not an exact instance of either source:

- The Platkiewicz–Brette central mechanism — a voltage-dependent threshold
  equilibrium `θ∞(V)` through sodium-channel inactivation — is **outside**
  this model.
- The Mihalas–Niebur voltage-coupling term `a(V - E_L)` and its adaptation
  currents are **outside** this model.
- All defaults (`theta_rest=-50 mV`, `delta_theta=5 mV`, `tau_m=10 ms`,
  `tau_theta=50 ms`, `dt=0.1 ms`) are catalogue/model-family choices, not
  source-derived parameters.

An earlier revision of the model docstring named "Platkiewicz & Bhatt
2010" (a typographical error) and the descriptor recorded
`integration.method = "euler"` while production always used the exact
constant-input relaxation. Both are corrected findings, not harmless prose;
the identity above replaces them.

## Exact relaxation and the event convention

Both state equations are linear with piecewise-constant input, so the
maintained step is the exact closed-form flow, never an Euler step:

```
v'     = (v_rest + I) + (v - (v_rest + I)) * exp(-dt / tau_m)
theta' = theta_rest   + (theta - theta_rest) * exp(-dt / tau_theta)
spike  = (v' >= theta')
on spike: v <- v_reset; theta <- theta' + delta_theta
```

The event is detected on the post-update candidates at maintained step
boundaries; no within-step root localisation is claimed.

## Independent reference evidence

The committed reference trace
`reference_trace_data/adaptive_threshold_if_tonic_adaptation_doi.json`
is reproduced by an independent re-derivation (no production `step` code):
tonic drive `I = 20.0`, 160 steps of `dt = 0.1`, one spike at step 138 with
the threshold jumping from `-50` toward `-45` and decaying back. Every
feature (spike count, first spike step, and final/min/max/mean of both
states) matches at `1e-12` absolute.

## Numerical and atomic contract

- Invalid input, invalid configuration, and non-finite candidates are
  rejected before any state mutation in every maintained lane.
- `reset()` restores only the documented state (`v = v_rest`,
  `theta = theta_rest`) and preserves the complete configuration.
- The batch contract returns complete `v`/`theta`/`spikes` trajectories
  plus final-state and spike-count receipts; the dispatcher validator
  re-checks the reset and the fixed threshold shift on every spike.

## Executable parity matrix

| Lane | Evidence | Result |
|---|---|---|
| Python golden | `tests/test_model_adaptive_threshold_if.py` | exact reference |
| Rust engine (PyO3) | `tests/test_adaptive_threshold_if_parity.py` | complete traces within `1e-12` |
| Standalone Rust safety | `tests/test_adaptive_threshold_if_rust_parity.py` | complete traces within `2e-15` |
| Julia (juliacall) | `tests/test_adaptive_threshold_if_julia_parity.py` | within `1e-12`; typed buffer failures atomic |
| Go (C ABI) | `tests/test_adaptive_threshold_if_go_parity.py`, `tests/test_adaptive_threshold_if_native_abi.py` | within `1e-12`; all overlap/null/status classes rejected without writes |
| Mojo (C ABI) | `tests/test_adaptive_threshold_if_mojo_parity.py`, `tests/test_adaptive_threshold_if_native_abi.py` | within `1e-10`; identical ABI classes |
| Dispatch contracts | `tests/test_adaptive_threshold_if_{input_validation,backend_selection,result_validation,c_facade}.py` | input bounds, selection/reload, result validation, C-facade status boundaries |

## Paired schema and Q32.32 co-simulation

The TOML and JSON schema models are structurally identical and preserve the
hand model's exact relaxation within `5e-12` over a varied 256-step drive,
in both the subthreshold and the spiking regimes (6/6 events).

Generated Q32.32 SystemVerilog is validated at the enrolled grid-exact
operating point (`tau_m = tau_theta = 0.8`, `dt = 0.1`, so both exponential
arguments land exactly on the 0.125-step lookup grid):

- measured maximum state error over the 256-step sign-changing drive:
  `v: 1.22e-8`, `theta: 2.65e-9` (declared envelope `0.01` mV);
- the complete 256-entry event vector is identical to the Python golden;
- every RTL spike resets `v` to `v_reset` and shifts `theta` by exactly
  `delta_theta`;
- a depth-4 Z3 bounded job proves reset safety only
  (`hdl/formal/catalogue/sc_adaptive_threshold_if.sby`, PASS);
- Yosys `synth` completes the generated module.

The enrolled default configuration (`tau_m = 10`, `tau_theta = 50`) is
**not** claimed for the generated RTL: its exponential arguments do not lie
on the lookup grid, and a grid-quantised envelope is not presented as
fidelity. No formal equivalence, synthesis timing, device, or PPA claim is
made; the silicon tier is H1.

## Controlled benchmark

`benchmarks/results/bench_adaptive_threshold_if.json` records the
five-runtime 200,000-step batch (`22.0 + 6.0*sin(i*0.037) + 1.5*cos(i*0.011)`
drive, non-default initial state) measured with five repeats on one pinned
logical CPU, with host load, tool versions, source and binary SHA-256
hashes, and run order recorded. All five lanes return matching events,
final states, and trace digests within the declared tolerances. This is
local regression evidence only; no production speed claim is made.

## Boundaries

- No voltage-dependent threshold equilibrium, voltage coupling, adaptation
  current, refractory period, or synaptic conductance.
- Defaults are model-family choices, not source parameters.
- The benchmark is non-exclusive single-CPU evidence, not an isolated
  production measurement.

## Reproduction

```bash
rustc --edition 2021 --test \
  src/sc_neurocore/accel/rust/safety/adaptive_threshold_if.rs -o /tmp/atif && /tmp/atif
python -m pytest \
  tests/test_model_adaptive_threshold_if.py \
  tests/test_adaptive_threshold_if_dynamics.py \
  tests/test_adaptive_threshold_if_backends.py \
  tests/test_adaptive_threshold_if_parity.py \
  tests/test_adaptive_threshold_if_rust_parity.py \
  tests/test_adaptive_threshold_if_julia_parity.py \
  tests/test_adaptive_threshold_if_go_parity.py \
  tests/test_adaptive_threshold_if_mojo_parity.py \
  tests/test_adaptive_threshold_if_native_abi.py \
  tests/test_adaptive_threshold_if_input_validation.py \
  tests/test_adaptive_threshold_if_backend_selection.py \
  tests/test_adaptive_threshold_if_result_validation.py \
  tests/test_adaptive_threshold_if_c_facade.py \
  tests/test_cosim_adaptive_threshold_if.py \
  tests/test_reference_adaptive_threshold_if.py \
  tests/test_bench_adaptive_threshold_if.py -q
cd hdl/formal/catalogue && sby -f sc_adaptive_threshold_if.sby
taskset -c 0 env PYTHONPATH=$WHEEL_SITE:src:. python \
  benchmarks/bench_model_adaptive_threshold_if.py
```
