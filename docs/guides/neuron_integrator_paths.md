# Neuron Integrator Paths

SC-NeuroCore now exposes explicit baseline and higher-order integration paths
for selected neuron models where integrator choice materially affects the
numerics.

## Purpose

The goal is clarity, not silent replacement.

- the historical path remains the default
- the alternative path is explicit and opt-in
- tests compare both paths immediately

This avoids confusion between:

- preserving established behaviour
- evaluating a more accurate integrator on the same model

## Current Models

| Model | Default path | Alternative path | Why it exists |
|---|---|---|---|
| `SCIzhikevichNeuron` | `baseline_half_euler` | `rk4` | quadratic voltage term benefits from a clearer explicit higher-order reference |
| `HodgkinHuxleyNeuron` | `baseline_euler` | `rk4`, `rosenbrock` | four coupled ion-channel ODEs are sensitive to step method; Rosenbrock adds a linearly implicit stiff-system route |
| `AdExNeuron` | `baseline_euler` | `rk4`, `rosenbrock` | exponential spike-initiation term benefits from higher-order and linearly implicit alternatives |

## How To Use

```python
from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.neurons.models.adex import AdExNeuron

izh = SCIzhikevichNeuron(integrator="rk4")
hh = HodgkinHuxleyNeuron(integrator="rk4")
adex = AdExNeuron(integrator="rk4")
hh_stiff = HodgkinHuxleyNeuron(integrator="rosenbrock")
adex_stiff = AdExNeuron(integrator="rosenbrock")
```

Baseline-preserving construction:

```python
izh = SCIzhikevichNeuron()          # baseline_half_euler
hh = HodgkinHuxleyNeuron()          # baseline_euler
adex = AdExNeuron()                 # baseline_euler
```

## Rust RK4 Parity Path

The Rust engine exposes `py_rk4_neuron_simulate(model_name, current_trace, dt=None)`
for explicit RK4 batch simulation of the same first three priority models:

- `izhikevich`
- `hodgkin_huxley`
- `adex`

This is an opt-in FFI parity path. It does not change Python constructor
defaults or the Rust network-runner defaults.

## Julia RK4 Parity Path

The maintained JuliaCall entry point is
`sc_neurocore.accel.julia.neurons.simulate_rk4_neuron(model_name, current_trace, dt=None)`.
It exposes the same output schema as the Rust parity path and uses fixed-step
RK4 arithmetic so tests can compare directly against Python and Rust
trajectories. Install the optional bridge with `sc-neurocore[julia]`.
The older per-model Julia mirror files are not authoritative unless loaded
through a maintained Python wrapper.

## Go RK4 Parity Path

The Go shared-library entry point is
`sc_neurocore.accel.go.rk4_neurons.simulate_rk4_neuron(model_name, current_trace, dt=None)`.
Build the local shared object before using it:

```bash
cd src/sc_neurocore/accel/go/rk4_neurons
go build -buildmode=c-shared -o librk4_neurons.so rk4_neurons.go
```

The generated shared object is platform-specific and is not committed. The Go
source is included in the Python package so wheel builders can precompile it.

## Mojo RK4 Parity Path

The Mojo shared-library entry point is
`sc_neurocore.accel.mojo.rk4_neurons.simulate_rk4_neuron(model_name, current_trace, dt=None)`.
Build the local shared object before using it:

```bash
cd src/sc_neurocore/accel/mojo/rk4_neurons
~/.pixi/bin/mojo build --emit shared-lib -o librk4_neurons.so rk4_neurons.mojo
```

Mojo recurrence kernels run as compiled shared-library code but still preserve
the same fixed-step RK4 recurrence as Python, Rust, Julia, and Go. The time
axis is not vectorised because each RK4 step depends on the previous state.

## Cross-Language Harness

`benchmarks/bench_neuron_integrators.py` runs the shared deterministic
1 000-step parity trace for Python, Rust, Julia, Go, and Mojo backend slots.
Unavailable optional backends are reported with explicit missing-runtime or
missing-shared-library reasons. Available backends are then timed on the same
model/current traces and written to `benchmarks/results/bench_neuron_integrators.json`.

```bash
python benchmarks/bench_neuron_integrators.py
python benchmarks/bench_neuron_integrators.py --parity-only
```

Latest local E2E benchmark evidence:

- `benchmarks/results/bench_neuron_integrators_2026-05-09T1848.json`
- `benchmarks/results/bench_neuron_integrators.json`

The 2026-05-09 run used 1 000-step parity traces, 10 000-step timing traces,
and three timing repeats on CPython 3.12. All five backend slots were available
and within tolerance for `SCIzhikevichNeuron`, `HodgkinHuxleyNeuron`, and
`AdExNeuron`.

## Input Contract

All maintained RK4 parity entry points reject invalid runtime inputs before
integration:

- `dt` must be a positive finite scalar
- `current_trace` must be a non-empty one-dimensional finite `float64` trace
- Python `SCIzhikevichNeuron.step(...)` rejects non-finite `input_current`
- Python `SCIzhikevichNeuron` rejects non-finite `a`, `b`, `c`, `d`, invalid
  `dt`, and invalid `noise_std`

## Design Rules

- default construction must preserve historical behaviour
- alternative paths must be named explicitly in the constructor
- tests must cover default preservation and candidate-path stability
- docs must state what each path means

## What This Does Not Claim

- it does not claim that RK4 is universally the best method for every neuron
- it does not claim that Rosenbrock-Euler replaces model-specific validation
- it does not claim that every model in `neurons/models/` has already been
  migrated

The current state is explicit:

- baseline path kept
- RK4 path added for the first three priority models across Python, Rust,
  Julia, Go, and Mojo parity paths
- Rosenbrock path added for the two priority stiff neuron models
- further integrator work should follow the same pattern
