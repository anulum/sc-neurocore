# DPINeuron

`DPINeuron` is the normalised current-domain adaptive integrate-and-fire
circuit described by Indiveri, Stefanini, and Chicca (2010). The maintained
model couples the membrane differential-pair integrator (DPI), the nonlinear
positive-feedback current, and the spike-driven after-hyperpolarisation DPI.

Primary source: G. Indiveri, F. Stefanini, and E. Chicca, *Spike-based learning
with a generalized integrate and fire silicon neuron*, ISCAS 2010,
[doi:10.1109/ISCAS.2010.5536980](https://doi.org/10.1109/ISCAS.2010.5536980).
The circuit context is also reviewed in Indiveri et al. (2011),
[doi:10.3389/fnins.2011.00073](https://doi.org/10.3389/fnins.2011.00073).

## Equations and maintained convention

The source positive-feedback current is

\[
I_{fb} =
I_0^{1/(\kappa+1)} I_{mem}^{\kappa/(\kappa+1)}
\left[1+\exp\!\left(-\alpha(I_{mem}-I_{th})\right)\right]^{-1}.
\]

The two coupled current-domain dynamics are

\[
\tau\frac{dI_{mem}}{dt} = \frac{I_{mem}}{I_\tau}
\left(
\frac{I_{in}}{1+I_{mem}/I_g}
- I_\tau + I_{fb} - I_{ahp}
\right),
\]

\[
\tau_{ahp}\frac{dI_{ahp}}{dt} =
\frac{I_{ahp}}{I_{\tau ahp}}
\left(
\frac{I_{spk}r(t)}{1+I_{ahp}/I_{ga}}
- I_{\tau ahp}
\right).
\]

SC-NeuroCore uses `I_in = i_rest + current`. All public current fields are
normalised positive current quantities; the implementation does not assign
fabricated-device amperes to them.

The equations advance with a simultaneous explicit-Euler macro-step. When no
spike pulse is active, a post-Euler `i_mem >= i_threshold` crossing emits one
event, resets `i_mem`, and loads `refractory_period`. On following pulse steps,
`r(t)=1`, the membrane is held at `i_reset`, adaptation receives `i_spike`, and
the timer decreases by `dt` to exactly zero. Between pulses, `r(t)=0` and the
adaptation current decays.

The nonlinear feedback is evaluated in the log domain and its logistic gate
uses sign-stable exponential branches. State is mutation-atomic: all input,
parameter, derivative, and pre-reset candidate checks succeed before any state
is committed. In particular, a non-finite membrane candidate cannot be hidden
by a threshold reset.

## Maintained defaults

| Field | Default | Meaning |
|---|---:|---|
| `i_mem` | `0.01` | membrane DPI output current |
| `i_ahp` | `0.01` | after-hyperpolarisation DPI current |
| `refractory_time` | `0.0` | remaining spike-pulse duration |
| `i_threshold` | `1.0` | post-Euler event threshold |
| `i_reset` | `0.01` | membrane current held during the pulse |
| `i_rest` | `0.1` | resting current added to injected drive |
| `i_tau` | `1.0` | membrane leakage current |
| `i_g` | `1.0` | membrane input-saturation current |
| `i_tau_ahp` | `0.1` | adaptation leakage current |
| `i_ga` | `1.0` | adaptation input-saturation current |
| `i_spike` | `5.0` | pulse current injected into adaptation |
| `i_0` | `0.01` | positive-feedback reference current |
| `kappa` | `0.7` | subthreshold slope factor |
| `alpha` | `10.0` | feedback-gate transition steepness |
| `tau` | `20.0` | membrane DPI time constant |
| `tau_ahp` | `100.0` | adaptation DPI time constant |
| `refractory_period` | `2.0` | spike-pulse duration |
| `dt` | `0.1` | explicit-Euler macro-step |

`reset()` restores `i_mem`, `i_ahp`, and `refractory_time` to their factory
baselines while preserving configured parameters.

## Python API and acceleration

```python
from sc_neurocore.neurons.models.dpi_neuron import DPINeuron

neuron = DPINeuron()
trace, spikes = neuron.simulate(1_000, current=5.0, backend="auto")
assert spikes == 3
assert trace.shape == (1_000,)
```

`trace[t]` is the post-step membrane current. Successful calls also commit the
final `i_mem`, `i_ahp`, and `refractory_time` to the instance. Accepted backend
selectors are `python`, `rust`, `julia`, `go`, `mojo`, and `auto`.

| Backend | Maintained contract |
|---|---|
| Python | complete 18-field state and parameter contract |
| Rust engine | executable factory-default contract; explicit rejection otherwise |
| Rust safety | separately executable complete 18-field contract |
| Julia | complete 18-field contract |
| Go C ABI | complete 18-field contract with staged output writes |
| Mojo C ABI | complete 18-field contract with staged output writes |

`auto` probes Go, Julia, Mojo, compatible Rust, then Python. Explicit backend
selection never silently falls back. Native C-ABI runs validate the complete
trajectory before writing caller-visible output, and a rejected public run
leaves the Python object unchanged.

## Reproducibility and parity

At factory defaults for 1,000 steps, every executable backend preserves this
event vector:

| Injected current | `-0.1` | `0` | `1` | `2` | `3` | `5` | `10` | `20` | `50` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Events | 0 | 0 | 0 | 0 | 1 | 3 | 6 | 11 | 21 |

The enrolled configured contract exercises every native field for 400 steps at
`current=5.0`; it emits four events and ends at
`i_mem=0.2`, `i_ahp=0.27412306389119817`, and
`refractory_time=0.0`. Compiled event counts are exact and measured floating
states remain within `5e-13` absolute error of Python.

The descriptor reference is the 1,000-step default trace at `current=5.0`.
Its float64 membrane trace SHA-256 is
`39b39196db3b637c0014e67183195d20080dfbcb32ffba3afbd981e4cf672ace`.

## Schema, RTL, and formal evidence

The paired TOML and JSON schemas encode both Eq. (3) states, the Eq. (2)
feedback current, and the refractory pulse. The hand recurrence and schema
runner agree on every state and event to `1e-13` over the enrolled varied-drive
sequence.

Generated RTL uses Q16.16 arithmetic plus the compiler's nonlinear LUTs and is
therefore an explicit fixed-point approximation, not float bit equivalence. At
`current=5.0` for 5,000 steps, float and RTL both emit 13 events. The first RTL
event is within three steps of float; before the first event, the measured
`i_mem` and `i_ahp` errors are at most `0.0032` and `0.0006`. The pulse reset,
adaptation increase, and timer decay are also checked. A generated depth-4 Z3
job checks reset hygiene over its enrolled bounded horizon.

Evidence anchors:

- `tests/test_model_dpi_neuron.py` — source equations, pulse ordering,
  validation, adaptation, and failure atomicity;
- `tests/test_dpi_neuron_backends.py` — executable five-backend parity, native
  ABI state, dispatch, and rejection paths;
- `src/sc_neurocore/accel/rust/safety/dpi_neuron.rs` — independent Rust safety
  recurrence and 12 focused tests;
- `src/sc_neurocore/accel/julia/dpi_neuron_parity_test.jl` — 69 standalone
  Julia assertions;
- `tests/test_cosim_dpi_neuron.py` — hand/schema parity and real Icarus Q16.16
  co-simulation;
- `tests/test_reference_dpi_neuron.py` — independent DOI-bound 5,000-step
  reference features;
- `hdl/formal/catalogue/sc_dpineuron.sby` — generated bounded formal job.

## Benchmark evidence

`benchmarks/bench_model_dpi_neuron.py` measures 100,000 public-dispatch steps
at `current=5.0` for all five backends. It records source hashes, exact event
parity, trace error, all final states, affinity, host load, governor, runtime
versions, and an executable Rust-safety receipt. The committed result is a
local regression record at
`benchmarks/results/local_python_2026-07-13_dpi_neuron_circuit.json`; it is not
a reserved-core or general throughput claim.

| Backend | Median call (ms) | Events | Maximum membrane difference |
|---|---:|---:|---:|
| Julia | 10.357 | 210 | `0` |
| Mojo | 11.039 | 210 | `2.568e-13` |
| Go | 15.893 | 210 | `1.110e-16` |
| Rust engine | 100.328 | 210 | `0` |
| Python | 857.834 | 210 | `0` |

## Scope limits

- This is a normalised equation-level circuit model, not a transistor netlist,
  layout, foundry-corner, device-mismatch, noise, power, or timing simulation.
- The spike pulse is the maintained deterministic macro-step convention used
  across software and generated RTL; sub-step analogue pulse shape is not
  modelled.
- Q16.16 validation proves only the declared operating point, horizon, and
  error envelope. It is not bit-true fabricated-chip equivalence.
- Higher silicon-readiness tiers require separate synthesis, timing, and
  hardware evidence.
