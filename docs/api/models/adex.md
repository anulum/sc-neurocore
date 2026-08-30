# AdExNeuron

**Python module:** `sc_neurocore.neurons.models.adex`

**Rust engine:** `sc_neurocore_engine.AdExNeuron`

**Acceleration kernels:** Rust safety, Go, Julia, and Mojo

**Primary model source:** Brette & Gerstner (2005),
[doi:10.1152/jn.00686.2005](https://doi.org/10.1152/jn.00686.2005)

`AdExNeuron` implements the adaptive exponential integrate-and-fire model. It
combines an exponential approach to spike initiation with a second state
variable for subthreshold and spike-triggered adaptation.

## Maintained equations

For membrane state `v`, adaptation state `w`, and injected current `I`, the
maintained normalised recurrence evaluates

\[
\dot v = \frac{-(v-v_{rest}) + \Delta_T
  \exp((v-v_{rh})/\Delta_T)}{\tau}
  + \frac{-w+I}{C_m},
\]

\[
\dot w = \frac{a(v-v_{rest})-w}{\tau_w}.
\]

The baseline path advances both candidates with explicit Euler. The
exponential argument is clipped to `[-20, 20]` as a maintained numerical guard.
If the candidate voltage reaches `v_threshold`, the same step emits an event,
sets `v = v_reset`, and commits `w = w_candidate + b`. Invalid input, state, or
candidate values fail before either state variable is committed.

The DOI establishes the model equations and reset/adaptation structure. The
repository defaults are a configurable maintained operating point; they are not
presented as the fitted parameter set of a particular cell in the paper.

## Python API

```python
from sc_neurocore.neurons.models.adex import AdExNeuron

neuron = AdExNeuron()
event = neuron.step(current=250.0)

trace, events = neuron.simulate(
    n_steps=1_000,
    current=500.0,
    backend="auto",
)

source_fit = AdExNeuron.brette_gerstner_2005()
v_trace, w_trace, event_trace = source_fit.simulate_complete(
    n_steps=1_000,
    current=800.0,
    backend="rust",
)
```

`step()` returns `0` or `1`. `simulate()` returns the post-update voltage trace
and the total event count, and advances the instance to the final `(v, w)`.
`simulate_complete()` returns aligned post-step voltage, adaptation, and binary
event traces. Every batch is failure-atomic: malformed output or a rejected
later step leaves the receiving instance unchanged. Passing zero steps returns
empty traces without changing state.

`brette_gerstner_2005()` selects the regular-spiking fit reported in the paper:
`C=281 pF`, `gL=30 nS`, `EL=Vr=-70.6 mV`, `VT=-50.4 mV`,
`DeltaT=2 mV`, `tau_w=144 ms`, `a=4 nS`, `b=80.5 pA`, and
`Vpeak=20 mV`. The runtime stores `tau=C/gL`; the caller still chooses the
numerical timestep and integrator explicitly.

### Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `v` | `-65.0` | Initial membrane voltage |
| `w` | `0.0` | Initial adaptation state |
| `v_rest` | `-65.0` | Resting voltage |
| `v_reset` | `-68.0` | Post-event reset voltage |
| `v_threshold` | `-50.0` | Candidate-voltage event threshold |
| `v_rh` | `-55.0` | Exponential rheobase voltage |
| `delta_t` | `2.0` | Exponential slope factor |
| `tau` | `20.0` | Membrane time constant |
| `tau_w` | `100.0` | Adaptation time constant |
| `a` | `0.5` | Subthreshold adaptation coupling |
| `b` | `7.0` | Event-triggered adaptation increment |
| `c_m` | `200.0` | Membrane capacitance scale |
| `dt` | `0.1` | Integration step |
| `integrator` | `"baseline_euler"` | `baseline_euler`, `rk4`, or `rosenbrock` |

All state and current values must be finite. `delta_t`, `tau`, `tau_w`, `c_m`,
and `dt` must also be positive.

## Backend contract

| Backend | Public selection | Supported contract |
|---|---|---|
| Python | `backend="python"` | Full parameter surface; baseline Euler, RK4, and Rosenbrock |
| Rust engine | `backend="rust"` | Full numeric state and parameter surface through the checked PyO3 batch; baseline Euler |
| Julia | `backend="julia"` | Full numeric state and parameter surface; baseline Euler |
| Go | `backend="go"` | Full numeric state and parameter surface through a C-shared bridge; baseline Euler |
| Mojo | `backend="mojo"` | Full numeric state and parameter surface through a C ABI; baseline Euler |

Compiled model-specific lanes reject RK4 or Rosenbrock rather than silently
changing the configured integrator. Those integrators remain available in the
Python model and in the separate generic polyglot RK4 dispatcher.

For baseline Euler, `backend="auto"` follows the measured order recorded by the
committed benchmark: Rust, Julia, Go, Mojo, then Python. Alternative
integrators use Python. Explicit backend requests fail closed when their runtime
or shared library is unavailable.

## Executed parity envelope

The acceleration tests execute every compiled lane without skip decorators.
With maintained defaults over 1,000 steps, all five backends preserve these
event-count goldens:

| Current | Events |
|---:|---:|
| `0.0` | `0` |
| `200.0` | `4` |
| `500.0` | `12` |

Every compiled lane transports both state traces, the complete event vector,
and the final `(v,w)` state. The enrolled benchmark requires exact event-vector
parity and bounds both state traces by `5e-12`. A non-default full-parameter
case is exercised across Rust, Julia, Go, and Mojo.

The Python-to-Verilog route is tracked separately. Q16.16 Icarus co-simulation
preserves the exact 500-step event counts `0/2/6/12` at
`I=0/200/500/1000`. The committed Q16.16 design passes Yosys coarse synthesis
with 52,014 cells, and the generated Q8.8 catalogue RTL passes a depth-6 Z3
proof of exact reset state and `event => reset-voltage` safety. This establishes
H2 only; it is not timing, PPA, device, physical-silicon, or universal numerical
equivalence evidence.

The independent receipt
`src/sc_neurocore/neurons/reference_receipts/adex_brette_gerstner_2005.json`
re-evaluates the paper's unnormalised equations for the published fit. Over
1,000 steps at `I=800 pA`, it records three exact events and a complete-state
difference below `2e-12` after the runtime's `tau=C/gL` normalisation. The older
feature receipt remains supplementary evidence rather than the primary oracle.

## Benchmark evidence

`benchmarks/results/bench_adex.json` is generated by
`benchmarks/bench_adex.py`. It records source hashes, runtimes, CPU affinity,
governor, host load, exact event-vector parity, final states, complete packet
digests, and both voltage/adaptation trace errors.

The committed single-logical-CPU run uses 100,000 steps, seven repeats, and
`I=500`. Its median call times are:

| Backend | Median call time | Speed-up vs Python | Events |
|---|---:|---:|---:|
| Rust | `3.065 ms` | `253.45x` | `1065` |
| Julia | `3.454 ms` | `224.88x` | `1065` |
| Go | `5.700 ms` | `136.28x` | `1065` |
| Mojo | `7.120 ms` | `109.10x` | `1065` |
| Python | `776.794 ms` | `1.00x` | `1065` |

The run used a powersave governor on a non-isolated, loaded workstation. These
numbers are local source-regression evidence for this workload, not a hardware
throughput claim or a general ranking across machines.

Reproduce it from a checkout with every optional runtime built:

```bash
PYTHONPATH=src:bridge taskset -c <cpu> python \
  benchmarks/bench_adex.py \
  --json benchmarks/results/bench_adex.json
```

## Evidence surfaces

- `tests/test_model_adex_source_contract.py` and `tests/test_reference_adex.py`
  — source-fit parameters, paired-schema identity, independent equations, and
  complete-state receipt custody.
- `tests/test_model_adex_ad_ex_simulate.py` and
  `tests/test_adex_backends_backend_parity.py` — public dispatcher, complete
  full-parameter transport, and all-runtime parity.
- `tests/test_adex_backends_batch_atomicity.py` and
  `tests/test_adex_backends_c_abi_and_loaders.py` — real native ABI failure
  atomicity, packet rejection, and loader failures.
- `tests/test_adex_engine_binding.py` — direct production PyO3 complete-state
  binding and checked rejection.
- `tests/test_bench_adex.py` — benchmark schema, fail-closed behaviour, source
  hashes, and real five-backend execution.
- `src/sc_neurocore/accel/go/services/adex_test.go` — Go recurrence goldens,
  reset, fail-closed state, and native benchmark.
- `src/sc_neurocore/accel/julia/adex_parity_test.jl` — Julia goldens and
  mutation-free rejection.
- `tests/test_cosim_adex.py` — four-current Q16.16 event parity and the exact
  committed Yosys synthesis receipt.

## Current boundary

The compiled model-specific kernels implement the maintained baseline-Euler
recurrence. They do not claim polyglot RK4/Rosenbrock support. The generated RTL
and coarse-synthesis receipt establish the honest H2 boundary; timing, PPA,
board/HIL, device, physical silicon, and universal formal equivalence remain
unclaimed.
