# EscapeRateNeuron

`EscapeRateNeuron` is a stochastic-threshold point neuron with deterministic
passive-membrane dynamics. It evaluates an exponential conditional firing
intensity after an exact constant-current RC step, then draws the event from a
model-scoped, replayable 16-bit LFSR stream.

- Module: `sc_neurocore.neurons.models.escape_rate`
- Family: stochastic integrate-and-fire / escape noise
- Dynamic state: membrane voltage `v` and RNG state `rng_state`
- Source: W. Gerstner (2000), *Population Dynamics of Spiking Neurons: Fast
  Transients, Asynchronous States, and Locking*, Neural Computation 12:43–89,
  [doi:10.1162/089976600300015899](https://doi.org/10.1162/089976600300015899)
- Fidelity evidence: [EscapeRate stochastic fidelity](../../validation/escape_rate_stochastic_fidelity.md)

## Mathematical contract

The maintained passive membrane obeys

$$
\tau_m \frac{dV}{dt}=-(V-V_{rest})+R I.
$$

For current held constant over one macro step, SC-NeuroCore advances it exactly:

$$
V_\infty=V_{rest}+RI,
\qquad
V_{n+1}^{*}=V_\infty+(V_n-V_\infty)e^{-\Delta t/\tau_m}.
$$

The candidate voltage drives the Gerstner escape intensity:

$$
\rho(V_{n+1}^{*})=\rho_0
\exp\!\left(\frac{V_{n+1}^{*}-V_{threshold}}{\Delta u}\right).
$$

Gerstner (2000), Eqs. (2.13)–(2.15), gives the conditional intensity,
survival function, and firing-time density. Assuming the intensity is constant
within the discrete step, the survival function yields the bounded event
probability

$$
p_n=1-\exp[-\rho(V_{n+1}^{*})\Delta t].
$$

On an event, `v_reset` is committed; otherwise the exact RC candidate is
committed. The exact RC discretisation, piecewise-constant finite-step hazard,
RNG, comparator quantisation, and reset ordering are maintained SC-NeuroCore
engineering conventions rather than claims about those source equations.

## Canonical RNG and comparator

Every execution surface uses the same right-shift maximal-length LFSR16:

$$
x^{16}+x^{14}+x^{13}+x^{11}+1.
$$

The implementation contract is:

1. Replace a zero seed with `0xACE1`.
2. Advance taps 0, 2, 3, and 5 eight primitive LFSR states.
3. Commit the resulting non-zero 16-bit word as `rng_state`.
4. Convert the float probability to a 17-bit threshold:

   $$
   T(p)=
   \begin{cases}
   0,&p\le 0,\\
   65536,&p\ge 1,\\
   \lfloor 65535p\rfloor+1,&0<p<1.
   \end{cases}
   $$

5. Emit an event exactly when `rng_state < T(p)`.

Eight is coprime with the 65,535-state primitive period, so the decimated trial
stream still visits every non-zero state. Advancing eight states also avoids the
strong adjacent-word correlation of exposing consecutive right-shift states.
This generator exists for deterministic simulation, cross-runtime parity, and
RTL reproducibility; it is not a cryptographic RNG.

The Python constructor defaults to seed `0xACE1`. Passing an integer selects a
replayable stream. Passing `seed=None` explicitly requests fresh entropy for a
Python instance; portable backend and RTL contracts always receive a concrete
16-bit state.

## Parameters

| Parameter | Default | Unit | Contract |
|---|---:|---|---|
| `v` | `-70.0` | mV | current post-step membrane state |
| `v_rest` | `-70.0` | mV | passive resting potential and public `reset()` voltage |
| `v_reset` | `-70.0` | mV | voltage committed after an escape event |
| `v_threshold` | `-50.0` | mV | reference voltage in the exponential intensity |
| `tau_m` | `10.0` | ms | positive membrane time constant |
| `rho_0` | `0.001` | 1/ms | positive intensity at `v_threshold` |
| `delta_u` | `3.0` | mV | positive soft-threshold voltage scale |
| `resistance` | `1.0` | mV/current-unit | positive current-to-voltage gain |
| `dt` | `1.0` | ms | positive macro-step duration |
| `seed` | `0xACE1` | integer | initial LFSR16 state; explicit `None` is Python entropy |

The exponential argument is clipped to `[-700, 700]` before evaluation. All
mutable voltage fields and the input must be finite. `tau_m`, `rho_0`,
`delta_u`, `resistance`, and `dt` must be finite and positive. Invalid scalar or
batch updates fail before committing voltage or RNG state.

## Step and batch ordering

One successful step performs the following atomic sequence:

1. validate the live parameters and input current;
2. compute the exact constant-current RC candidate;
3. compute the clipped exponential intensity and bounded hazard probability;
4. advance the private LFSR eight primitive states and compare the threshold;
5. commit `v_reset` on an event, otherwise commit the RC candidate.

`simulate(n_steps, current, backend=...)` passes the complete physical and RNG
state to the selected batch lane. On success it atomically commits the returned
voltage and RNG state. An unavailable backend, malformed result, or numerical
failure leaves both live states unchanged. `reset()` restores `v_rest` and the
concrete initial seed, making the next run replay the same stream.

## Backend contract

| Backend | Public path | Maintained result |
|---|---|---|
| Python | hand model / `backend="python"` | canonical recurrence and event stream |
| Rust | PyO3 `py_escape_rate_simulate` | events, voltage, and RNG exact |
| Julia | `EscapeRateAccel.simulate_trace` | events, voltage, and RNG exact |
| Go | generated C-shared ABI | events, voltage, and RNG exact |
| Mojo | generated shared-library ABI | events/RNG exact; voltage within `2e-14` |

The configured 4,096-step parity protocol uses every ABI field, `I=17`, and
seed `0x1234`. Every lane emits 29 events and finishes at RNG state 45,999.
Rust, Julia, and Go are bit-exact to Python; Mojo's maximum voltage difference
is bounded by `2e-14`.

`backend="auto"` currently selects the installed Rust batch first, followed by
Mojo, Go, Julia, and Python. The benchmark's measured ordering does not change
that stable dispatch policy.

## Schema and Python-to-Verilog contract

The paired `escape_rate.toml` and `escape_rate.json` schemas use `exp_euler`,
the same source DOI, and seed `0xACE1`. `EquationNeuron` and `UniversalNeuron`
own private LFSR state, preserve it on failed steps, and reset it with the
membrane state. The paired EscapeRate equations do not reference diffusion
noise and therefore do not consume NumPy's process-global random stream.

The Verilog compiler lowers the probability to an unsigned 17-bit threshold
and supports two state-ownership modes:

- registered mode stores one LFSR16 state inside the generated neuron;
- folded mode accepts and returns caller-owned per-neuron RNG state.

The production Q24.24 co-simulation holds `V=-50` and `rho*dt=0.25` for a full
65,535-state period. Python and RTL emit the same 65,535-bit event vector,
14,496 events, threshold 14,497, final seed `0xACE1`, and constant voltage. The
generated catalogue formal job separately proves its stated depth-4 bounded
safety property; it does not prove floating-point equivalence, timing closure,
or RNG distribution quality.

## Statistical reference

`escape_rate_lfsr16_statistical_v1.json` independently re-evaluates the
polynomial and comparator without importing the production RNG helper. At the
full-period `rho*dt=0.25` operating point it records:

| Observable | Value |
|---|---:|
| continuous probability | `0.22119921692859512` |
| realised probability | `0.22119478141451132` |
| event count | `14,496` |
| mean inter-event interval | `4.520869265263884` steps |
| inter-event CV | `0.8842846076062356` |
| final RNG state | `0xACE1` |

The artifact also pins 4,096-step event hashes, counts, and final states for
seeds `1`, `42`, `0xACE1`, `0xBEEF`, and `0xFFFF`.

## Benchmark evidence

The source-hashed artifact
`benchmarks/results/local_python_2026-07-14_escape_rate_lfsr16.json` records
seven warmed 200,000-step repetitions of the complete non-default contract on
one logical CPU. The CPU was not exclusively isolated, so the figures are local
regression evidence and not production throughput claims.

| Backend | Median ns/step | Events | Maximum parity error |
|---|---:|---:|---:|
| Python | `9147.478725` | 1,523 | `0` |
| Rust | `97.627185` | 1,523 | `0` |
| Julia | `63.027875` | 1,523 | `0` |
| Go | `154.90144` | 1,523 | `0` |
| Mojo | `118.841755` | 1,523 | `1.4210854715202004e-14` |

All five lanes finish at RNG state 46,746. The artifact retains affinity, host
load, runtime versions, every repetition, exact source hashes, and the executed
Rust-safety result.

## Usage

```python
from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron

cell = EscapeRateNeuron(seed=0x1234)
trace, event_count = cell.simulate(4096, current=17.0, backend="rust")
final_state = (cell.v, cell.rng_state)

cell.reset()
replay, replay_count = cell.simulate(4096, current=17.0, backend="python")
assert replay_count == event_count
```

For intentionally non-replayable exploratory Python runs:

```python
cell = EscapeRateNeuron(seed=None)
```

Record `cell.initial_seed` if the resulting run must later be reproduced.

## Evidence surfaces

- Python and RNG: `models/escape_rate.py` and `_stochastic_threshold.py`
- Paired schemas and DSL: `model_schemas/escape_rate.{toml,json}`,
  `equation_builder.py`, and `universal_dsl.py`
- Native dispatcher: `sc_neurocore.accel.escape_rate`
- Engine and PyO3: `engine/src/neurons/trivial/escape_rate.rs`,
  `engine/src/pyo3_neurons.rs`
- Julia, Go, Mojo, and Rust safety: `src/sc_neurocore/accel/`
- Verilog compiler and co-simulation: `verilog_compiler.py` and
  `tests/test_cosim_escape_rate.py`
- Independent statistical reference: `tests/test_reference_escape_rate.py`
- Formal job: `hdl/formal/catalogue/sc_escaperateneuron.sby`
- Benchmark contract: `benchmarks/bench_model_escape_rate.py` and
  `tests/test_bench_escape_rate.py`

The evidence establishes the declared discrete implementation and seeded
distribution contract. It does not establish cryptographic randomness,
biological parameter identifiability, external-simulator equivalence, FPGA
timing closure, or hardware equivalence beyond the stated tests.
