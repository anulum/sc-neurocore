# PoissonNeuron

`PoissonNeuron` is a homogeneous Poisson event source sampled into binary
time bins. It carries an explicit replayable LFSR16 state so Python, native
backends, and generated RTL can emit the same event stream.

- Module: `sc_neurocore.neurons.models.poisson`
- Family: statistical point-process input generator
- Dynamic state: `rng_state`
- Source: W. Gerstner, W. M. Kistler, R. Naud, and L. Paninski (2014),
  *Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition*,
  Sections 7.2 and 7.7,
  [doi:10.1017/CBO9781107447615](https://doi.org/10.1017/CBO9781107447615)
- Fidelity evidence:
  [Poisson stochastic fidelity](../../validation/poisson_stochastic_fidelity.md)

## Mathematical contract

For a homogeneous intensity \(\lambda\) in hertz and a binary time bin
\(\Delta t\) in milliseconds, the probability of at least one arrival is

$$
p = 1-\exp\!\left(-\lambda\frac{\Delta t}{1000}\right).
$$

The emitted value is one when at least one event occurs in the bin and zero
otherwise. Multiple arrivals in the same bin collapse to one bit. The
maintained discrete process is therefore Bernoulli with geometric inter-event
intervals:

$$
\operatorname{E}[N]=np,\quad
\operatorname{Var}(N)=np(1-p),\quad
\operatorname{E}[\mathrm{ISI}]=\frac{1}{p},\quad
\operatorname{CV}(\mathrm{ISI})=\sqrt{1-p}.
$$

This representation does not retain within-bin event time or multiplicity.
The implementation evaluates `-expm1(-hazard)` for small-hazard precision and
rejects non-finite rate, bin width, override, hazard, or probability before RNG
state advances.

## Canonical RNG and comparator

Every execution surface uses the same right-shift maximal-period LFSR16:

$$
x^{16}+x^{14}+x^{13}+x^{11}+1.
$$

One accepted trial performs this exact sequence:

1. Replace a zero seed with `0xACE1`.
2. Advance taps 0, 2, 3, and 5 eight primitive LFSR states.
3. Commit the resulting non-zero word as `rng_state`.
4. Convert the probability to the integer threshold

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
stream still visits every non-zero state. This generator supports reproducible
simulation and hardware parity; it is not a cryptographic RNG.

The constructor defaults to seed `0xACE1`. Passing an integer selects a
replayable stream. Passing `seed=None` explicitly requests fresh entropy for
that Python instance; native and RTL calls always receive a concrete state.

## Parameters

| Parameter | Default | Unit | Contract |
|---|---:|---|---|
| `rate_hz` | `100.0` | Hz | finite non-negative homogeneous intensity |
| `dt_ms` | `1.0` | ms | finite positive binary-bin width |
| `seed` | `0xACE1` | integer | initial LFSR16 state; explicit `None` requests Python entropy |

`step(rate_override=-1.0)` uses `rate_hz` when the override is negative. A
non-negative override is interpreted in hertz and must be finite.

## Step, batch, and reset ordering

One successful `step()` validates the live contract, evaluates the finite-bin
probability, advances the private LFSR, compares the threshold, and returns the
binary event.

`simulate(n_steps, rate_override, backend=...)` passes the complete rate/bin/RNG
contract to one batch lane. On success it atomically commits the returned RNG
state. An unavailable backend, malformed output, non-binary event, invalid
final state, or numerical failure leaves the instance unchanged.

`reset()` restores the concrete initial seed. It therefore replays the same
stream for an explicit seed and for the one concrete seed captured when
`seed=None` constructed the object.

## Backend contract

| Backend | Public path | Maintained result |
|---|---|---|
| Python | hand model / `backend="python"` | canonical events and RNG state |
| Rust | PyO3 `py_poisson_simulate` | events, count, and RNG exact |
| Julia | `PoissonAccel.simulate_trace` | events, count, and RNG exact |
| Go | generated C-shared ABI | events, count, and RNG exact |
| Mojo | shared-library C ABI | events, count, and RNG exact |

The configured 4,096-bin protocol uses `rate_hz=250`, `dt_ms=1`, seed
`0x1234`, and the configured-rate sentinel. Every lane emits 918 events and
finishes at RNG state 45,999 with an identical event array.

`backend="auto"` selects the integrated Rust engine first, followed by Mojo,
Go, Julia, and Python. The committed non-exclusive benchmark records warm
timing order separately; diagnostic timing noise does not rewrite this stable
production policy.

## Schema and Python-to-Verilog contract

The paired `poisson.toml` and `poisson.json` schemas carry the source DOI,
finite-bin law, event encoding, seed, polynomial, decimation count, and
comparator convention. `EquationNeuron` and `UniversalNeuron` support the
stateless physical equation while preserving private RNG state and failure
atomicity.

The Verilog compiler lowers the probability to a 17-bit threshold and exposes:

- registered mode, which stores one LFSR16 state inside the event source; and
- folded mode, which accepts caller-owned RNG state for population sharing.

The Q24.24 co-simulation runs both forms for all 65,535 non-zero states at
250 Hz with 1 ms bins. Python, registered RTL, and folded RTL emit the same
65,535-bit vector, 14,496 events, threshold 14,497, probability word, and final
seed `0xACE1`.

The generated formal job proves its stated depth-4 reset/spike safety property.
It does not prove floating-point equivalence, timing closure, or distribution
quality.

## Independent statistical reference

`poisson_lfsr16_statistical_v1.json` independently evaluates the polynomial,
eight-step advance, threshold, and event predicate without importing the
production RNG helper. At 250 Hz, 1 ms bins, and seed `0xACE1` it records:

| Observable | Value |
|---|---:|
| continuous probability | `0.22119921692859512` |
| realised probability | `0.22119478141451132` |
| comparator threshold | `14,497` |
| event count | `14,496` |
| first / last event index | `0` / `65,530` |
| mean inter-event interval | `4.520869265263884` steps |
| inter-event standard deviation | `3.9977351042729645` steps |
| inter-event CV | `0.8842846076062356` |
| final RNG state | `0xACE1` |
| event-byte SHA-256 | `6f118617f2ecb7a54c5a7ca68ee38a80a68dd15494e361c77aa228397614bfa8` |

The artifact also pins 4,096-bin event hashes, counts, and final states for
seeds `1`, `42`, `0xACE1`, `0xBEEF`, and `0xFFFF`.

## Benchmark evidence

The source-hashed artifact
`benchmarks/results/local_python_2026-07-14_poisson_lfsr16.json` records seven
warmed 200,000-bin repetitions on one logical CPU. The CPU was not exclusively
isolated, so these values are local regression evidence and not production
throughput claims.

| Backend | Median ns/bin | Events | Mismatches | Final RNG |
|---|---:|---:|---:|---:|
| Python | `3861.113070` | 44,256 | 0 | 46,746 |
| Rust | `29.860540` | 44,256 | 0 | 46,746 |
| Julia | `44.094735` | 44,256 | 0 | 46,746 |
| Go | `58.792510` | 44,256 | 0 | 46,746 |
| Mojo | `34.201810` | 44,256 | 0 | 46,746 |

All arrays have SHA-256
`edf44e21373abf717fefaa6de1b527400c1cb1a4cbbab62d6aec86b5b7f642be`.
The artifact retains affinity, host load, runtime versions, every repetition,
source hashes, and the executed eight-test Rust-safety result.

## Usage

```python
from sc_neurocore.neurons.models.poisson import PoissonNeuron

source = PoissonNeuron(rate_hz=250.0, dt_ms=1.0, seed=0x1234)
events, event_count = source.simulate(4096, backend="rust")
final_rng = source.rng_state

source.reset()
replay, replay_count = source.simulate(4096, backend="python")
assert replay_count == event_count
assert replay.tolist() == events.tolist()
```

For intentionally non-replayable exploratory Python runs:

```python
source = PoissonNeuron(seed=None)
recorded_seed = source.initial_seed
```

Record `initial_seed` if the run must later be reproduced.

## Evidence surfaces

- Python and RNG: `models/poisson.py` and `_stochastic_threshold.py`
- Native dispatcher: `sc_neurocore.accel.poisson`
- Engine and PyO3: `engine/src/neurons/special.rs` and
  `engine/src/pyo3_neurons.rs`
- Julia, Go, Mojo, and Rust safety: `src/sc_neurocore/accel/`
- Paired schemas and DSL: `model_schemas/poisson.{toml,json}`,
  `equation_builder.py`, and `universal_dsl.py`
- Verilog compiler and co-simulation: `verilog_compiler.py` and
  `tests/test_cosim_poisson.py`
- Independent statistical reference: `tests/test_reference_poisson.py`
- Formal job: `hdl/formal/catalogue/sc_poissonneuron.sby`
- Benchmark contract: `benchmarks/bench_model_poisson.py` and
  `tests/test_bench_poisson.py`

The evidence establishes the declared binary-bin process and seeded
implementation. It does not establish cryptographic randomness, external
simulator equivalence, within-bin timing, event multiplicity, FPGA timing
closure, or hardware equivalence beyond the stated tests.
