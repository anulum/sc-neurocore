# Poisson stochastic fidelity evidence

This page records the source, executable parity, independent statistical
reference, Python-to-Verilog co-simulation, benchmark, descriptor, and formal
evidence used to promote `PoissonNeuron` to the polyglot-complete catalogue.

## Source and maintained conventions

Primary source: W. Gerstner, W. M. Kistler, R. Naud, and L. Paninski (2014),
*Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition*,
Sections 7.2 and 7.7,
[doi:10.1017/CBO9781107447615](https://doi.org/10.1017/CBO9781107447615).
Those sections describe the homogeneous Poisson process, exponential waiting
times, and the probability of at least one event in a finite interval.

SC-NeuroCore samples that process into binary bins:

$$
p = 1-\exp\!\left(-\lambda\frac{\Delta t_{\mathrm{ms}}}{1000}\right).
$$

Multiple arrivals in one bin collapse to one event bit. The binary-bin
representation, LFSR polynomial, eight-state decimation, integer threshold, and
default replay seed are maintained engineering conventions.

The portable random contract is a right-shift maximal-period LFSR16 with taps
0, 2, 3, and 5, corresponding to
\(x^{16}+x^{14}+x^{13}+x^{11}+1\). One logical trial advances eight primitive
states before comparison. Zero maps to `0xACE1`; an interior probability maps
to `floor(p*65535)+1`; the event predicate is `sample < threshold`. Eight is
coprime with 65,535, so the decimated stream retains the complete non-zero
period.

## Executable evidence matrix

| Surface | Executed contract | Result |
|---|---|---|
| Python hand model | finite-bin probability, private RNG, deterministic default, explicit entropy, reset/replay, validation, batch atomicity | focused model checks pass |
| Equation/Universal DSL | stateless physical equation plus private seeded state, paired-schema parity, reset and failed-step atomicity | hand/TOML/JSON events and RNG exact |
| Rust engine/PyO3 | complete rate/bin/seed/steps/override ABI | events, count, and final RNG exact |
| Rust safety module | standalone source compiled and executed with `rustc` | 8/8 module tests pass; 4,096-bin stream matches |
| Julia | complete seeded batch contract | events, count, and final RNG exact |
| Go | service and reproducible C-shared ABI | events, count, and final RNG exact |
| Mojo | shared-library C ABI | events, count, and final RNG exact |
| Generated RTL | registered and folded 48-bit Q24.24 seeded datapaths | complete-period vector, count, threshold, probability, and final RNG exact |
| Independent statistical artifact | exhaustive polynomial/comparator re-derivation plus five seeds | hashes, counts, final RNG, rate, and ISI statistics pass |
| Catalogue descriptor | source, backend, reproducibility, validation, and silicon evidence | S5/H1 terminal descriptor under the declared policy |
| SymbiYosys | depth-4 Z3 bounded reset/spike safety property | `PASS` |

The configured native protocol uses `rate_hz=250`, `dt_ms=1`, seed `0x1234`,
4,096 bins, and the configured-rate sentinel. Every runtime emits the same 918
events and finishes at RNG state 45,999. A full-period protocol uses seed
`0xACE1`; every native lane matches all 65,535 Python bits, 14,496 events, and
the returned seed.

## Independent statistical reference

The artifact
`src/sc_neurocore/neurons/reference_trace_data/poisson_lfsr16_statistical_v1.json`
does not import the production RNG helper. Its test independently evaluates the
polynomial, eight-step advance, threshold, event predicate, event digest, and
geometric interval statistics.

At 250 Hz with 1 ms bins, the interval hazard is 0.25 and the continuous
probability is `1-exp(-0.25) = 0.22119921692859512`. Across all 65,535
non-zero states:

- threshold 14,497 and exactly 14,496 events;
- realised probability `0.22119478141451132`;
- first and last event indices 0 and 65,530;
- mean interval `4.520869265263884` bins;
- interval standard deviation `3.9977351042729645` bins;
- interval coefficient of variation `0.8842846076062356`;
- final RNG state `0xACE1`; and
- event-byte SHA-256
  `6f118617f2ecb7a54c5a7ca68ee38a80a68dd15494e361c77aa228397614bfa8`.

The same artifact pins 4,096-bin hashes, counts, and final states for seeds
`1`, `42`, `0xACE1`, `0xBEEF`, and `0xFFFF`. The permitted
continuous-to-discrete probability error is below one LFSR state; mean and CV
tolerances are 0.001 and 0.01.

## Python-to-Verilog co-simulation

The production co-simulation emits two real hardware forms:

1. `UniversalNeuron.to_verilog()` produces the state-owning registered source.
2. `compile_to_datapath()` produces the folded combinational element driven by
   caller-owned RNG state.

Both use 48-bit Q24.24 probability arithmetic. Icarus Verilog executes every
non-zero LFSR state at 250 Hz, 1 ms bins, and seed `0xACE1`.

Python, registered RTL, and folded RTL produce the same complete event vector:

| Observable | Exact result |
|---|---:|
| events | 14,496 |
| final LFSR state | 44,257 (`0xACE1`) |
| comparator threshold | 14,497 |
| Q24.24 probability | `2^24 - round(exp(-0.25)*2^24)` |

The test also checks the observed interval mean and CV against their geometric
targets. Compiler tests separately exercise state ownership, zero/one
probability boundaries, invalid configuration rejection, and deterministic
generation.

## Five-backend benchmark

`benchmarks/bench_model_poisson.py` measures seven warmed 200,000-bin calls
through each public dispatcher. It fails if a backend is missing, an event bit,
count, or final RNG differs, the process is unpinned without acknowledgement,
or the standalone Rust-safety tests fail.

The committed run used one logical CPU with no exclusive-isolation claim:

| Backend | Median call | Median ns/bin | Events | Mismatches | Final RNG |
|---|---:|---:|---:|---:|---:|
| Python | 772.222614 ms | 3861.113070 | 44,256 | 0 | 46,746 |
| Rust | 5.972108 ms | 29.860540 | 44,256 | 0 | 46,746 |
| Julia | 8.818947 ms | 44.094735 | 44,256 | 0 | 46,746 |
| Go | 11.758502 ms | 58.792510 | 44,256 | 0 | 46,746 |
| Mojo | 6.840362 ms | 34.201810 | 44,256 | 0 | 46,746 |

All arrays have SHA-256
`edf44e21373abf717fefaa6de1b527400c1cb1a4cbbab62d6aec86b5b7f642be`.
The artifact records every sample, source hash, runtime version, affinity,
governor, and load average. These timings are local regression evidence, not a
production speed claim. The stable auto policy remains Rust, Mojo, Go, Julia,
then Python.

## Descriptor and formal boundary

`PoissonNeuron.toml` records the book DOI, complete parameter/state/backend
contract, reproducibility digest, class-correct `metric = "statistical"`,
full-period reference, registered/folded co-simulation, and science S5 with
silicon H1.

The catalogue emitter creates `sc_poissonneuron.v`, its port-only harness, and
a depth-4 SymbiYosys job. The bounded property proves reset clears
`spike_out`. It does not prove distribution quality, the full-period
co-simulation theorem, floating-point equivalence, synthesis timing, or
hardware deployment.

## Reproduction

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest \
  tests/test_model_poisson.py \
  tests/test_poisson_backend_loading.py \
  tests/test_poisson_backends.py \
  tests/test_poisson_schema_dsl.py \
  tests/test_reference_poisson.py \
  tests/test_cosim_poisson.py \
  tests/test_bench_poisson.py -p no:cov -q

taskset -c 4 env PYTHONPATH=src:. .venv/bin/python \
  benchmarks/bench_model_poisson.py \
  --json benchmarks/results/local_python_2026-07-14_poisson_lfsr16.json

cd hdl/formal/catalogue
sby -f sc_poissonneuron.sby
```

The evidence establishes the declared binary-bin and seeded hardware contract.
It does not establish cryptographic randomness, within-bin timing, same-bin
multiplicity, external-simulator parity, FPGA timing closure, or hardware
equivalence beyond the stated checks.
