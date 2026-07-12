# Quantum Annealing Bridge

**Public module:** `sc_neurocore.bridges.quantum_annealing`
**Tier:** Research bridge
**Compatibility surface:** 24 historical exports
**Optional runtimes:** SC-NeuroCore Rust engine, `dimod`, and D-Wave Ocean

The bridge compiles stochastic-computing (SC) network structures into
validated Ising and QUBO models. It provides deterministic classical solving,
optional native acceleration, optional D-Wave submission, analysis,
decomposition, transformations, and model export.

This is research software. It does not establish quantum advantage, certify a
minor embedding, validate a physical QPU, or claim parity with unrelated
quantum-circuit modules in `sc_neurocore.quantum`.

## Architecture

The former 1,910-line implementation is now a 98-line compatibility façade and
nine responsibility modules. Historical imports and pickle identities remain
at `sc_neurocore.bridges.quantum_annealing`.

| Module | Responsibility | Lines |
|---|---|---:|
| `quantum_annealing.py` | Stable exports, optional-backend observables, pickle identity | 98 |
| `annealing_backends.py` | Optional native, `dimod`, and D-Wave adapters | 139 |
| `annealing_models.py` | Validated Ising/QUBO value objects and backend selection | 276 |
| `annealing_compilers.py` | SC adjacency, bitstream, and pruning compilers | 235 |
| `annealing_solvers.py` | Python/native simulated annealing and D-Wave adapter | 267 |
| `annealing_analysis.py` | Landscape, embedding, sample, and TTS analysis | 317 |
| `annealing_hardware.py` | Hardware-capacity estimates and chain resolution | 203 |
| `annealing_transforms.py` | Schedules, gauges, and SC precision encodings | 245 |
| `annealing_io.py` | Deterministic export and text visualisation | 118 |
| `annealing_decomposition.py` | Overlapping partitioning and reconstruction | 173 |

The responsibility graph is acyclic. No responsibility module imports the
façade, and an architecture test enforces the dependency direction and size
limits.

## Public surface

All 24 names are available from both the historical module and
`sc_neurocore.bridges`.

| Group | Public names |
|---|---|
| Models | `ProblemType`, `QubitSpec`, `CouplerSpec`, `IsingModel`, `QUBOModel` |
| Compilers | `SCToIsing`, `SCToQUBO`, `SCBitstreamQUBO` |
| Solvers | `SimulatedAnnealer`, `DWaveInterface` |
| Analysis | `EnergyLandscape`, `EmbeddingAnalyzer`, `SampleAggregator`, `TTSAnalyzer` |
| Hardware and decomposition | `HardwareGraph`, `ChainBreakResolver`, `ProblemDecomposer` |
| Transformations | `AnnealingSchedule`, `GaugeTransform`, `SCPrecisionEncoder` |
| Export | `export_ising_json`, `export_qubo_json`, `export_bqm`, `visualize_ising` |

## Quick start

```python
import numpy as np

from sc_neurocore.bridges.quantum_annealing import (
    SCToIsing,
    SimulatedAnnealer,
)

adjacency = np.array(
    [
        [0.0, 0.8, -0.2],
        [0.8, 0.0, 0.4],
        [-0.2, 0.4, 0.0],
    ],
    dtype=np.float64,
)
model = SCToIsing().compile(adjacency, node_labels=["a", "b", "c"])
result = SimulatedAnnealer(
    n_sweeps=200,
    seed=42,
    backend="python",
).solve_ising(model, num_reads=10)

print(result["best_spins"], result["best_energy"])
```

Choose `backend="python"` when repeatable Python-path evidence is required.
Choose `backend="rust"` to require the native engine; a missing native backend
then raises a stable `RuntimeError`. `backend="auto"` may select Rust according
to the thresholds documented below.

## Model contracts

`IsingModel` uses

```text
E(s) = offset + Σᵢ hᵢsᵢ + Σᵢ<ⱼ Jᵢⱼsᵢsⱼ,  sᵢ ∈ {-1, +1}
```

and `QUBOModel` uses

```text
E(x) = offset + Σᵢ≤ⱼ Qᵢⱼxᵢxⱼ,  xᵢ ∈ {0, 1}.
```

Both models canonicalise pair indices, combine reversed duplicates, remove
zero-valued terms, infer `n_qubits` from terms when it is omitted, and reject
non-finite values, invalid indices, out-of-bounds labels, and duplicate label
values. Missing variables preserve the historical defaults: `+1` for an Ising
spin and `0` for a QUBO bit.

`QUBOModel.to_ising()` uses `x = (s + 1) / 2` and is exactly
energy-equivalent. In particular, a diagonal term `Qᵢᵢxᵢ` contributes
`Qᵢᵢ / 2` to both `hᵢ` and the offset. Off-diagonal terms contribute
`Qᵢⱼ / 4` to the coupling, both incident fields, and the offset. Exhaustive
tests compare every assignment for small models.

## Compilers

### SC adjacency compilers

`SCToIsing.compile()` and `SCToQUBO.compile()` accept a finite, non-empty,
square matrix and optional unique labels. Directed pairs are averaged before a
single canonical coupling is emitted.

- `SCToIsing` maps positive averaged weights to ferromagnetic (negative)
  couplings and applies the configured field and coupling scales.
- `SCToQUBO` uses negative absolute column sums on the diagonal and scaled
  averaged off-diagonal weights.

Both reject malformed matrices, non-finite values, mismatched biases, and
invalid names or labels.

### SC bitstream QUBOs

`SCBitstreamQUBO.weight_optimization()` encodes
`||target - candidate_weights @ x||²` for binary `x`. `n_bits` selects the first
candidate columns and must be a positive integer no greater than the available
column count; it is never silently clamped.

`SCBitstreamQUBO.pruning()` creates one variable for every undirected candidate
edge found in either adjacency direction. It averages the two directed
importance scores and applies a quadratic penalty for selecting exactly
`max_connections`. Impossible cardinalities fail before a model is returned.

### SC precision encoding

`SCPrecisionEncoder` supports binary, unary (thermometer), and one-hot
representations of finite values clipped to `[0, 1]`.

| Encoding | Levels | Qubits per value |
|---|---:|---:|
| `binary` | `2**n_bits` | `n_bits` |
| `unary` | `n_bits + 1` | `n_bits` |
| `one_hot` | `n_bits` | `n_bits` |

Array encoding requires a finite, non-empty one-dimensional array and assigns
non-overlapping global qubit indices. Decoding rejects invalid indices, values,
and multiple active one-hot bits.

## Solver and backend behaviour

### Simulated annealing

`SimulatedAnnealer` implements seeded single-spin Metropolis sweeps. The Python
and Rust paths return the same mapping-shaped contract:

- `best_spins`: `dict[int, int]`
- `best_energy`: `float`
- `energies`: one finite value per returned sample
- `samples`: `list[dict[int, int]]`
- `n_sweeps`, `num_reads`, and `backend`

`solve_qubo()` converts through the exact Ising mapping and returns the
corresponding bit-valued contract. Native responses are validated for shape,
spin domain, energy finiteness, and aligned sample counts before they cross the
public boundary. The configured seed is forwarded to Rust.

Backend dispatch is explicit:

| Caller | `auto` native threshold | Explicit behaviour |
|---|---:|---|
| `IsingModel.energy()` | Rust available and more than 20 qubits | `python` stays Python; `rust` requires Rust |
| `SimulatedAnnealer.solve_ising()` | Rust available and more than 10 qubits | `python` stays Python; `rust` requires Rust |
| `EnergyLandscape.analyze()` | Rust available and more than 100 samples | `python` stays Python; `rust` requires Rust |

### D-Wave adapter

`DWaveInterface.available` means only that `dimod` and the required Ocean SDK
classes are importable. It does not prove credentials, network access, solver
entitlement, or QPU health.

When those imports are absent, `solve_ising()` uses a bounded local simulated
annealing fallback with at most 20 reads and labels the result
`simulated_annealing_fallback`. When the SDK is installed, sampler construction,
authentication, provider, and submission failures propagate; they are not
misreported as local success. A successful QPU result is checked for a valid
spin mapping and finite best energy.

Install the optional local annealing dependencies with:

```bash
python -m pip install 'sc-neurocore[annealing]'
```

Real QPU access additionally requires a compatible Ocean installation and
provider credentials outside this repository.

## Analysis, hardware, and decomposition

- `EnergyLandscape` exhaustively enumerates models up to 20 qubits. Larger
  models use a deterministic configurable random sample unless samples are
  supplied by the caller.
- `EmbeddingAnalyzer` reports graph density, degree, and a Pegasus-oriented
  chain-size estimate. It is a capacity estimate, not a minor embedding.
- `SampleAggregator` validates aligned samples and energies, deduplicates spin
  patterns, and reports histogram and Boltzmann-weighted summaries.
- `TTSAnalyzer` validates probabilities, times, energy samples, and solver
  payloads before computing the standard cumulative-success estimate.
- `HardwareGraph` models idealised Chimera, Pegasus, and Zephyr capacity. Its
  degree-based `can_embed()` result is not hardware placement evidence.
- `ChainBreakResolver` validates non-overlapping physical chains and resolves
  by majority vote or local energy minimisation. The latter requires a model.
- `AnnealingSchedule` rejects non-finite or non-positive timings and invalid
  anneal fractions for linear, pause-and-quench, and reverse schedules.
- `GaugeTransform` produces deterministic energy-equivalent Python models and
  validates samples when returning them to the original spin frame.
- `ProblemDecomposer` creates deterministic graph-aware partitions with up to
  the configured overlap from connected assigned neighbours. Submodels use
  local indices, while reconstruction retains an exact local-to-global index
  map; it does not infer identity from optional labels.

Decomposition is a heuristic orchestration strategy. It does not guarantee a
global optimum or a favourable embedding.

## Export behaviour

`export_ising_json()` and `export_qubo_json()` write sorted, deterministic UTF-8
JSON through a same-directory temporary file, flush and synchronise it, and
atomically replace the destination. A failed write removes the temporary file.
`export_bqm()` returns a `dimod` spin BQM when available and `None` otherwise.
`visualize_ising()` returns a stable human-readable text representation.

## Polyglot authority

The Python bridge is the maintained orchestration authority. The maintained
native implementation is `engine/src/quantum.rs`, exposed through the PyO3
engine package and covered by 12 focused Rust tests.

Earlier generated files under the Rust safety registry, Go services, Julia
bridges, and Mojo kernels were not alternate implementations: the Rust safety
file returned constants, the Go functions were empty, and the Julia and Mojo
files did not build. They have been removed together with their registry
entries. No Go, Julia, Mojo, or generated-Rust parity or performance claim is
made for this bridge.

The maintained native comparison harness remains available:

```bash
python benchmarks/bench_quantum_annealing_rust_vs_python.py
```

Its results are machine- and load-specific; rerun it in the target release
environment before publishing a native speed claim.

## Modularisation benchmark

`benchmarks/bench_quantum_annealing_modularisation.py` compares the committed
single-file parent (`9308910a5d863ebfb338244b43d10f73f25cfbc6`) with the
modular candidate. It uses 30 measured child processes after five warm-ups,
alternates variant order, pins each child with `taskset`, records raw samples
and load context, and binds both variants to source digests.

| Metric | Parent median | Modular median | Delta |
|---|---:|---:|---:|
| Cold import | 383.554 ms | 286.231 ms | -25.37% |
| Eight-node compile | 0.0389 ms | 0.0776 ms | +99.65% |
| Python solve | 1.8522 ms | 1.5434 ms | -16.67% |
| Child wall time | 569.268 ms | 437.154 ms | -23.21% |
| Maximum RSS | 41,326 KiB | 39,888 KiB | -3.48% |

The candidate source digest is
`9d5152b02377c5358a7eb678ab9b35633fbe40deefd6c0a7a8b7996978023d1a`.
The workstation load average rose from 7.46 to 11.53 during capture, and CPU 0
was affinity-pinned but not exclusively isolated. These measurements are local
regression diagnostics, not release throughput or hardware claims. Repeat on
reserved isolated cores before promotion.

Reproduce the committed evidence with:

```bash
python benchmarks/bench_quantum_annealing_modularisation.py \
  --baseline-root /path/to/clean-parent \
  --candidate-root . \
  --output benchmarks/results/local_python_quantum_annealing.json
```

## Verification

The focused cohort is split by responsibility instead of rebuilding a test
GodFile. Its largest file is 445 lines. The linked cohort passes 250 tests with
one optional `neal` parity test skipped when that extra is absent. Exact-file
coverage passes 216 tests and records 100% of 1,040 statements and 442 branches
with no misses or partial branches.

```bash
PYTHONPATH=src:. python -m pytest \
  tests/test_bridges/test_quantum_annealing_models_compilers.py \
  tests/test_bridges/test_quantum_annealing_solvers_backends.py \
  tests/test_bridges/test_quantum_annealing_analysis_hardware.py \
  tests/test_bridges/test_quantum_annealing_transforms_io.py \
  tests/test_bridges/test_quantum_annealing_decomposition_architecture.py \
  tests/test_bridges/test_quantum_annealing_neal_parity.py \
  tests/test_bench_quantum_annealing_modularisation.py

cargo test --manifest-path engine/Cargo.toml quantum --lib
```

Coverage includes exact QUBO-to-Ising energy equivalence, backend selection and
malformed native returns, QPU fallback/error boundaries, compiler objectives,
schedule and hardware validation, atomic export cleanup, exact decomposition
index reconstruction, historical import and pickle identities, the architecture
DAG, and benchmark schema/source binding.

## Limitations

- No real QPU submission is exercised by the dependency-light test cohort.
- `available` checks imports, not credentials or provider health.
- Hardware and embedding results are conservative estimates, not placement
  proofs.
- Problem decomposition is heuristic and offers no optimum guarantee.
- Optional `neal` parity requires `sc-neurocore[annealing]`.
- Benchmark evidence does not demonstrate quantum speedup or production
  throughput.

## Auto-rendered API

::: sc_neurocore.bridges.quantum_annealing
    options:
      show_root_heading: true
      show_source: true
      members:
        - ProblemType
        - QubitSpec
        - CouplerSpec
        - IsingModel
        - QUBOModel
        - SCToIsing
        - SCToQUBO
        - SCBitstreamQUBO
        - SimulatedAnnealer
        - DWaveInterface
        - EnergyLandscape
        - EmbeddingAnalyzer
        - SampleAggregator
        - TTSAnalyzer
        - HardwareGraph
        - ChainBreakResolver
        - ProblemDecomposer
        - AnnealingSchedule
        - GaugeTransform
        - SCPrecisionEncoder
        - export_ising_json
        - export_qubo_json
        - export_bqm
        - visualize_ising
