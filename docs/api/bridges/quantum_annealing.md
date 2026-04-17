# Quantum Annealing Bridge

**Module:** `sc_neurocore.bridges.quantum_annealing`
**Source:** `src/sc_neurocore/bridges/quantum_annealing.py` — 1883 LOC
**Status (v3.14.0):** 24 public exports across 18 classes + 4
exporters + 1 enum + 1 helper; 198-test bridges suite passes;
**Rust accelerator path declared via `sc_neurocore_engine`** (see §6
honesty notice — engine wheel not installed in this measurement
environment, so the Rust speedup numbers are NOT measured here).
`__tier__ = "research"`. The `dimod` and `dwave-ocean-sdk` deps are
soft-imported (graceful fallback).

This page covers the third of three speculative hardware bridges.
Sister pages:
- DNA strand displacement: [`api/bridges/dna_mapper.md`](dna_mapper.md)
- Photonic NoC: [`api/bridges/photonic_noc.md`](photonic_noc.md)

---

## 1. What this bridge does

Compiles an SC neural network's adjacency matrix into Ising or
QUBO form for D-Wave annealers and classical simulated-annealing
solvers:

```
SC Network adjacency  →  SCToIsing / SCToQUBO  →  IsingModel / QUBOModel
       (NxN)                   ↓                          ↓
                          Compiler              SimulatedAnnealer  ── Rust path ──
                                                          ↓                  ↓
                                                  best_spins +         Python fallback
                                                  best_energy
                                                          ↓
                                                  DWaveInterface (optional QPU)
                                                          ↓
                                                  Sample distribution
```

Six analysis / utility classes wrap the core path:
`EmbeddingAnalyzer` (D-Wave Pegasus topology fit),
`ChainBreakResolver` (post-processing), `AnnealingSchedule`
(custom annealing curves), `GaugeTransform` (gauge averaging for
ICE mitigation), `ProblemDecomposer` (large-problem partitioning),
`TTSAnalyzer` (time-to-solution scaling).

---

## 2. Public surface

24 symbols re-exported from `sc_neurocore.bridges.__init__`:

| Group | Symbols |
|-------|---------|
| Enums + dataclasses | `ProblemType`, `QubitSpec`, `CouplerSpec`, `IsingModel`, `QUBOModel` |
| Compilers | `SCToIsing`, `SCToQUBO`, `SCBitstreamQUBO`, `SCPrecisionEncoder` |
| Solvers / interfaces | `SimulatedAnnealer`, `DWaveInterface` |
| Analysis | `EnergyLandscape`, `EmbeddingAnalyzer`, `TTSAnalyzer`, `SampleAggregator` |
| Hardware-graph utilities | `HardwareGraph`, `ChainBreakResolver`, `AnnealingSchedule`, `GaugeTransform`, `ProblemDecomposer` |
| Exporters | `export_ising_json`, `export_qubo_json`, `export_bqm`, `visualize_ising` |

Module-level constants:

| Constant | Value | Note |
|----------|------:|------|
| `_DEFAULT_CHAIN_STRENGTH` | `2.0` | for D-Wave embedding |
| `_DEFAULT_NUM_READS` | `1000` | per QPU call |
| `_DEFAULT_ANNEALING_TIME_US` | `20.0` | μs |
| `_BOLTZMANN_K` | `1.380649e-23` | J/K, physical |

---

## 3. Compilers: `SCToIsing` / `SCToQUBO`

Both compilers accept an N×N adjacency matrix and produce a model
with N qubits + couplings derived from non-zero off-diagonal
weights.

```python
ising_model: IsingModel = SCToIsing().compile(adjacency)
qubo_model:  QUBOModel  = SCToQUBO().compile(adjacency)
```

- `IsingModel.h: dict[int, float]` — bias per qubit
- `IsingModel.J: dict[(int, int), float]` — coupling per edge
- `QUBOModel.Q: dict[(int, int), float]` — full upper-triangular
  matrix (diagonal = bias, off-diagonal = coupling)

The two are mathematically equivalent
(`s_i = 2*x_i - 1, x_i ∈ {0,1}`, `s_i ∈ {-1,+1}`); the
representation choice depends on the downstream solver.

`SCBitstreamQUBO` (line 1259) is a specialised QUBO compiler that
encodes raw SC bitstreams as binary variables — useful when the
problem is bitstream-level rather than network-level.

`SCPrecisionEncoder` (line 1473) supports `binary` (1-hot per bit)
or `unary` (thermometer) encoding of analog values into multiple
qubits. Default 8 bits per value.

---

## 4. Solvers

### 4.1 `SimulatedAnnealer`

```python
class SimulatedAnnealer:
    def __init__(
        self,
        n_sweeps: int = 1000,
        beta_start: float = 0.1,
        beta_end: float = 10.0,
        seed: int = 42,
    ): ...

    def solve_ising(self, model: IsingModel, num_reads: int = 10) -> dict: ...
```

Single-spin Metropolis sweeps with geometric beta schedule from
`beta_start` to `beta_end` over `n_sweeps`. Returns `dict` with:
- `best_spins`: `np.ndarray[int8]` of length `n_qubits`
- `best_energy`: `float`
- `energies`: `list[float]` per `num_reads`
- `samples`: `np.ndarray[num_reads, n_qubits]`

Per-instance `seed` → reproducible runs. Same seed → identical
output (confirmed by reading source — uses `np.random.default_rng`).

### 4.2 Rust acceleration path (declared, not measured here)

`SimulatedAnnealer.solve_ising` (line 467) branches on
`_HAS_RUST_QA and model.n_qubits > 10`:

```python
if _HAS_RUST_QA and model.n_qubits > 10:
    return self._solve_ising_rust(model, num_reads)
return self._solve_ising_python(model, num_reads)
```

The Rust path uses 6 PyO3 bindings exported by `sc_neurocore_engine`:
- `py_qa_ising_energy` — single-state energy
- `py_qa_simulated_annealing` — full SA loop
- `py_qa_batch_ising_energy` — vectorised batch energy
- `py_qa_gauge_transform` — gauge transform for ICE mitigation
- `py_qa_generate_gauges` — random gauge generator
- `py_qa_greedy_partition` — `ProblemDecomposer` accelerator

The class docstring claims "100×+ speedup for models with >20
qubits" — **this number is from the source comment**, not measured
in this environment. The `sc_neurocore_engine` wheel is not
installed on this workstation, so the Rust path falls back to
Python every time. To verify the claim:

```bash
pip install sc-neurocore-engine    # provides the Rust .so
PYTHONPATH=src pytest tests/test_bridges/test_quantum_annealing.py::test_rust_parity
```

Tracked as task #49.

### 4.3 `DWaveInterface`

```python
class DWaveInterface:
    def __init__(self, solver: str = "Advantage_system6.4"): ...

    def submit(
        self,
        ising: IsingModel,
        num_reads: int = 1000,
        annealing_time_us: float = 20.0,
        chain_strength: float = 2.0,
    ) -> dict: ...
```

Soft-imports `dwave-ocean-sdk` at runtime. Raises
`ImportError("dwave-ocean-sdk required for DWave QPU access")` if
absent. The interface wraps `EmbeddingComposite(DWaveSampler())`
and returns the sample-set as a dict (energies, occurrences,
chain-break fraction).

The wheel + an active D-Wave Leap account are required to
exercise this path. Not measured here.

---

## 5. Analysis classes

| Class | Role | Cited basis |
|-------|------|-------------|
| `EnergyLandscape` | exhaustive enumeration of small problems (≤16 qubits) | classical |
| `EmbeddingAnalyzer` | embed a logical problem into D-Wave Pegasus topology | Choi 2008 minor-embedding |
| `TTSAnalyzer` | time-to-solution scaling per Rønnow et al. 2014 | *Science* 345:420-424 |
| `SampleAggregator` | de-duplicate samples by spin pattern + summary statistics | classical |
| `HardwareGraph` | model D-Wave Pegasus or Chimera graph topology | D-Wave hardware spec |
| `ChainBreakResolver` | resolve broken chains via majority vote / energy minimisation | D-Wave practice |
| `AnnealingSchedule` | non-monotonic annealing curves (e.g. pause-and-quench) | Marshall et al. 2017 |
| `GaugeTransform` | gauge averaging to mitigate intrinsic control errors (ICE) | Pelofske et al. 2020 |
| `ProblemDecomposer` | partition large problems into hardware-fitting chunks | Booth et al. 2017 (qbsolv) |

All 9 classes are pure-Python with the exception of `GaugeTransform`
and `ProblemDecomposer`, which delegate to the Rust engine when
available (`_rust_gauge`, `_rust_gen_gauges`, `_rust_partition`).

---

## 6. Honesty notice — Rust speedup is unverified in this env

Three classes claim Rust acceleration:
1. `SimulatedAnnealer` (line 467) — "100×+ speedup for models with
   >20 qubits"
2. `GaugeTransform` (line 1180) — uses `_rust_gauge` /
   `_rust_gen_gauges`
3. `ProblemDecomposer` (line 1588) — uses `_rust_partition`

In this measurement environment, `sc_neurocore_engine` is not
importable (`_HAS_RUST_QA = False`), so the Python fallback runs
in every test. The §7 numbers below are **pure-Python**.

The "100×" claim in the docstring is based on the source-comment
expectation, not a measurement reproduced this session. To
characterise it properly: install the engine, then run a benchmark
matrix at N ∈ {10, 20, 50, 100, 200} qubits comparing
`_solve_ising_rust` vs `_solve_ising_python` wall time, repeat
each ≥5 times, report median ratio. Tracked as task #49.

---

## 7. Performance — pure-Python path (this workstation)

Random Erdős–Rényi adjacency at p=0.1, undirected, single
compile + 5-read SA with 100 sweeps:

| N | density | `SCToQUBO.compile` | `SCToIsing.compile` | SA solve (5 reads × 100 sweeps) |
|---:|--------:|-------------------:|--------------------:|--------------------------------:|
| 10 | 0.100 | 0.48 ms | 0.10 ms | 10.19 ms |
| 50 | 0.100 | 1.22 ms | 0.92 ms | 278.81 ms |
| 100 | 0.100 | 3.29 ms | 2.36 ms | 2 205.23 ms |

Compile cost is roughly linear in `n_edges`. The SA solve cost is
**super-linear** (~4× per 2× N) — confirming the spin-by-spin
Python loop is the bottleneck and motivating the Rust path. At
N=100 a single solve already takes 2 seconds; N=1000 with default
sweeps would take ~3 minutes per read, ~50 minutes for the
default 1000 reads.

Hardware: Intel i5-11600K, NumPy 2.2.6, no Rust wheel. With the
Rust wheel installed and assuming the docstring claim, the N=100
case should drop to ~22 ms (100× speedup) — unverified.

---

## 8. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.bridges.quantum_annealing import SCToIsing, ...` | `bridges/__init__.py` re-exports all 24 symbols | `tests/test_bridges/test_quantum_annealing.py` |
| `SCToQUBO.compile` ↔ `SCToIsing.compile` | independent compilers; both accept N×N matrix | dedicated tests for each |
| `SimulatedAnnealer.solve_ising` Rust dispatch | `_HAS_RUST_QA and model.n_qubits > 10` branch | covered when engine wheel present; falls through to Python otherwise |
| `DWaveInterface` | soft-imports `dwave.system` lazily | tests skip when wheel absent |
| `export_bqm` | requires `dimod`; raises if absent | dimod-skip path tested |

---

## 9. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_bridges/test_quantum_annealing.py -q
# (part of the 198-test bridges suite — verified 2026-04-17)
```

`tests/test_bridges/test_quantum_annealing.py` is 652 lines
covering: dataclass round-trip, `SCToQUBO`/`SCToIsing` compilation
on small matrices, `SimulatedAnnealer` solver determinism with
fixed seed, `EnergyLandscape` exhaustive enumeration on ≤4 qubits
(matches Python brute-force), `EmbeddingAnalyzer` chain length
estimation, `ChainBreakResolver` majority-vote correctness,
`GaugeTransform` round-trip, `TTSAnalyzer` scaling estimate,
`SampleAggregator` de-duplication.

What is NOT covered:
- Rust speedup verification (engine wheel absent — task #49)
- Real D-Wave QPU submission (requires Leap account)
- Large-problem decomposition (`ProblemDecomposer` tested only on
  N≤30; partition correctness at N=10⁴ would need stress test)
- `DWaveInterface` happy-path (skip-if-no-dwave)

---

## 10. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 24 symbols re-exported and tested |
| 2 | Multi-angle tests | ✅ PASS | 652-line dedicated test file in 198-test bridges suite |
| 3 | Rust path | ⚠️ WARN | Path **declared** via 6 PyO3 bindings (good); engine wheel **not installed** in this env so fallback runs (acceptable); docstring "100× speedup" claim unverified here (task #49) |
| 4 | Benchmarks | ⚠️ WARN | Pure-Python numbers measured (§7); Rust comparison not measured |
| 5 | Performance docs | ✅ PASS | §7 with explicit "pure-Python only" caveat |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | SPDX header ✅. Soft-imports for `dimod`, `dwave-ocean-sdk`, `sc_neurocore_engine` all guarded. British English in this doc; source uses standard scientific-Python identifiers (acceptable per docs-vs-code rule). |

Net: **2 WARN, 0 FAIL.** Both WARNs trace to the unverified Rust
speedup claim — closeable by installing the engine wheel and
re-running the benchmark.

---

## 11. Known issues

### 11.1 Rust speedup unverified (task #49)

See §6. Headline issue. Install engine wheel, run benchmark
matrix, update §7 with the comparison. Until then the "100×"
docstring claim is aspirational.

### 11.2 `SCBitstreamQUBO` and `SCPrecisionEncoder` are advanced — undocumented in this page

The two specialised compilers (lines 1259, 1473) handle bitstream-
level QUBO and multi-bit precision encoding. They have separate
APIs and use cases that warrant their own subsection or sister
page. Tracked as task #50.

### 11.3 No D-Wave hardware-parity test

`SCToIsing → SimulatedAnnealer` is tested. `SCToIsing →
DWaveInterface → QPU → samples` is not (no Leap account in CI).
Adding a parity test against `neal.SimulatedAnnealingSampler`
(D-Wave's reference SA) would validate the compiler output without
needing real hardware. Tracked as task #51.

### 11.4 `EmbeddingAnalyzer` assumes Pegasus topology

`EmbeddingAnalyzer.__init__(topology="pegasus", size=16)` defaults
to D-Wave Advantage's Pegasus graph. Older Chimera (D-Wave 2000Q)
and the new Zephyr (Advantage2) need explicit topology selection.
Document the topology options in the class docstring.

### 11.5 Rust dispatch threshold is hard-coded

`SimulatedAnnealer.solve_ising` only dispatches to Rust when
`model.n_qubits > 10` (line 467). The threshold is a magic
number; expose as `__init__` parameter or class constant.

---

## 12. References

Quantum annealing theory:

- Kadowaki T., Nishimori H. "Quantum annealing in the transverse
  Ising model." *Phys Rev E* 58:5355-5363 (1998). The original
  QA proposal.
- Farhi E. *et al.* "Quantum computation by adiabatic evolution."
  arXiv:quant-ph/0001106 (2000). Adiabatic quantum computation
  formalism.

D-Wave hardware + minor-embedding:

- Choi V. "Minor-embedding in adiabatic quantum computation: I.
  The parameter setting problem." *Quantum Inf Process* 7:193-209
  (2008). Minor-embedding theory for `EmbeddingAnalyzer`.
- Boothby K. *et al.* "Next-Generation Topology of D-Wave Quantum
  Processors." arXiv:2003.00133 (2020). Pegasus topology used in
  `HardwareGraph`.

Solvers + analysis:

- Rønnow T. F. *et al.* "Defining and detecting quantum speedup."
  *Science* 345:420-424 (2014). TTS methodology used by
  `TTSAnalyzer`.
- Marshall J. *et al.* "Power of pausing: Advancing understanding
  of thermalization in experimental quantum annealers." *Phys Rev
  Applied* 11:044083 (2019). Inspiration for `AnnealingSchedule`
  pause-and-quench.
- Pelofske E. *et al.* "Decomposition Algorithms for Solving NP-hard
  Problems on a Quantum Annealer." *J Signal Process Syst* 93:405-420
  (2021). `ProblemDecomposer` ancestor.
- Booth M. *et al.* "Partitioning Optimization Problems for Hybrid
  Classical/Quantum Execution." D-Wave Technical Report (2017).
  qbsolv methodology.

Internal:

- Bridges sister: [`api/bridges/dna_mapper.md`](dna_mapper.md),
  [`api/bridges/photonic_noc.md`](photonic_noc.md)
- IBM Credits Application: outside this repo's scope; see
  agent-shared session logs for QPU access status.

---

## 13. Auto-rendered API

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
        - SimulatedAnnealer
        - DWaveInterface
        - EnergyLandscape
        - EmbeddingAnalyzer
        - HardwareGraph
        - ChainBreakResolver
        - AnnealingSchedule
        - GaugeTransform
        - SCBitstreamQUBO
        - SampleAggregator
        - SCPrecisionEncoder
        - ProblemDecomposer
        - TTSAnalyzer
        - export_ising_json
        - export_qubo_json
        - export_bqm
        - visualize_ising
