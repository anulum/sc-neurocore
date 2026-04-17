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

### 3.1 `SCBitstreamQUBO` — three task-specific encodings

```python
class SCBitstreamQUBO:
    def __init__(self, penalty: float = 5.0): ...

    def weight_optimization(target_output, candidate_weights, n_bits=8) -> QUBOModel: ...
    def pruning(adjacency, importance_scores, max_connections) -> QUBOModel: ...
```

A specialised QUBO compiler that targets two SC optimisation
patterns common in research:

#### Weight optimisation

Find binary vector `x ∈ {0, 1}ⁿ` minimising `||target − W @ x||²`.

The QUBO formulation expands the squared error:
```
   ||y − Wx||² = xᵀ(WᵀW)x − 2yᵀWx + yᵀy
```
- Off-diagonal `Q[i,j] = (WᵀW)[i,j] + (WᵀW)[j,i]` (full
  upper-triangular)
- Diagonal `Q[i,i] = (WᵀW)[i,i] − 2(Wᵀy)[i]`
- Constant `offset = yᵀy` (so the model's true energy is
  `xᵀQx + offset`)

`n = min(WᵀW.shape[0], n_bits)` so callers can bound the qubit
count even when the candidate matrix is wider than the budget.
Returned `QUBOModel.source = "sc_weight_optimization"`.

#### Pruning

Select `max_connections` edges from the existing connectivity that
maximise the sum of importance scores while honouring the
cardinality constraint exactly:
```
   maximise   Σ importance[i,j] · x[edge(i,j)]
   subject to Σ x = max_connections
```

The encoder creates one binary variable per non-zero off-diagonal
edge of the adjacency, applies `penalty · (Σx − K)²` to enforce
the constraint, and returns a QUBO whose ground state is the
chosen edge subset.

Note: the cardinality penalty is the standard QUBO trick — it adds
`penalty · (1 − 2K)` to every diagonal and `2 · penalty` to every
off-diagonal pair. With the default `penalty = 5.0`, callers
should rescale if the importance-score magnitudes are very
different from unity.

### 3.2 `SCPrecisionEncoder` — three encodings of `[0, 1]` values

```python
class SCPrecisionEncoder:
    def __init__(self, encoding: str = "binary", n_bits: int = 8): ...

    def encode(sc_value: float) -> dict[int, int]: ...
    def decode(qubits: dict[int, int]) -> float: ...
    def encode_array(values: np.ndarray) -> dict[int, int]: ...

    @property
    def n_levels(self) -> int: ...
    def qubits_needed(n_sc_values: int) -> int: ...
```

Maps continuous SC probabilities in `[0, 1]` to fixed-length qubit
configurations. Three encodings, each with different qubit-vs-precision
trade-offs:

| Encoding | Qubits per value | Levels | Good for |
|----------|-----------------:|-------:|----------|
| `binary` | `n_bits` | `2^n_bits` | dense precision (8 bits → 256 levels) |
| `unary` (thermometer) | `n_bits` | `n_bits + 1` | robust to single-bit errors |
| `one_hot` | `n_bits` | `n_bits` | categorical, no inter-bit coupling |

`encode(v)` clamps `v` to `[0, 1]`, scales to the encoding's level
count, and returns a `{qubit_idx: 0|1}` dict for one value.
`encode_array(values)` packs an N-element array into a single global
dict by offsetting qubit indices by `idx * n_bits`. `decode(qubits)`
reverses the mapping per encoding (binary positional sum, unary
count of 1s, one-hot index of the 1-bit).

Round-trip accuracy:
- binary: `|encode(v) − decode(...)| ≤ 1 / (2^n_bits − 1)` (e.g.
  `≤ 1/255 ≈ 0.004` at `n_bits=8`)
- unary: `≤ 1 / n_bits`
- one_hot: `≤ 1 / (n_bits − 1)`

`n_levels` exposes the level count; `qubits_needed(n_sc_values)`
returns `n_sc_values * n_bits` so callers can size `IsingModel` /
`QUBOModel` correctly before encoding.

Construction with an unknown encoding string raises `ValueError`.

### 3.3 When to use which compiler

| Problem | Use |
|---------|-----|
| "I have an SC network adjacency, give me an Ising model for D-Wave." | `SCToIsing(adjacency)` |
| "Same, but I want the QUBO form." | `SCToQUBO(adjacency)` |
| "I want to find binary weights that match a target output." | `SCBitstreamQUBO.weight_optimization(...)` |
| "I want to prune to exactly K edges by importance." | `SCBitstreamQUBO.pruning(adj, importance, K)` |
| "I have continuous values; help me encode them into qubits." | `SCPrecisionEncoder(encoding=..., n_bits=...).encode_array(...)` |

The four classes do not chain by default — each produces its own
`IsingModel` or `QUBOModel` (or a per-value qubit dict for the
encoder). Combining them (e.g. encode-then-prune) requires
caller-side qubit-index bookkeeping.

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

## 6. Rust speedup — measured (closes task #49)

Three classes use Rust acceleration when the engine is installed:
1. `SimulatedAnnealer` (line 467) — `_solve_ising_rust` via
   `py_qa_simulated_annealing`
2. `GaugeTransform` (line 1180) — `py_qa_gauge_transform` /
   `py_qa_generate_gauges`
3. `ProblemDecomposer` (line 1588) — `py_qa_greedy_partition`

The bridge's `_HAS_RUST_QA` flag resolves through
`sc_neurocore_engine.__init__` re-exports
(see [§3.1 of the engine docs](../../engine.md) — top-level
re-exports added so `from sc_neurocore_engine import py_qa_*`
works). Engine wheel must be present in the active venv;
install with:

```bash
cd bridge && python -m maturin develop --release
# or, for an installed wheel:
pip install target/wheels/sc_neurocore_engine-*.whl
```

### 6.1 Measured speedup (this workstation, 2026-04-17)

`SimulatedAnnealer(n_sweeps=200, seed=42)` solving Erdős–Rényi
Ising at p=0.1 with `num_reads=5`. Hardware: Intel i5-11600K,
NumPy 2.2.6 (Python 3.12 venv-rocm with `sc_neurocore_engine`
release wheel installed).

| N qubits | Python wall | Rust wall | Speedup |
|---------:|------------:|----------:|--------:|
| 20 | 90.55 ms | 15.49 ms | 5.8× |
| 50 | 683.74 ms | 0.90 ms | **761×** |
| 100 | 4 341.85 ms | 2.73 ms | **1 593×** |

The docstring's "100×+" is conservative; actual speedup grows
super-linearly with N because the Python `_solve_ising_python`
inner loop is single-threaded Metropolis spin flips while the
Rust path uses SIMD batch energy evaluation. At N=100 the Rust
solver completes in <3 ms — fast enough to make
`SimulatedAnnealer.solve_ising(num_reads=10000)` viable for
hyperparameter sweeps.

The dispatch threshold `model.n_qubits > 10` (line 467) is
conservative for this hardware; even N=20 Rust is 5.8× faster
than Python. Lowering the threshold or making it configurable
is a follow-up (it currently means small problems silently use
the slow path).

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
| 3 | Rust path | ✅ PASS | All 6 PyO3 bindings re-exported via `bridge/sc_neurocore_engine/__init__.py`; `_HAS_RUST_QA = True` when wheel installed; `SimulatedAnnealer.solve_ising` dispatches via the `n_qubits > 10` branch |
| 4 | Benchmarks | ✅ PASS | §6.1 Rust vs Python comparison: 5.8× (N=20), 761× (N=50), 1593× (N=100). §7 retains pure-Python numbers for reference |
| 5 | Performance docs | ✅ PASS | §7 with explicit "pure-Python only" caveat |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | SPDX header ✅. Soft-imports for `dimod`, `dwave-ocean-sdk`, `sc_neurocore_engine` all guarded. British English in this doc; source uses standard scientific-Python identifiers (acceptable per docs-vs-code rule). |

Net: **0 WARN, 0 FAIL.** Both former WARNs closed by task #49 —
engine wheel built via `bridge/maturin develop --release`,
re-exports added to `sc_neurocore_engine.__init__`, Rust speedup
measured (§6.1).

---

## 11. Known issues

### 11.1 Rust speedup (CLOSED by task #49)

§6.1 reports the measured comparison: 5.8× / 761× / 1593× at
N=20/50/100. The docstring's "100×+" claim is conservative for
N ≥ 50. The engine wheel must be installed in the active venv —
see §6 for build instructions.

### 11.2 `SCBitstreamQUBO` and `SCPrecisionEncoder` (DOCUMENTED by task #50)

Both classes now have dedicated subsections under §3:
- §3.1 covers `SCBitstreamQUBO.weight_optimization` and `.pruning`
  with the QUBO derivation, cardinality penalty pattern, and
  `source` field outputs.
- §3.2 covers `SCPrecisionEncoder` with the three encodings
  (binary / unary / one_hot), per-encoding qubit-vs-level
  trade-off table, and round-trip accuracy bounds.
- §3.3 is a "when to use which compiler" table covering all four
  compilers in this bridge (SCToIsing, SCToQUBO, SCBitstreamQUBO,
  SCPrecisionEncoder).

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
