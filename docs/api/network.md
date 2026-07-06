# Network Simulation Engine

**Module:** `sc_neurocore.network` (re-exported from `sc_neurocore.network.__init__`)
**Source:** `src/sc_neurocore/network/` — 12 files, 4065 LOC
**Status (v3.14.0):** core orchestrator + Population/Projection/topology/monitors fully wired; Rust network dispatch and MPI per-rank Rust dispatch are implemented, with Python fallback when the engine wheel is absent; the local `sc_neurocore_engine==3.16.0` wheel was installed and verified for the 2026-07-07 WC-E1 audit closure; `MPIRunner` has 12 mocked-mpi4py tests; topology/projection compute paths are still pure-Python (no Rust path yet).

This page covers the **simulation engine** — the declarative `Network`
container, populations of neurons, sparse `Projection` connectivity (with
delays + STDP), connectivity generators, spike/state/rate monitors, stimulus
sources, the multi-backend dispatcher (auto/python/rust/mpi), and the
LIF-network Verilog exporter. Two specialised circuit models that ship in
the same directory will be documented on their own pages once written
(planned: `api/cortical_column.md`, `api/gamma_oscillation.md`); both are
flagged in §14.1 below for fidelity violations.

---

## 1. Overview

The simulation engine is **declarative**: you instantiate populations and
projections, hand them to a `Network`, and call `.run(duration, dt)`. The
network registers each object by type (`Population`, `Projection`,
`SpikeMonitor`, `StateMonitor`, `RateMonitor`, `TimedArray`, `PoissonInput`,
`StepCurrent`) and runs the appropriate per-step pipeline:

```
each timestep t:
  for each population: zero its current accumulator
  apply stimuli → currents
  apply projections (CSR matvec + delay) → currents
  for each population: step neurons, record spikes
  update plasticity (STDP) on projections that have it
  optional: Fisher Information Metric self-observation feedback
```

`Network.run` selects one of three backends:

- `'python'` — the pure-Python loop above
- `'rust'` — delegates to `sc_neurocore_engine.NetworkRunner` (PyO3) when
  every population's `model_name` is in `NetworkRunner.supported_models()`
  *and* there are no Python-only dynamics
- `'mpi'` — partitions populations round-robin across MPI ranks and
  exchanges spikes via `Allgatherv`

`'auto'` (default) picks `'rust'` when its preconditions hold, otherwise
falls back to `'python'`.

---

## 2. Public Surface

The module re-exports 18 public symbols from
`sc_neurocore.network.__init__`:

| Symbol | Source file | Role |
|--------|-------------|------|
| `Network` | `network.py` | Top-level orchestrator + dispatcher |
| `Population` | `population.py` | Vectorised group of identical neurons |
| `Projection` | `projection.py` | CSR connectivity + delay + STDP |
| `SpikeMonitor` | `monitor.py` | Records (neuron_id, timestep) pairs |
| `StateMonitor` | `monitor.py` | Records state-variable traces |
| `RateMonitor` | `monitor.py` | Population firing rate per bin |
| `TimedArray` | `stimulus.py` | Time-varying scalar stimulus |
| `PoissonInput` | `stimulus.py` | Random Poisson spike trains |
| `StepCurrent` | `stimulus.py` | Rectangular step current |
| `random_connectivity` | `topology.py` | Erdős–Rényi |
| `small_world` | `topology.py` | Watts–Strogatz |
| `scale_free` | `topology.py` | Barabási–Albert |
| `ring_topology` | `topology.py` | Ring with k-NN |
| `grid_topology` | `topology.py` | 2-D Manhattan-radius lattice |
| `all_to_all` | `topology.py` | Dense connectivity |
| `export_verilog` | `export.py` | Multi-population LIF → Verilog |
| `MPIRunner` | `mpi_runner.py` | Distributed simulation runner |
| `HAS_MPI` | `mpi_runner.py` | `bool` — was `mpi4py` importable? |

The submodule `cortical_column` exports `CorticalColumn`; the submodule
`gamma_oscillation` exports `PINGCircuit`. Both are documented separately.

---

## 3. `Network` — orchestrator

### 3.1 Constructor

```python
Network(*objects, seed: int = 42, fim_lambda: float = 0.0)
```

Accepts any number of `Population`, `Projection`, monitor or stimulus objects
positionally. Each is registered into the corresponding internal list by
`isinstance` dispatch; unknown types raise `TypeError`.

`fim_lambda` enables a **Fisher Information Metric self-observation
feedback** (`_apply_fim`, `network.py:249`): each timestep the per-source
weight is pulled toward the population mean of its source spike vector by
`λ_fim · (activity_i − μ) / N`. The mechanism is derived from
scpn-quantum-control NB26-28 (FIM alone synchronises at K=0, λ ≥ 8,
increases Φ by 73 %, and is topology-universal). Set to `0.0` (default) to
disable.

### 3.2 `run`

```python
def run(
    duration: float,
    dt: float = 0.001,
    progress: bool = False,
    backend: str = "auto",
    spike_gating: bool = False,
) -> None
```

- `duration` is in seconds; `dt` is the timestep in seconds. `n_steps` is
  `int(round(duration/dt))`.
- `backend` ∈ `{"auto", "python", "rust", "mpi"}`. `"rust"` raises if the
  engine wheel is not importable or Python-only semantics are present;
  `"auto"` silently falls back to Python.
- `spike_gating` (Python backend only) skips neurons with zero input current
  whose voltage is within 1 % of resting potential. Useful for sparse
  networks where most neurons are silent.

The Python loop (`_run_python`) is the reference implementation; the Rust
loop (`_run_rust`) round-trips populations and projections through
`NetworkRunner.add_population` / `add_projection`, runs `n_steps`, syncs
final voltages back into each `Population`, then decodes packed spike events
(`u64 = neuron_id<<32 | timestep`) back into Python `SpikeMonitor` records.

### 3.3 Rust dispatch criteria (`_can_use_rust`)

`Network._can_use_rust` returns `True` only when:

1. `len(self.stimuli) == 0` — no `TimedArray`/`PoissonInput`/`StepCurrent`
   in this network.
2. `spike_gating` is disabled.
3. `fim_lambda == 0.0`.
4. No `StateMonitor` or `RateMonitor` is attached, because those require
   per-step traces the Rust runner does not export yet.
5. The Rust engine import succeeded (`_get_rust_engine()` is not `False`).
6. Every `pop.model_name` (or its `*Neuron`-stripped form) is in
   `NetworkRunner.supported_models()`.
7. No projection has a non-empty `plasticity` field.

When any of these fails and `backend="auto"`, the network falls back to
Python without warning. Pass `backend="python"` explicitly when you need
deterministic dispatch.

### 3.4 Backend matrix

| Feature | Python | Rust | MPI |
|---------|--------|------|-----|
| Heterogeneous neuron models | ✅ any | ⚠️ only `supported_models()` | ✅ per-rank |
| Stimuli | ✅ all | ❌ disqualifies Rust | ✅ rank 0 |
| STDP / plasticity | ✅ | ❌ disqualifies Rust | ✅ |
| Per-synapse delays | ✅ | ⚠️ uniform only via `add_projection(... max_delay)` | ✅ |
| SpikeMonitor | ✅ per step | ✅ decoded events after run | ✅ rank-local |
| StateMonitor / RateMonitor | ✅ per step | ❌ disqualifies Rust | ⚠️ partial |
| Spike gating | ✅ | ❌ | ❌ |
| FIM feedback (`fim_lambda > 0`) | ✅ | ❌ | ❌ |
| Multi-rank | ❌ | ❌ | ✅ |

---

## 4. `Population` — vectorised neurons

```python
Population(model: type | str, n: int,
           params: dict | None = None,
           label: str | None = None)
```

`model` may be a class or a string name resolved through
`sc_neurocore.neurons.models` (lazy registry of 130 lazy-loaded classes —
see `api/neurons.md`). The constructor instantiates `n` independent neuron
objects with identical parameters and exposes:

- `population.neurons` — `list[NeuronProtocol]` of length `n`
- `population.voltages` — read-only `np.ndarray` view (kept in sync via
  `_sync_voltages`)
- `population.step_all(currents, spike_gating=False)` — returns binary
  spike vector `np.ndarray[int8]` of length `n`
- `population.reset_all()` — calls `reset()` or `reset_state()` on each neuron
- `population.get_states()` — collects all per-neuron state variables into
  arrays (uses `get_state()`, `__dataclass_fields__`, or falls back to
  `["v"]`)
- `population.set_voltages(arr)` — sync voltages from an external source
  (used by the Rust round-trip)

Populations are **hashable by identity** (`id(pop)`); the Network keeps
`pop_to_currents` keyed by `id(pop)` so two `Population` objects with
identical parameters are still independent.

### 4.1 Spike gating

When `spike_gating=True`, `step_all` skips neurons that have no input current
*and* sit within 1 % of their resting potential. This makes per-step compute
roughly proportional to the **active** neuron count, which matters for
sparse networks where most neurons are silent for most of the simulation.
The skipped neurons do not advance — when the population is queried later,
their voltages are stale until they receive input again. Models that track
sub-threshold leak via internal calls to `step()` lose that leak during
skipped steps.

---

## 5. `Projection` — CSR connectivity + delays + STDP

```python
Projection(
    source: Population,
    target: Population,
    weight: float,
    probability: float = 1.0,
    delay: float | np.ndarray = 0.0,
    topology: str | tuple[np.ndarray, np.ndarray, np.ndarray] = "random",
    plasticity: str | None = None,
    seed: int = 42,
    weight_threshold: float = 0.0,
)
```

Stores connectivity in **CSR** (`indptr`, `indices`, `data`). Source spikes
propagate via `_csr_matvec` (no delay) or one of two delayed variants:

- **`delay = 0.0`** — direct CSR matvec, no buffering
- **`delay = scalar`** — uniform axonal delay; output goes through a
  circular buffer of shape `(steps, target.n)` and is read out
  one timestep behind
- **`delay = ndarray`** of length `n_synapses` — per-synapse delay;
  source spike history is stored as a ring buffer of shape
  `(max_delay+1, source.n)` and each synapse reads from
  `spike_history[(hist_idx − d_k) % max_delay]`

Per-synapse delays implement Hammouamri et al. 2023 (DCLS) and Masquelier's
DelRec — learnable synaptic delays in spiking nets.

### 5.1 `weight_threshold` pruning

When `weight_threshold > 0.0`, the matvec skips synapses with
`|data[k]| ≤ weight_threshold`. Useful after sparse pruning to avoid wasted
multiplications. The branch is in the inner Python loop (`projection.py:47`)
so the speed-up is real but linear in connection count.

### 5.2 STDP plasticity

`plasticity="stdp"` activates trace-based STDP in `update_plasticity`
(`projection.py:258`):

- Pre/post traces decay with `tau` (default 20 timesteps), incremented on
  spike
- LTD on pre-spike: `data[k] -= a_minus * post_trace[j]`
- LTP on post-spike: `data[k] += a_plus * directional_bias * pre_trace[i]`
- Weights clipped at 0 (no sign change)

`directional_bias` scales `a_plus` per projection. The scpn-quantum-control
NB19 measurement of autonomic → cortical asymmetry suggests **1.36** for
bottom-up (sensory → higher-order) projections; default 1.0 keeps learning
symmetric.

Self-projections (`source is target`) additionally trigger
`_enforce_symmetry` after each STDP update — `W_ij` and `W_ji` are averaged.
Required because gradient/STDP updates break `W = W^T` after ~30 steps
(SPO Finding #7); asymmetric coupling hurts synchronisation by +12 %
(quantum-control NB24).

---

## 6. Topology generators

All generators in `topology.py` return a `(indptr, indices, data)` CSR
tuple ready to feed into `Projection(..., topology=tuple)`.

| Generator | Algorithm | Cited basis | Symmetric? |
|-----------|-----------|-------------|------------|
| `random_connectivity` | Erdős–Rényi | classic | no |
| `small_world` | Watts–Strogatz | Watts & Strogatz 1998 | yes (added both directions) |
| `scale_free` | Barabási–Albert preferential attachment | Barabási & Albert 1999 | yes |
| `ring_topology` | Ring + k nearest neighbours both directions | — | yes |
| `grid_topology` | 2-D lattice within Manhattan radius | — | no (loops add both directions only when within radius) |
| `all_to_all` | Dense | — | yes when `n_src == n_tgt` |

`scale_free(n, m, weight, seed=...)` enforces the Barabasi-Albert domain before
allocation or random sampling: `n` and `m` must be non-boolean integers,
`n >= 2`, `1 <= m < n`, and `weight` must be finite. Invalid values raise
`ValueError` before any CSR arrays are emitted.

### 6.1 Performance (this workstation, 2026-04-17)

Measured directly by calling each generator and timing with
`time.perf_counter()`. All inputs are deterministic seeds.

| Generator | n | args | wall (ms) | synapses |
|-----------|---|------|-----------|----------|
| `random_connectivity` | 200 | `p=0.1` | 32.5 | 3 941 |
| `all_to_all` | 200 | — | 11.7 | 40 000 |
| `small_world` | 200 | `k=8, p=0.1` | 1.6 | 1 600 |
| `scale_free` | 200 | `m=4` | 10.0 | 1 568 |
| `ring_topology` | 200 | `k=4` | 0.5 | 1 600 |
| `grid_topology` | 196 (14²) | `r=1` | <0.01 | 1 404 |
| `random_connectivity` | 1 000 | `p=0.1` | 66.3 | 99 869 |
| `all_to_all` | 1 000 | — | 472.2 | 1 000 000 |
| `small_world` | 1 000 | `k=8, p=0.1` | 26.0 | 8 000 |
| `scale_free` | 1 000 | `m=4` | 67.8 | 7 968 |
| `ring_topology` | 1 000 | `k=4` | 3.8 | 8 000 |
| `grid_topology` | 961 (31²) | `r=1` | <0.01 | 7 320 |

`small_world` and `scale_free` use Python lists during construction; for
n ≥ 10 000 they become noticeably slow and are candidates for the planned
Rust path (task #13). `grid_topology` and `ring_topology` already vectorise
adequately for typical sizes.

---

## 7. Stimulus sources

| Class | Returns from `get_current` | Notes |
|-------|----------------------------|-------|
| `TimedArray(values, dt)` | scalar `float` at `min(t_step, len-1)` | clamps past the end |
| `PoissonInput(n, rate_hz, weight, dt, seed)` | `np.ndarray[n]` | `(rng.random < rate_hz·dt)·weight` |
| `StepCurrent(onset, offset, amplitude)` | scalar `float` if `onset ≤ t < offset` else `0.0` | rectangular |

Each stimulus carries a `target: Population | None` slot that the network
populates when adding the stimulus to a population (the convenience pattern
is to set `stim.target = pop` after construction). When `target is None`,
the network broadcasts to `populations[0]`.

`PoissonInput` owns a `np.random.default_rng(seed)`; runs are deterministic
when seeds are pinned. `TimedArray` and `StepCurrent` are deterministic by
construction.

---

## 8. Monitors

### 8.1 `SpikeMonitor`

Records `(neuron_id, timestep)` pairs. Two ingestion paths:

- `record(spikes, t)` — accepts the binary spike vector emitted by
  `Population.step_all` (Python backend)
- `record_event(neuron_id, t)` — accepts a single decoded event (used by
  the Rust round-trip, which packs `u64 = neuron_id<<32 | timestep` for
  efficient transport)

Read-out helpers:

| Property/method | Returns | Notes |
|-----------------|---------|-------|
| `spike_times` | `np.ndarray[int64]` | every spike's timestep |
| `spike_trains` | `dict[int, np.ndarray]` | per-neuron timestep arrays |
| `count` | `int` | total spikes recorded |
| `raster_data()` | `(times, neuron_ids)` | tuple ready for raster plot |
| `firing_rates(n_steps, dt)` | `np.ndarray[n]` | mean Hz per neuron |
| `isi(neuron)` | `np.ndarray[int64]` | inter-spike intervals (timesteps) |
| `cross_correlation(i, j, max_lag)` | `(corr, lags)` | delegates to `analysis.spike_stats.cross_correlation` |

### 8.2 `StateMonitor`

Captures `population.get_states()` snapshots at each call to `snapshot(t)`.
Configure with `variables: list[str]` (default `["v"]`) and an optional
`record: list[int]` to subset recorded neurons.

### 8.3 `RateMonitor`

Bins spike counts into fixed-duration windows (`bin_ms`), then converts
to per-neuron mean rate (`Hz`). The bin-completion check uses
`steps_per_bin = max(1, int(bin_ms / 1000.0 / dt))`, so a 10 ms bin at
`dt=1 ms` flushes every 10 steps.

---

## 9. Verilog export

`export_verilog(network, output_dir, target="ice40") -> str` emits a
top-level `sc_network_top` SystemVerilog module that wires one
`sc_lif_array` instance per population. Each population's parameters are
read from the **first** neuron in `pop.neurons` (`v_threshold`, `v_reset`,
`tau`) and converted to fixed-point by multiplying by 256 (Q8.8). The
allowed neuron model whitelist is `_LIF_MODELS` (17 LIF variants); any
non-LIF model raises `SCHardwareError`.

The exporter writes two files:
- `sc_network_top.v` — module wrapper with one `pop_<i>` instance per
  population
- `params.vh` — `\`define POP_<i>_SIZE n` per population

Limitations:
- Only supports populations of LIF-family models
- Does not emit projections or topology — only neuron arrays
- Hard-codes Q8.8 scaling (see `cli.md` §9.1 for the related `dt=0.001`
  underflow bug)
- Not test-covered in `tests/test_network_*.py`; add coverage when the LIF
  network export is exercised end-to-end

For full equation → Verilog compilation including state-machine generation
and FPGA project files, use `sc-neurocore compile` (see `api/cli.md`) or
`api/compiler.md`.

---

## 10. `MPIRunner` (distributed)

`mpi_runner.py` provides round-robin partitioning of populations across MPI
ranks with `MPI_Allgatherv` spike exchange every timestep:

```
each rank r:
  local_pops = {i for i in range(n_pops) if i % size == r}
  for t in range(n_steps):
    propagate local + cross-rank projections into local_currents
    step local populations → local_spikes
    Allgatherv local_spikes → all_spikes
    if r == 0: feed monitors with all_spikes
```

`mpi_runner.py:79` (`_identify_cross_rank_projections`) walks every
projection and tags it as **local** (source and target on the same rank) or
**cross-rank**. The implementation works with both Python and Rust
per-rank stepping, but currently dispatches via `Population.step_all`
(Python loop) regardless of backend selection.

### 10.1 Status

- 191 LOC, including a custom spike packing protocol
  (`pop_index | n | spike_data` packed as `int8` blob) and `Allgather` +
  `Allgatherv` choreography.
- `tests/test_mpi_runner.py` (177 lines, 8 tests) covers MPIRunner via
  mocked `mpi4py` — partition correctness, RuntimeError when mpi4py is
  absent, single-rank end-to-end equivalence with the Python backend,
  cross-rank vs local projection routing, spike-exchange round-trip.
  **Real multi-rank semantics with `mpirun -n 2` are NOT exercised**;
  task #17 tracks adding a `pytest-mpi`-style real test.
- Does not implement spike gating, FIM feedback, or per-rank
  Rust dispatch.

---

## 11. Performance — Python backend (this workstation)

Measured via:

```python
net = Network(pop, proj, mon, stim, seed=1)
net.run(duration=0.2, dt=0.001, backend="python")
```

with 200 timesteps, recurrent random connectivity at p=0.2, Poisson input
(500 Hz, w=2.0, seed=11), `LapicqueNeuron` populations (default
parameters: tau=20 ms, threshold=1.0). Hardware: Intel i5-11600K, 32 GB,
Python 3.12.3.

| Population n | Synapses | 200-step wall | steps / s | Recorded spikes |
|--------------|---------:|--------------:|----------:|---------------:|
| 50 | 489 | 13.9 ms | 14 428 | 185 |
| 200 | 7 911 | 56.6 ms | 3 532 | 782 |
| 500 | 49 570 | 193.8 ms | 1 032 | 2 385 |
| 1 000 | 200 283 | 854.8 ms | 234 | 7 464 |

Scaling is roughly `O(n_synapses)` because the Python `_csr_matvec` inner
loop dominates. `step_all` walks the population list one neuron at a time
(line 78: `neuron.step(float(currents[i]))`) which adds ~5 µs of Python
overhead per neuron per step. For dense recurrent networks at n ≥ 1000 the
Python backend rapidly becomes the bottleneck — the Rust backend exists for
exactly this case but requires the engine wheel installed (see §13).

### 11.1 Delay-mode cost (n=200, p=0.2, 200 steps)

| Mode | `max_delay` | wall (ms) |
|------|-------------|-----------|
| `none` | 0 | 114.0 |
| `uniform` (5 steps) | 5 | 113.1 |
| `per_synapse` (rand 1–7) | 7 | 637.0 |

Per-synapse delays are ~5.6× slower than no-delay or uniform-delay because
`_csr_delayed_matvec` walks every synapse without the early-exit
`if x[i] == 0: continue` of `_csr_matvec`. (`_csr_delayed_matvec` reads
spike history at varying offsets, so it cannot skip rows up front.) Use
uniform delay when the biological detail tolerates it.

---

## 12. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.network import Network, Population, ...` | `__init__.py` re-exports 18 symbols | `tests/test_network_basic.py` |
| `Population(model="LapicqueNeuron", n=...)` | resolves through `sc_neurocore.neurons.models._CLASS_TO_MODULE` | `Population._resolve_model` (population.py:19) |
| `net.run(backend="rust")` | imports `sc_neurocore_engine.NetworkRunner` lazily | `_get_rust_engine` (network.py:25) |
| `net.run(backend="mpi")` | imports `mpi4py.MPI` lazily, raises if absent | `_require_mpi` (mpi_runner.py:36) |
| `Projection(plasticity="stdp")` | activates `update_plasticity` per timestep | tested in `test_network_basic.py` |
| `export_verilog(net, dir)` | calls `_check_exportable` first | raises `SCHardwareError` for non-LIF models |

Every public symbol terminates either in tested code or in an explicit
runtime check. There are no orphan helpers.

---

## 13. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 18 public symbols wired; backend dispatcher complete |
| 2 | Multi-angle tests | ⚠️ WARN | Network tests cover the orchestrator, monitors, topology, cortical column, gamma circuit, and 12 mocked-mpi4py MPIRunner paths including per-rank Rust dispatch. Real multi-rank coverage is still missing (task #17). `export.py` is not directly covered. |
| 3 | Rust path | ✅ PASS | `Network._run_rust` and the installed `sc_neurocore_engine==3.16.0` wheel are covered by `tests/test_rust_integration.py` plus the WC-E1 execution-path tests. Rust dispatch rejects `StateMonitor`, `RateMonitor`, `spike_gating`, and `fim_lambda` instead of silently skipping those Python-only semantics. `topology.py`, `_csr_matvec`/`_csr_delayed_matvec`, `update_plasticity` remain pure Python — task #13 tracks that separate Rustification gap. |
| 4 | Benchmarks | ✅ PASS | §6.1, §11, §11.1 measured this session. `benchmarks/sc_network_benchmark.py` exists (306 lines) but covers SC pipeline (encode/MAC/decode), not network orchestration — that gap is now filled by §11. |
| 5 | Performance docs | ✅ PASS | §11 + §6.1 + §11.1 |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ⚠️ WARN | SPDX headers on every source file ✅. `gamma_oscillation.py:66-67` has `# type: ignore[arg-type]` without rationale (mirrors `cli.py:298`). British English in docstrings is mixed (`vectorized`, `synchronization` appear); see §14. |

Net: **3 WARN, 0 FAIL.** Tracked follow-ups: tasks #10–#13.

---

## 14. Known issues & follow-ups

### 14.1 Two model fidelity violations (CRITICAL)

`CorticalColumn` and `PINGCircuit` ship in this directory but **simplify
their cited publications** in ways that break the no-simplifications rule:

- `CorticalColumn` cites Douglas & Martin 2004 + Potjans & Diesmann 2014;
  implements 5 of 8 populations (no L4i/L5i/L6i), 7 of 64 connections from
  the Binzegger matrix, no PSP kernel, no Poisson background input.
  Tracked: task #10.
- `PINGCircuit` cites Whittington et al. 1995 + Börgers & Kopell 2003;
  implements a mean-field rate approximation (population firing rate ×
  scalar weight) instead of the spiking conductance-based PING with
  α-function synapses. Tracked: task #11.

Both will be documented in detail on their own pages once written
(`api/cortical_column.md`, `api/gamma_oscillation.md`); the audit findings
above are the canonical source until then.

### 14.2 Rustification gap

Topology generators and projection matvec / STDP are pure-Python loops.
For n ≥ 1000 dense or any per-synapse-delay setup, this is the dominant
cost. The Rust engine has `NetworkRunner` for the network loop but no
counterpart for the topology generators, and the `add_projection` call
takes a Python-side CSR tuple. Task #13 tracks closing this gap.

### 14.3 MPIRunner real multi-rank coverage missing

`tests/test_mpi_runner.py` covers 12 paths via mocked `mpi4py`, including
the `NetworkRunner.step_population` per-rank Rust dispatch contract. The
custom spike-packing protocol and `Allgatherv` choreography are not
exercised against real mpi4py + `mpirun -n 2`; a regression in real-MPI
buffer ordering or datatype matching would not be caught. Task #17
tracks adding a `pytest-mpi`-style real test.

### 14.4 American spellings in source docstrings

Docstrings in `network.py`, `population.py`, `projection.py` use
`vectorized`, `synchronization`, `optimize` etc. — should be British per
SHARED_CONTEXT.md. Not blocking; future cleanup.

### 14.5 `# type: ignore[arg-type]` without rationale

`gamma_oscillation.py:66-67` (dataclass field defaults). Mirror of
`cli.py:298`. Should either type-correctly or annotate the reason.

---

## 15. Tests

```bash
PYTHONPATH=src python3 -m pytest \
    tests/test_network_basic.py \
    tests/test_network_monitors_stimulus.py \
    tests/test_network/test_to_torch_bridge.py \
    tests/test_cortical_column.py \
    tests/test_cortical_column_dynamics.py \
    tests/test_gamma_oscillation.py \
    tests/test_topology.py \
    tests/test_topology_generators.py -q
# 87 passed in 2.26s  (verified 2026-04-17)
```

What the existing tests cover:

- `test_network_basic.py` — Network construction, add(), run() with each
  backend dispatch path, all monitor types, stimuli, plasticity flag
- `test_network/test_to_torch_bridge.py` — Network-to-Torch projection
  contracts, malformed connectivity rejection, and bridge parity checks
- `test_network_monitors_stimulus.py` — Spike/State/Rate monitor
  determinism, Poisson seed reproducibility, TimedArray clamping
- `test_topology.py`, `test_topology_generators.py` — every generator's
  output shape, symmetry where claimed, deterministic seeding
- `test_cortical_column*.py` — `CorticalColumn` dynamics smoke tests
  (does NOT verify Potjans/Binzegger fidelity — see task #10)
- `test_gamma_oscillation.py` — `PINGCircuit` smoke tests (does NOT verify
  spectral peak in 30–80 Hz band — see task #11)

What the existing tests do **not** cover:

- `MPIRunner` real multi-rank semantics — 12 mocked-mpi4py tests exist; real `mpirun -n 2` coverage missing (task #17)
- `export_verilog` — no direct test; covered transitively by FPGA flow
  smoke tests at most
- Performance regressions — no `pytest-benchmark` cases for the network
  loop; §11 numbers are point measurements

---

## 16. References

Network simulation engineering:

- Brette R., Rudolph M. *et al.* "Simulation of networks of spiking
  neurons: a review of tools and strategies." *J Comp Neurosci* 23:349-398
  (2007).
- Eppler J. M. *et al.* "PyNEST: A convenient interface to the NEST
  simulator." *Front Neuroinform* 2:12 (2008).

Connectivity models:

- Watts D. J., Strogatz S. H. "Collective dynamics of small-world
  networks." *Nature* 393:440-442 (1998).
- Barabási A.-L., Albert R. "Emergence of scaling in random networks."
  *Science* 286:509-512 (1999).
- Erdős P., Rényi A. "On random graphs." *Publicationes Mathematicae*
  6:290-297 (1959).

Synaptic delays / plasticity:

- Hammouamri I. *et al.* "Learning delays in spiking neural networks
  using dilated convolutions with learnable spacings." *NeurIPS* (2023).
- Bi G., Poo M. "Synaptic modifications in cultured hippocampal neurons:
  dependence on spike timing, synaptic strength, and postsynaptic cell
  type." *J Neurosci* 18:10464-10472 (1998).

MPI:

- Message Passing Interface Forum. *MPI: A Message-Passing Interface
  Standard, Version 4.0* (2021).

Internal:

- CLI: [`api/cli.md`](cli.md)
- Neuron registry: [`api/neurons.md`](neurons.md), [`api/neuron_models.md`](neuron_models.md)
- Compiler: [`api/compiler.md`](compiler.md)
- Cortical column model + audit: planned `api/cortical_column.md`
- Gamma oscillation model + audit: planned `api/gamma_oscillation.md`

---

## 17. Auto-rendered API

::: sc_neurocore.network
    options:
      show_root_heading: true
      show_source: true
      members:
        - Network
        - Population
        - Projection
        - SpikeMonitor
        - StateMonitor
        - RateMonitor
        - TimedArray
        - PoissonInput
        - StepCurrent
        - random_connectivity
        - small_world
        - scale_free
        - ring_topology
        - grid_topology
        - all_to_all
        - export_verilog
        - MPIRunner
        - HAS_MPI
