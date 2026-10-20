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
| `grid_topology` | `topology.py` | 2-D Chebyshev-radius (square) lattice |
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
- `spike_gating` (Python backend only) skips a neuron only where skipping it
  and stepping it are the same: its input is exactly zero and its whole state
  equals the measured fixed point of its model (`network/quiescence.py`, a
  probe that steps a copy with zero input). Only models whose purity is
  declared are probed: the `sc_neurocore` catalogue models, and a user class
  that sets `quiescence_pure = True` to state that `step` touches nothing but
  its own attributes; any other model is never skipped. A probe that moves the
  NumPy or the standard `random` generator refuses the model and restores both.
  Gated and ungated runs give the same spikes.

The Python loop (`_run_python`) is the reference implementation; the Rust
loop (`_run_rust`) round-trips populations and projections through
`NetworkRunner.add_population` / `add_projection`, runs `n_steps`, syncs
final voltages back into each `Population`, then decodes packed spike events
(`u64 = neuron_id<<32 | timestep`) back into Python `SpikeMonitor` records.

`NetworkRunner.add_projection(src, tgt, row_offsets, col_indices, values,
delay=0)` admits the CSR arrays before allocating anything. Both population
indices must exist. `row_offsets` must have one more entry than the source
population has neurons, start at 0 and never decrease. `col_indices` and
`values` must have equal lengths, and the last offset must not exceed them.
Every used column must index the target population. Any other input raises
`ValueError` naming the first violation, and the runner keeps its topology
and state. Malformed connectivity therefore cannot silently drop a driven
connection or index outside its storage.

### 3.3 Rust dispatch criteria (`_can_use_rust`)

`Network._can_use_rust` returns `True` only when `_rust_incompatibilities()`
is empty and the engine can run every model:

1. No embedded stimulus (`TimedArray`, `PoissonInput`, `StepCurrent`): the
   runner has no stimulus input and would run as if there were none.
2. `spike_gating` is disabled and `fim_lambda == 0.0`.
3. No `StateMonitor` or `RateMonitor` is attached, because those require
   per-step traces the Rust runner does not export yet.
4. Every population is what the runner builds from its model name alone
   (no constructor parameters, derived seeds or moved state;
   `rust_dispatch.population_divergence`).
5. Every model's whole state is `v`: the runner returns only voltages, so a
   model whose descriptor declares more state, or whose default neuron moves
   anything besides `v` when stepped (AdEx's `w`, Rulkov's `x` and `y`, a
   refractory counter, a generator), would keep that state at its initial
   value (`rust_dispatch.evolving_state`).
6. No model draws from the process-wide NumPy or standard `random` generator
   (`StochasticIFNeuron` calls `np.random.randn()`): a native run would not
   advance it, so every later draw of the caller would differ
   (`rust_dispatch.draws_process_random`). The probe behind this check and
   check 5 constructs and steps default neurons inside a generator
   transaction, so checking, a fallback and a refusal leave the caller's
   generators exactly where they were.
7. The model name means the same model on both sides: `LapicqueNeuron` is the
   SC LIF compatibility profile in Python but the 1907 source profile in the
   engine, so it is refused; `SCLapicqueLIFNeuron` is the same on both.
8. No projection has per-synapse delays, a non-zero `weight_threshold` or
   plasticity (`rust_dispatch.projection_divergences`): the runner takes the
   CSR arrays and one uniform delay.
9. The Rust engine import succeeded and every `pop.model_name` (or its
   `*Neuron`-stripped form) is in `NetworkRunner.supported_models()`.

When any of these fails and `backend="auto"`, the network falls back to
Python without warning; a forced `backend="rust"` raises
`NotImplementedError` naming every reason. Pass `backend="python"` explicitly
when you need deterministic dispatch.

### 3.4 Backend matrix

| Feature | Python | Rust | MPI |
|---------|--------|------|-----|
| Heterogeneous neuron models | ✅ any | ⚠️ only `supported_models()` | ✅ per-rank |
| Stimuli | ✅ all | ❌ disqualifies Rust | ✅ rank 0 |
| STDP / plasticity | ✅ | ❌ disqualifies Rust | ✅ |
| Per-synapse delays | ✅ | ❌ disqualifies Rust (uniform delay only) | ✅ |
| `weight_threshold` pruning | ✅ | ❌ disqualifies Rust | ✅ |
| Models with state besides `v` | ✅ | ❌ disqualifies Rust (voltages only) | ✅ |
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

When `spike_gating=True`, `step_all` skips a neuron only where skipping it and
stepping it are the same thing (`network/quiescence.py`): its input is exactly
zero, its model declares a pure step (the catalogue models, or a user class
with `quiescence_pure = True`), and its whole state equals the signature the
model's zero-input step was measured to leave unchanged — exactly, not within
a tolerance. A model with no such fixed point (a pacemaker, an oscillator) is
never skipped. Per-step compute then tends to the **active** neuron count
without the run differing from the same run ungated; the earlier near-rest
heuristic, which froze leaks and countdowns, is gone.

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
| `small_world` | Watts–Strogatz simple graph: even `k < n`, rewired edges avoid self and existing neighbours, `n·k` entries | Watts & Strogatz 1998 | yes (added both directions) |
| `scale_free` | Barabási–Albert: star seed on `m+1` nodes, then `m` distinct targets per node, drawn by actual degree; simple graph | Barabási & Albert 1999 | yes |
| `ring_topology` | Simple ring: `k` nearest neighbours each side, `0 <= k <= (n-1)//2` (no doubled antipode, no self connection) | — | yes |
| `grid_topology` | 2-D lattice within a Chebyshev (square) radius: 8 neighbours at `r=1`, 24 at `r=2` | — | yes |
| `all_to_all` | Dense | — | yes when `n_src == n_tgt` |

`scale_free(n, m, weight, seed=...)` enforces the Barabasi-Albert domain before
allocation or random sampling: `n` and `m` must be non-boolean integers,
`n >= 2`, `1 <= m < n`, and `weight` must be finite. Invalid values raise
`ValueError` before any CSR arrays are emitted. The seed graph is a star on
`m + 1` nodes, so every node that can be drawn has an edge, and each later node
joins `m` distinct earlier nodes with probability proportional to the degree
they actually have in the graph built so far. `ring_topology(n, k, weight)`
refuses a `k` above `(n - 1) // 2`, where the two sides would meet at the
antipode twice or reach the node itself; a multigraph ring is not offered.

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
| `cross_correlation(i, j, max_lag)` | `(corr, lags)` | delegates to `analysis.spike_stats.cross_correlation` at one sample per timestep; `lags` are timesteps `-max_lag..max_lag` |

### 8.2 `StateMonitor`

Captures `population.get_states()` snapshots at each call to `snapshot(t)`.
The values come from each model's `get_state()`, so a model that holds its
state in a mapping (`EquationNeuron`) is recorded as it is, not as zeros.
Configure with `variables: list[str]` (default `["v"]`) and an optional
`record: list[int]` to subset recorded neurons; an empty list records none,
and an index outside the population is refused when the monitor is built.

### 8.3 `RateMonitor`

Bins spike counts into windows of whole steps, then converts to per-neuron
mean rate (`Hz`). A bin holds `max(1, floor(bin_ms / 1000 / dt))` steps, so a
10 ms bin at `dt=1 ms` flushes every 10 steps, and its rate divides by the time
those steps cover: a 10 ms bin at `dt=3 ms` is a 9 ms bin, a 1 ms bin at
`dt=10 ms` a 10 ms one.

### 8.4 Repeated runs

A second `Network.run` continues the first: monitors record the step index
from where the previous run stopped, and the spikes of its last step reach
their targets through the projections on the next step, as in one longer run.
Continuing at another `dt` is refused. A network with spikes or delayed
current still in flight is not dispatched to the native runner, which starts
empty.

---

## 9. Verilog export

`export_verilog(network, output_dir, target="ice40", *, data_width=16, fraction=8) -> str`
writes RTL that elaborates (Icarus, Yosys `hierarchy -check`) and steps as the
Python neurons do. Each population is lowered by the equation compiler to a
neuron module (`sc_net_pop<i>_neuron`, signed fixed point, Q8.8 by default)
from an `EquationNeuron` with the model's own
update, `>=` threshold and hard reset, and `sc_network_top` instantiates one
module per neuron:

- `SCLapicqueLIFNeuron`, or `LapicqueNeuron` in its SC LIF profile: the exact
  flow `v_inf + (v - v_inf)·e^(-dt/tau)` with `v_inf = v_rest + R·I`;
- `StochasticLIFNeuron` with `noise_std == 0` and `refractory_period == 0`:
  forward Euler `v - (v - v_rest)·dt/tau_mem + R·I·dt`.

Every neuron of a population must share its parameters and initial voltage.
Other models, the 1907 Lapicque source profile, noise, refractoriness, an empty
network and an empty population raise `SCHardwareError` before anything is
written. The top has one `data_width`-bit current word (`current_<i>`) and one
spike bit (`spike_<i>`) per neuron.

Before anything is written, every lowered parameter (`decay`, `r`, `leak`,
`gain`, `v_rest`, `v_reset`, `v_threshold`) and the initial voltage are checked
at the format with `hardware_numeric_contract`: a value outside the format's
range, which the encoder would wrap to another value, or a non-zero value that
rounds to zero refuses the export with `SCHardwareError` naming the quantity.
Q8.8 holds `[-128, 127.99609375]`; `v_threshold=300` or a gain `R·dt = 256`
needs a wider format, for example `data_width=32, fraction=16`.

The exporter writes two files:
- `sc_network_top.v` — the neuron modules and the top
- `params.vh` — the data width, fraction and `\`define POP_<i>_SIZE n`

Limitations:
- Projections and topology are not exported, only the neurons
- The format rounds the parameters (`e^(-dt/tau)`, `dt/tau_mem`); spike trains
  equal the model's exactly where the arithmetic is exact
  (`tests/test_network_verilog_export_fidelity.py`)
- Only parameters and the initial state are range-checked; the voltage a run
  reaches is not, and the RTL saturates it
- The Rust, Julia and Mojo files named `export` under `accel/` are unfinished
  skeletons, marked as such, and are not called; export runs in Python only

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
(500 Hz, w=2.0, seed=11), historical SC-profile `LapicqueNeuron` class
populations (default parameters: tau=20 ms, threshold=1.0). This predates the
explicit source/SC routing split and is not a benchmark of the 1907
polarization profile. Hardware: Intel i5-11600K, 32 GB,
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
| `Population(model="LapicqueNeuron", n=...)` | exact string selects the source profile; SC-only parameters preserve legacy compatibility; `SCLapicqueLIFNeuron` is explicit | `tests/test_model_lapicque_source_contract.py` |
| `net.run(backend="rust")` | imports `sc_neurocore_engine.NetworkRunner` lazily | `_get_rust_engine` (network.py:25) |
| `net.run(backend="mpi")` | imports `mpi4py.MPI` lazily, raises if absent | `_require_mpi` (mpi_runner.py:36) |
| `Projection(plasticity="stdp")` | activates `update_plasticity` per timestep | tested in `test_network_basic.py` |
| `export_verilog(net, dir)` | calls `_check_exportable` first | raises `SCHardwareError` for unsupported models and unrepresentable formats; `tests/test_network_verilog_export_fidelity.py` |

Every public symbol terminates either in tested code or in an explicit
runtime check. There are no orphan helpers.

---

## 13. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 18 public symbols wired; backend dispatcher complete |
| 2 | Multi-angle tests | ⚠️ WARN | Network tests cover the orchestrator, monitors, topology, cortical column, gamma circuit, and 12 mocked-mpi4py MPIRunner paths including per-rank Rust dispatch. Real multi-rank coverage is still missing (task #17). `export.py` is covered by `tests/test_network_basic_export.py` and `tests/test_network_verilog_export_fidelity.py` (Icarus simulation against the Python neurons, Yosys elaboration, refusals). |
| 3 | Rust path | ✅ PASS | `Network._run_rust` and the installed `sc_neurocore_engine==3.16.0` wheel are covered by `tests/test_rust_integration.py` plus the WC-E1 execution-path tests. Rust dispatch rejects `StateMonitor`, `RateMonitor`, `spike_gating`, and `fim_lambda` instead of silently skipping those Python-only semantics. `topology.py`, `_csr_matvec`/`_csr_delayed_matvec`, `update_plasticity` remain pure Python — task #13 tracks that separate Rustification gap. |
| 4 | Benchmarks | ✅ PASS | §6.1, §11, §11.1 measured this session. `benchmarks/sc_network_benchmark.py` covers the SC pipeline (encode/MAC/decode); its current [per-repetition report contract](../benchmarks/BENCHMARKS.md#stochastic-network-cli-records) is separate from network orchestration — that gap is now filled by §11. |
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
    tests/test_network_basic_*.py \
    tests/test_network_monitors_stimulus_*.py \
    tests/test_network/ \
    tests/test_torch_bridge_fidelity.py \
    tests/test_network_verilog_export_fidelity.py \
    tests/test_cortical_column_*.py \
    tests/test_gamma_oscillation_*.py \
    tests/test_topology_*.py -q
# 428 passed, 29 skipped in 200.76s  (2026-10-19, this workstation; the skips
# are Go, Julia and Mojo parity cases whose kernels are not built here, and an
# unavailable-backend case that cannot run while the Go backend is built)
```

What the existing tests cover:

- `test_network_basic_*.py` — Network construction, add(), run() with each
  backend dispatch path, all monitor types, stimuli, plasticity flag
- `test_network/test_to_torch_bridge_*.py`, `test_torch_bridge_fidelity.py` —
  Network-to-Torch projection contracts, malformed connectivity rejection, and
  spike trains, resets and weights compared with the Python neurons
- `test_network_basic_export.py`, `test_network_verilog_export_fidelity.py` —
  `export_verilog`: Icarus simulation against the Python neurons, Yosys
  elaboration, format range checks and refusals that write nothing
- `test_network_monitors_stimulus_*.py` — Spike/State/Rate monitor
  determinism, Poisson seed reproducibility, TimedArray clamping
- `test_topology_*.py` — every generator's
  output shape, symmetry where claimed, deterministic seeding
- `test_cortical_column_*.py` — `CorticalColumn` dynamics smoke tests
  (does NOT verify Potjans/Binzegger fidelity — see task #10)
- `test_gamma_oscillation_*.py` — `PINGCircuit` smoke tests (does NOT verify
  spectral peak in 30–80 Hz band — see task #11)

What the existing tests do **not** cover:

- `MPIRunner` real multi-rank semantics — 12 mocked-mpi4py tests exist; real `mpirun -n 2` coverage missing (task #17)
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
