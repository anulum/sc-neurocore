# MPI Distributed Runner

**Module:** `sc_neurocore.network.mpi_runner`
**Source:** `src/sc_neurocore/network/mpi_runner.py` — 191 LOC
**Status (v3.14.0):** code path complete; 8 mocked-mpi4py tests pass
(2 added by task #19 verifying the new fail-fast guards); real
multi-rank semantics with `mpirun -n 2` are not yet exercised (task #17).
Spike gating and FIM feedback are now refused by the dispatcher with
`NotImplementedError` (was: silently ignored). Per-rank Rust dispatch
is not implemented in this runner.

`MPIRunner` provides distributed execution of a `Network` across MPI ranks.
It partitions populations round-robin, identifies cross-rank projections,
and exchanges spike vectors every timestep via `Allgatherv`. The orchestrator
that selects this runner lives in [`api/network.md`](network.md) — invoke
with `Network.run(backend="mpi")`.

---

## 1. When to use it

Use the MPI backend when:

- A single Network exceeds the memory budget of one process (≳ 10⁵ neurons
  with dense connectivity).
- Multiple compute nodes are available and connected by MPI (a workstation
  cluster, an HPC partition, or a single host with many cores).
- The workload is **not** a candidate for the Rust backend
  (`Network._can_use_rust` returns `False`) — for example because there
  are stimuli, plasticity, or non-supported neuron models.

Do **not** use MPI when:

- The Rust backend is applicable — single-process Rust is far cheaper than
  multi-process Python with cross-rank communication.
- Spike gating or FIM feedback (`fim_lambda > 0`) is needed — neither is
  implemented in `MPIRunner` today; `Network._run_mpi` raises
  `NotImplementedError` rather than silently ignoring the flag.
- Per-rank Rust acceleration is needed — `MPIRunner._step_local` always
  calls `Population.step_all`, which is the Python loop.

---

## 2. Quick example

`MPIRunner` is selected by passing `backend="mpi"` to `Network.run`:

```python
# script.py
from sc_neurocore.network import Network, Population, Projection

pop_a = Population("LapicqueNeuron", n=2_000, label="A")
pop_b = Population("LapicqueNeuron", n=2_000, label="B")
proj = Projection(pop_a, pop_b, weight=0.05, probability=0.05, seed=7)

net = Network(pop_a, pop_b, proj, seed=1)
net.run(duration=1.0, dt=0.001, backend="mpi")
```

Run with two ranks:

```bash
pip install mpi4py        # required dependency
mpirun -n 2 python script.py
```

When `mpi4py` is absent, `Network.run(backend="mpi")` raises
`RuntimeError("mpi4py is required for MPI backend: pip install mpi4py")`
without attempting any partitioning.

---

## 3. Architecture

```
                       ┌─────────────────────────┐
                       │ Network.run(backend=    │
                       │   "mpi", duration, dt)  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Network._run_mpi        │
                       │  n_steps = round(d/dt)  │
                       │  MPIRunner(self).run(   │
                       │     n_steps, dt)        │
                       └────────────┬────────────┘
                                    │
   ┌────────────────────────────────┼────────────────────────────────┐
   │  per rank (constructor)                                          │
   │                                                                  │
   │  comm = MPI.COMM_WORLD; rank = ...; size = ...                   │
   │  _partition_populations()         # round-robin pop → rank       │
   │  _identify_cross_rank_projections()  # local vs cross-rank       │
   └────────────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  per rank, per timestep                                          │
   │                                                                  │
   │  pop_to_currents = {idx: zeros(n) for idx in local_indices}      │
   │  for proj in local_projs:                                        │
   │      pop_to_currents[proj.target_idx] += proj.propagate(spikes)  │
   │  for src_idx, proj in cross_rank_projs:                          │
   │      pop_to_currents[proj.target_idx] += proj.propagate(spikes)  │
   │  local_spikes = _step_local(pop_to_currents, all_spikes)         │
   │  all_spikes = _exchange_spikes(local_spikes)   # Allgatherv      │
   │  if rank == 0: monitors.record(all_spikes, t)                    │
   └─────────────────────────────────────────────────────────────────┘
```

Three responsibilities sit inside the runner:

1. **Partition** — decide which rank owns each population.
2. **Route projections** — every projection is either *local* (source and
   target on the same rank, no MPI traffic) or *cross-rank* (the rank
   needs the source's spikes to compute the target's input current; the
   spikes arrive via the per-step Allgatherv).
3. **Exchange spikes** — pack the local population's spikes into a
   contiguous buffer, allgather the buffer sizes, allgather the buffers
   themselves, unpack back into a per-population dict.

Monitoring is centralised on rank 0 — only rank 0 calls
`SpikeMonitor.record` and `RateMonitor.record`. `StateMonitor` is not
currently driven by this runner.

---

## 4. Round-robin partitioning

`MPIRunner._partition_populations` (`mpi_runner.py:71`) walks the
network's population list and assigns each population to rank
`i % size`:

```python
for i in range(len(self._populations)):
    owner = i % self.size
    self._rank_of[i] = owner
    if owner == self.rank:
        self._local_indices.append(i)
```

Properties of this scheme:

- **Deterministic** — every rank computes the same mapping
  independently; no broadcast needed.
- **Order-sensitive** — the order populations were added to the network
  is the partition order. Re-ordering changes the partition.
- **Not load-balanced by neuron count** — a 5-rank job with one
  100k-neuron population and four 1k-neuron populations dumps the bulk
  of the work onto whichever rank gets `i=0`. Future enhancement
  candidate: weight by `pop.n` and use a balanced bin-packing assignment.
- **Not topology-aware** — a projection that crosses ranks pays the
  Allgatherv cost regardless of how heavy the source population is.

For ≤ 8 same-sized populations on ≤ 8 ranks, round-robin is adequate.
Beyond that, document the imbalance in your run script.

---

## 5. Projection routing

`_identify_cross_rank_projections` (`mpi_runner.py:79`) walks every
`Projection` and tags it as **local** (source and target on the same
rank, no MPI needed) or **cross-rank** (source spikes must arrive via
the global allgather):

```python
src_rank = self._rank_of.get(src_idx, -1)
tgt_rank = self._rank_of.get(tgt_idx, -1)
if src_rank != tgt_rank:
    self._cross_rank_projs.append((src_idx, proj))
else:
    if tgt_rank == self.rank:
        self._local_projs.append(proj)
```

Two consequences:

- A cross-rank projection is processed by the rank that **owns the
  target** (it computes `proj.propagate(src_spikes)` into its local
  current accumulator).
- A local projection is processed only by the rank that owns both
  endpoints — other ranks skip it entirely.

Both `_local_projs` and `_cross_rank_projs` are stored as plain lists; the
per-step inner loop walks each list once. There is no batching.

---

## 6. `_exchange_spikes` packing protocol

The spike-exchange path (`_exchange_spikes`, `mpi_runner.py:93`) is the
hot path of the runner — it executes once per timestep on every rank. The
protocol packs heterogeneous spike vectors (one per local population) into
a single `int8` blob, then uses two collectives to deliver every rank's
blob to every other rank.

### 6.1 Per-rank packing

For each local population index `idx` the rank prepends a 2-element
header (`int32 idx`, `int32 n`), then appends the spike bytes:

```
chunks = []
for idx in local_indices:
    spikes = local_spikes.get(idx, zeros(pop[idx].n, dtype=int8))
    header = array([idx, n], dtype=int32).view(int8)
    chunks.append(header)
    chunks.append(spikes)
send_buf = concatenate(chunks)        # int8 blob
```

The header is reinterpreted as 8 raw bytes via `.view(int8)` so the whole
buffer can be a contiguous `int8` array — `MPI.BYTE` is the on-wire type.
This sidesteps `mpi4py`'s typed-buffer machinery, at the cost of relying
on system endianness (little-endian on the GOTM workstation; relevant if
ever run cross-architecture).

### 6.2 Two-stage collective

```
send_count = int32(send_buf.size)
recv_counts = empty(size, dtype=int32)
comm.Allgather(send_count, recv_counts)        # who is sending how much

total = recv_counts.sum()
recv_buf = empty(total, dtype=int8)
displacements = cumsum(recv_counts) - recv_counts
comm.Allgatherv(send_buf, [recv_buf, recv_counts, displacements, MPI.BYTE])
```

`Allgather` distributes the sizes; `Allgatherv` distributes the variable-
length payloads. After the second collective, every rank has every other
rank's full packed blob.

### 6.3 Unpacking

```
all_spikes = {}
pos = 0
while pos < total:
    header = recv_buf[pos : pos + 8].view(int32)
    pop_idx = int(header[0])
    n = int(header[1])
    pos += 8
    all_spikes[pop_idx] = recv_buf[pos : pos + n].copy()
    pos += n
```

The `.copy()` decouples the returned arrays from the receive buffer (which
is reused next timestep). Skipping the copy would be faster but requires
the caller to consume `all_spikes` before the next `_exchange_spikes`
call — currently no caller takes that contract.

### 6.4 Per-step bandwidth

For `R` ranks each owning `P / R` populations of mean size `N̄`:

- Per-rank send: `(P/R) × (8 + N̄)` bytes
- Per-rank receive: `P × (8 + N̄)` bytes
- Per-step total network volume: `R × P × (8 + N̄)` bytes (each rank
  sends to every other rank via Allgatherv)

For `P = 16`, `N̄ = 5 000`, `R = 4`: 4 × 16 × 5008 ≈ 320 KB / step. At
1 ms timestep this is 320 MB / s aggregate — well under typical InfiniBand
bandwidth (≥ 10 GB/s) but worth noting on slower interconnects.

---

## 7. Run loop (`run`)

`MPIRunner.run(n_steps, dt=0.001)` is the per-rank simulation loop
(`mpi_runner.py:146`). Pseudocode:

```python
np.random.seed(network.seed + rank)         # deterministic per-rank RNG
all_spikes = {i: zeros(pop.n) for i, pop in enumerate(populations)}

for t in range(n_steps):
    # 1. Reset local current accumulators
    pop_to_currents = {idx: zeros(pop[idx].n) for idx in local_indices}

    # 2. Local projections (no MPI)
    for proj in local_projs:
        pop_to_currents[proj.target_idx] += proj.propagate(
            all_spikes[src_idx_of_proj]
        )

    # 3. Cross-rank projections (target rank does the work using
    #    spikes from previous step's Allgather)
    for src_idx, proj in cross_rank_projs:
        if proj.target_idx in pop_to_currents:
            pop_to_currents[proj.target_idx] += proj.propagate(
                all_spikes[src_idx]
            )

    # 4. Step local populations
    local_spikes = _step_local(pop_to_currents, all_spikes)

    # 5. Allgatherv local_spikes → all_spikes (every rank sees everything)
    all_spikes = _exchange_spikes(local_spikes)

    # 6. Rank 0 only: feed monitors
    if rank == 0:
        for mon in network.spike_monitors:
            idx = pop_id_to_idx[id(mon.population)]
            if idx is not None:
                mon.record(all_spikes[idx], t)
        for mon in network.rate_monitors:
            ...
```

Key invariants:

- **Spike causality** — cross-rank projections see source spikes from the
  *previous* step's exchange (steps 5/6 happen after current-step
  propagation in step 3). This matches the Python backend's invariant:
  `last_spikes` from step `t-1` drive currents at step `t`.
- **RNG independence** — each rank seeds its NumPy global RNG with
  `network.seed + rank` so stochastic stimuli (Poisson) and weight
  initialisation differ per rank in a deterministic way.
- **Monitor centralisation** — only rank 0 records, so monitor data is
  available only on rank 0. Other ranks must broadcast or write per-rank
  files if their data matters.

---

## 8. `HAS_MPI` and graceful absence

The module performs the `mpi4py` import at top-level inside a `try`:

```python
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    MPI = None
    HAS_MPI = False
```

This means:

- `from sc_neurocore.network.mpi_runner import MPIRunner` always
  succeeds, regardless of `mpi4py` availability.
- `_require_mpi()` (called from `MPIRunner.__init__`) raises
  `RuntimeError` with the install hint if `HAS_MPI is False`.
- The downstream `Network.run(backend="mpi")` path therefore fails fast
  at runner construction, not lazily during the first collective call.

This pattern keeps imports cheap for users who never enable the MPI
backend (most users), and it lets the test suite exercise the runner
with a fully mocked `MPI` namespace (see §10).

---

## 9. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `Network.run(backend="mpi")` | dispatches to `Network._run_mpi` (network.py:116) | `tests/test_mpi_runner.py::test_single_rank_matches_python` |
| `Network._run_mpi` | constructs `MPIRunner(self)` and calls `.run(n_steps, dt)` | (transitive) |
| `MPIRunner.__init__` | `_require_mpi`, then partition + projection routing | `test_partition_populations`, `test_cross_rank_projection_identification` |
| `_require_mpi` | raises if `HAS_MPI is False` | `test_require_mpi_raises_without_mpi` |
| `_exchange_spikes` | header pack + `Allgather` + `Allgatherv` + unpack | `test_exchange_spikes_mock` |
| Monitor recording | rank 0 only, after each step | (covered indirectly by single-rank end-to-end test) |

There are no orphan helpers — every defined method is reachable from
`MPIRunner.run`.

---

## 10. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_mpi_runner.py -v
# 8 passed in 0.85s (verified 2026-04-17)
```

Test coverage:

| Test | What it checks |
|------|----------------|
| `test_import_without_mpi` | `HAS_MPI` is a bool regardless of `mpi4py` presence |
| `test_require_mpi_raises_without_mpi` | RuntimeError raised when patched to no-mpi |
| `test_partition_populations` | round-robin assignment for size=2 |
| `test_single_rank_matches_python` | end-to-end MPI run (size=1, mocked Allgather) produces same spike count as `backend="python"` |
| `test_cross_rank_projection_identification` | projection from rank-0-owned A to rank-1-owned B is classified as cross-rank |
| `test_exchange_spikes_mock` | header pack + unpack round-trip preserves both populations' spike vectors |
| `test_run_mpi_raises_on_spike_gating` | `Network.run(backend="mpi", spike_gating=True)` raises NotImplementedError |
| `test_run_mpi_raises_on_fim_lambda` | `Network.run(backend="mpi")` on a network with `fim_lambda > 0` raises NotImplementedError |

All eight tests **mock** `mpi4py` via `unittest.mock.MagicMock` and
`patch("sc_neurocore.network.mpi_runner.MPI", ...)`. This validates the
code paths but does not exercise:

- Real `MPI.COMM_WORLD.Allgather` semantics (buffer ordering, datatype
  matching across ranks).
- Multi-rank correctness (`size > 1` with real ranks).
- Latency / throughput behaviour on a real interconnect.
- Failure modes (rank death, out-of-order delivery, flow control).

A `pytest-mpi`-style real test invoked via `mpirun -n 2 pytest ...` is
tracked as task #17.

---

## 11. Performance — not measured

This page does not include measured numbers because:

- `mpi4py` is **not installed** in this environment (`pip show mpi4py`
  reports no package).
- A meaningful MPI benchmark requires at least two ranks; mocked tests
  exercise code paths but not network performance.

When `mpi4py` is added, repeat the §11 measurement protocol from
`api/network.md` with `backend="mpi"` on one rank (sanity check), then
two ranks, then four. Capture: per-step wall, Allgatherv bytes/step,
load-balance ratio (slowest rank wall ÷ fastest rank wall).

---

## 12. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | `Network._run_mpi` → `MPIRunner.run` complete; every public method reachable |
| 2 | Multi-angle tests | ⚠️ WARN | 8 mocked-mpi4py tests pass (6 original + 2 fail-fast guards added by task #19); real multi-rank not exercised (task #17) |
| 3 | Rust path | ⚠️ WARN | `_step_local` always calls `Population.step_all` (Python loop) — per-rank Rust dispatch is not implemented |
| 4 | Benchmarks | ❌ FAIL | None — `mpi4py` absent in this env. Document as gap rather than fabricate numbers (§11) |
| 5 | Performance docs | ⚠️ WARN | Bandwidth model documented (§6.4) but not validated empirically |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | SPDX header ✅, no `# noqa`, no `# type: ignore` |

Net: **3 WARN, 1 FAIL.** All four items resolve to the same root cause —
no `mpi4py` in this environment. Adding `mpi4py` + a multi-rank test
harness (task #17) closes WARN #2, WARN #5, FAIL #4 in one stroke.
WARN #3 (per-rank Rust) is a separate enhancement.

---

## 13. Known issues & limitations

### 13.1 spike_gating refused (was: silently ignored)

`Network.run(spike_gating=True, backend="mpi")` now raises
`NotImplementedError` (validated in `Network._run_mpi` before
`MPIRunner` is constructed). Sparse-firing networks must use
`backend="python"` to benefit from gating. The runner itself still
calls `pop.step_all(currents)` without the flag — fail-fast at the
dispatcher prevents silent wrong results.

### 13.2 FIM feedback refused (was: silently ignored)

`Network.run(backend="mpi")` on a network constructed with
`fim_lambda > 0` raises `NotImplementedError`. `MPIRunner.run` does
not call `Network._apply_fim`; refusing the run is preferred over
silently dropping the synchronisation feedback. Use `backend="python"`.

### 13.3 No per-rank Rust dispatch

The class docstring previously claimed "Works with both Python and Rust
backends per-rank". The corrected docstring (since task #19) states the
actual behaviour: `_step_local` always calls `Population.step_all`
(pure-Python). The Rust fast-path is only available via
`backend="auto"` / `backend="rust"` on a single process.

### 13.4 Monitoring on rank 0 only

`StateMonitor` is not driven from `MPIRunner` at all. `SpikeMonitor` and
`RateMonitor` only see rank 0's view of `all_spikes`. Other ranks have
no read-out mechanism.

### 13.5 `MPI.BYTE` mocked as `int(0)` in tests

The mocks set `mpi_mock.BYTE = 0`. This works only because `_exchange_spikes`
treats it as an opaque token passed straight to `Allgatherv`. Real `mpi4py`
uses the `MPI.Datatype` object — the contract holds, but tests would not
catch a regression that started reading `MPI.BYTE.size` or similar.

### 13.6 Round-robin partition is not load-balanced

See §4. Affects performance, not correctness.

### 13.7 `np.random.seed(network.seed + rank)` mutates global state

The runner sets the global NumPy RNG state at the start of `run`. Other
code that relies on NumPy's global RNG before/after the MPI run will see
non-deterministic behaviour. Prefer per-population `default_rng(seed)`
patterns elsewhere; do not assume the global state is preserved.

---

## 14. References

- Message Passing Interface Forum. *MPI: A Message-Passing Interface
  Standard, Version 4.0* (2021). [mpi-forum.org/docs/](https://www.mpi-forum.org/docs/)
- Dalcín L., Paz R., Storti M. "MPI for Python." *Journal of Parallel and
  Distributed Computing* 65:1108-1115 (2005). The mpi4py origin paper.
- Plesser H. E. *et al.* "Efficient parallel simulation of large-scale
  neuronal networks on clusters of multiprocessor computers."
  *Euro-Par 2007 Lecture Notes in Computer Science* 4641:672-681. The
  inspiration for "every rank knows every spike" via collectives.

Internal:

- Network orchestrator: [`api/network.md`](network.md)
- Population / Projection: [`api/network.md#3-network--orchestrator`](network.md)

---

## 15. Auto-rendered API

::: sc_neurocore.network.mpi_runner
    options:
      show_root_heading: true
      show_source: true
      members:
        - MPIRunner
        - HAS_MPI
