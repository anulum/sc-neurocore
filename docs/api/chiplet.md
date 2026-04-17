# `sc_neurocore.chiplet` — Multi-die chiplet generator

## 1. Scope

The `sc_neurocore.chiplet` package generates the SystemVerilog
substrate, routing tables, and physical-design constraints for
**multi-die SC-NeuroCore deployments** — the architectural form
factor for ASIC + FPGA + interposer co-packages where a single
silicon die is too small to host the full network.

It targets two complementary problems:

1. **Chiplet generation** (`chiplet_gen`) — given a target die
   count, technology (UCIe / BoW / EMIB / CoWoS / Organic /
   Custom), and topology (mesh / torus / star / ring / 3D
   stacked), emit:
   - per-die SystemVerilog wrappers
   - die-to-die bridge IP (CDC, credit, CRC32 protection)
   - top-level + Vivado XDC constraints
   - link-energy + thermal + congestion + timing reports
2. **Hierarchical partitioning** (`hierarchical_partitioner`) —
   given a network graph (CSR + correlation edges), divide
   neurons across the dies so that (a) inter-die traffic is
   minimised, (b) per-die load is balanced, (c) ghost-cell halo
   exchange is bounded, (d) LFSR-seed allocation prevents
   correlation-induced bias.

The package is the bridge between the SC-NeuroCore network
description (`sc_neurocore.network`) and the physical-design
back-end (`sc_neurocore.asic_flow` for tape-out,
`sc_neurocore.uvm_gen` for verification).

## 2. Public API surface

The package re-exports 57 symbols from two modules:

- **`chiplet_gen`** — 36 symbols: 3 enums + 15 dataclasses +
  18 generators / analysers / SV emitters.
- **`hierarchical_partitioner`** — 21 symbols: 1 enum + 7
  dataclasses + 13 orchestrators / metric functions.

Top-level imports:

```python
from sc_neurocore.chiplet import (
    InterposerTech, InterposerLink, ChipletDie, ChipletTopology,
    ChipletGenerator, ChipletOutput, RoutingTable,
    HierarchicalPartitioner, CSRGraph, CorrelationAwareGraph,
    GhostCellManager, LFSRSeedAllocator, RankMapper,
    estimate_package_energy, simulate_thermal, estimate_congestion,
    make_torus, add_3d_stack, compute_decorrelation_seeds,
    # ... 38 more — see `__all__` for the full list
)
```

`__tier__ = "research"` — appropriate for research-tier
deployments where the user accepts that the generated artefacts
are inputs to downstream physical-design tools (Vivado, Innovus,
Genus) rather than tape-out-ready bitstreams on their own.

## 3. Interposer technology presets

`InterposerTech` enum has 6 members; `InterposerLink.from_tech(...)`
constructs a link with technology-specific defaults:

| Tech | Latency (ns) | Bandwidth (Gb/s) | BER (per bit) | Notes |
|---|---:|---:|---:|---|
| `UCIE` | 2.0 | 32.0 | 1e-15 | Universal Chiplet Interconnect Express, AMD/Intel/Arm consortium standard |
| `BOW` | 1.5 | 16.0 | 1e-12 | Bunch-of-Wires, Open Compute Project standard |
| `EMIB` | 1.0 | 64.0 | 1e-15 | Intel Embedded Multi-die Interconnect Bridge (silicon bridge) |
| `COWOS` | 0.5 | 128.0 | 1e-16 | TSMC Chip-on-Wafer-on-Substrate (silicon interposer) |
| `ORGANIC` | 5.0 | 8.0 | 1e-12 | Organic substrate (BGA-style routing, lowest cost, slowest) |
| `CUSTOM` | 2.0 | 32.0 | 1e-15 | User-defined, defaults to UCIe-like timing |

Latency ordering: CoWoS (0.5 ns) < EMIB (1.0) < BoW (1.5) <
UCIe = Custom (2.0) < Organic (5.0). Bandwidth is roughly in
inverse order (highest for the most expensive interposer).

3D stacking (`StackingType.TSV_3D`, `HYBRID_BONDING`,
`COPLANAR`) is handled by `add_3d_stack(...)` which constructs
both forward and reverse `InterposerLink` records (3D links are
inherently bidirectional).

## 4. Routing model

`compute_decorrelation_seeds(topology)` allocates a unique LFSR
seed per inter-die link to prevent correlated-noise injection
across the boundary. The function uses the **golden ratio
(φ⁻¹ ≈ 0.6180)** as a low-discrepancy hash modulo 65 535 — this
guarantees that the seeds form a quasi-uniform distribution over
the 16-bit space even when the link count is small, while
remaining deterministic.

The return type is `Dict[Tuple[int, int], int]` — the key is the
`(src_die, dst_die)` tuple. (This was a `mypy`-found bug in
Antigravity's draft: the signature said `Dict[int, int]` but the
implementation already used tuple keys; the consumer at
`ChipletGenerator.emit()` line 453 looked up the tuple key
correctly. Fixed by Arcane Sapience in this batch.)

## 5. Energy + thermal + congestion analysis

Per-link energy is computed by `link_energy_pj(link, bits)`
using a per-technology `_ENERGY_PJ_PER_BIT` lookup table:

| Tech | pJ / bit |
|---|---:|
| UCIE | 0.5 |
| BoW | 0.3 |
| EMIB | 0.2 |
| CoWoS | 0.15 |
| Organic | 1.0 |
| Custom | 0.5 |

`estimate_package_energy(topology, traffic_matrix)` aggregates
per-link energy weighted by traffic to produce a
`PackageEnergyReport` (per-die total, per-link breakdown,
total package draw).

`simulate_thermal(topology, power_per_die, ambient_temp)` runs
a finite-difference thermal solve over the die layout to
produce a `PackageThermalReport` (per-die junction temperature,
hot-spot location, thermal throttling alert).

`estimate_congestion(topology, routing)` returns a
`CongestionReport` describing per-link utilisation under the
specified routing — useful for finding bottleneck links before
silicon commitment.

## 6. Hierarchical partitioner

`HierarchicalPartitioner(num_partitions=N)` divides a `CSRGraph`
(compressed-sparse-row neuron connectivity) across N dies by
recursive bisection. The objective is multi-criteria:

1. **Edge cut** — minimise inter-die communication
   (`calculate_edge_cut`).
2. **Load balance** — keep `vertex_count[i] / mean(vertex_count)`
   within `imbalance_threshold` (`calculate_imbalance_ratio`).
3. **Boundary stochastic correlation coefficient (SCC)** — per
   `feedback_multi_language_accel.md`-style decorrelation, the
   boundary's mean SCC should be below a configurable threshold
   (`calculate_mean_boundary_scc`).

`MigrationRecommendation` captures a proposed `(vertex,
src_partition, dst_partition, gain)` quad emitted when a
partition is overloaded; consumers can apply or reject each
recommendation.

`GhostCellManager` orchestrates halo-exchange — the per-rank
overlap region used by MPI-distributed simulation
(`sc_neurocore.network.MPIRunner`) — by tracking which neurons
are owned by which rank and which need to be mirrored.

`LFSRSeedAllocator` allocates one LFSR seed per partition such
that no two partitions share a seed, preventing correlated
noise across the partition boundary.

`RankMapper` maps logical partition IDs onto MPI rank IDs,
respecting NUMA topology hints when supplied.

## 7. SystemVerilog emitters

The package emits SystemVerilog source files that are compiled
by downstream EDA tools (Vivado for FPGA, Innovus for ASIC):

- `emit_crc32_sv(data_width)` — CRC32-error-detecting bridge
- `emit_credit_controller_sv(config, link_name)` — credit-based
  flow control to prevent buffer overflow at the receiver
- `emit_power_gating_sv(domain)` — fine-grained power-gating
  state machine for each `PowerDomain`

These are pure string emitters (template substitution); the
generated code is consumed by `sc_neurocore.asic_flow` /
`sc_neurocore.uvm_gen` for further synthesis + verification.

## 8. Pipeline wiring

`sc_neurocore.chiplet` sits **between** the network description
and the physical-design back-end:

1. The user defines a network (`sc_neurocore.network.Network`).
2. `HierarchicalPartitioner` divides the neurons across N dies.
3. `ChipletGenerator(topology=..., routing=...)` emits the
   per-die SystemVerilog + bridges + top-level + XDC.
4. `simulate_thermal` + `estimate_package_energy` +
   `estimate_congestion` produce signoff reports.
5. The output (`ChipletOutput`) is fed to
   `sc_neurocore.asic_flow` for tape-out or to
   `sc_neurocore.uvm_gen` for verification testbench
   generation.

There is no Rust / Julia / Go / Mojo path in this package — it
is dominated by **string-template emission** and **graph
algorithms over Python dicts**, both of which are sub-second on
realistic problem sizes (≤ 1024 dies). Per the
`feedback_multi_language_accel.md` rule, this is the "I/O
adapter and visualisation" exemption category. If the partition
solve becomes a bottleneck on > 10 000 dies, the candidate
acceleration backend is Julia (Metis.jl is the published
state-of-the-art for graph partitioning), not Rust.

## 9. Pure-Python performance

| Operation | Problem size | Wall time |
|---|---|---:|
| `ChipletGenerator.emit()` | 4 dies, mesh, ~1 KB SV per die | ~10 ms |
| `ChipletGenerator.emit()` | 16 dies, torus, ~10 KB SV per die | ~80 ms |
| `make_torus(8, 8)` + emit | 64 dies, 256 links | ~400 ms |
| `compute_decorrelation_seeds` | 256 links | ~0.3 ms |
| `estimate_package_energy` | 64 dies, 256 links | ~5 ms |
| `simulate_thermal` | 64 dies, 1000-step solve | ~300 ms |
| `HierarchicalPartitioner.partition()` | 1000-vertex CSR, N=4 | ~50 ms |
| `HierarchicalPartitioner.partition()` | 10 000-vertex CSR, N=16 | ~3 s |

(Numbers from informal `python -m timeit` runs on Intel
i5-11600K, NumPy 2.2.0, Python 3.12.3; not from a committed
benchmark — see followup §12.1.)

## 10. Test coverage

Three test files cover this package:

| File | Tests | LOC | What it covers |
|---|---:|---:|---|
| `tests/test_chiplet/test_chiplet_gen.py` | 94 | 565 | Antigravity-authored 14 unittest classes covering interposer links, dies, topology, routing tables, decorrelation, generator, timing, star + torus topologies, link energy, congestion, disjoint paths, CDC, thermal, adaptive routing |
| `tests/test_debug/test_hierarchical_partitioner.py` | 52 | (existing) | Antigravity-authored partition correctness + correlation-aware partitioning + LFSR seed allocator + ghost cell manager + boundary sync |
| `tests/test_chiplet/test_chiplet_public_api.py` | 12 | new | Arcane Sapience: package re-exports identity for both modules, `__all__` membership for 57 symbols, `InterposerTech` 6-member enum, `InterposerLink.from_tech` smoke for all 6 presets, `compute_decorrelation_seeds` returns tuple-keyed dict (regression test for the mypy-found bug), `make_torus` smoke, `HierarchicalPartitioner` constructor smoke |

**Total: 158 tests.** All run in ~2 s combined; no skips, no
failures.

`tests/test_debug/test_hierarchical_partitioner.py` is mis-located
(should be under `tests/test_chiplet/`) but moving it is deferred
to a separate refactor commit.

## 11. Audit completeness — 7-point rule

| # | Criterion | Status | Notes |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 57 symbols re-exported via `__init__.py`; verified by `test_chiplet_public_api.py` |
| 2 | Multi-angle tests | ✅ PASS | 158 tests across 3 files; covers topology + routing + energy + thermal + partitioning + LFSR + ghost cells + boundary sync |
| 3 | Acceleration path | N/A (deferred) | String-template emission and graph algorithms; no current backend. Future Julia (Metis.jl) candidate noted in §8 |
| 4 | Benchmarks | ⚠️ WARN | Informal `timeit` numbers in §9; no committed benchmark script |
| 5 | Performance docs | ✅ PASS | §9 with explicit "informal" caveat |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | SPDX 2-line header on `__init__.py`, `chiplet_gen.py`, `hierarchical_partitioner.py` (`__init__.py` and `chiplet_gen.py` fixed in this batch from 1-line piped form; `chiplet_gen.py` also had `# mypy: ignore-errors` removed and 7 real mypy errors fixed). British English in this doc; source uses standard scientific-Python identifiers (acceptable per docs-vs-code rule). |

Net: **1 WARN, 0 FAIL.**

## 12. Known issues / follow-ups

### 12.1 No committed benchmark (WARN row 4)

Open follow-up: commit `benchmarks/bench_chiplet.py` reproducing
§9 numbers (5–10 representative `topology × N_dies` cells,
median-of-5 protocol). Lower priority because chiplet generation
is offline (run once per silicon revision) and sub-second.

### 12.2 Mypy fixes applied in this batch

`chiplet_gen.py` had `# mypy: ignore-errors` masking 7 real
type errors:

1. Line 88 + 1125 + 1134: `**presets[tech]` unpacking failed
   because `presets: dict[..., dict[str, float]]` was inferred
   homogeneously but the dataclass receives `int` and `bool`
   for some fields. Fixed by annotating
   `presets: Dict[..., Dict[str, Any]]`.
2. Line 255: `compute_decorrelation_seeds` declared
   `Dict[int, int]` return type but actually returned
   `Dict[Tuple[int, int], int]`. Annotation corrected.
3. Line 453: dict.get with tuple key was rejected because the
   variable was bound to the wrong-typed dict. Fixed by 12.2.2.
4. Line 934: `__post_init__` missing `-> None` annotation.
   Added.

`hierarchical_partitioner.py` had 1 mypy error: line 703
`recs = []` needed `list[MigrationRecommendation]` annotation.
Added.

### 12.3 `tests/test_debug/test_hierarchical_partitioner.py` mis-located

The file lives under `tests/test_debug/` but exercises
`sc_neurocore.chiplet.hierarchical_partitioner`. Should be moved
to `tests/test_chiplet/test_hierarchical_partitioner.py` for
discoverability. Deferred to a separate housekeeping commit.

### 12.4 Pre-existing doc was a stub with fabricated names

`docs/api/chiplet.md` was a 14-line `mkdocstrings` auto-gen
stub. The `Quick Start` block listed FABRICATED import names:
`InterconnectTopo`, `ThermalModel`, `YieldEstimator` — none of
which exist in the module. The actual class names are
`InterposerTech` (closest), `simulate_thermal` (function, not
class), and there is no `YieldEstimator`. Replaced with this
page in the same batch.

### 12.5 No semantic bugs found

Audit found:
- `# mypy: ignore-errors` on `chiplet_gen.py` was masking 7
  real type errors — all fixed (see 12.2).
- `__init__.py` did not re-export the 57 public symbols. Wired.
- 1-line piped SPDX header in `__init__.py` and `chiplet_gen.py`
  (the latter actually had BOTH headers stacked: piped at
  line 1 + canonical at line 7, with `# mypy: ignore-errors` at
  line 6). Cleaned up.
- 1-line piped SPDX in `hierarchical_partitioner.py` was
  ALREADY canonical 2-line — no fix needed.

No semantic bugs (sign errors, off-by-ones, wrong invariants,
fabricated constants) found in either source file. The 146
Antigravity tests pass; the 12 new public-API tests pass.

## 13. References

- Universal Chiplet Interconnect Express (UCIe) Consortium:
  *UCIe Specification 1.0*. Beaverton OR: UCIe Forum, 2022.
- Open Compute Project: *Bunch-of-Wires (BoW) Specification*.
  Menlo Park CA: OCP, 2022.
- Intel Foundry: *Embedded Multi-die Interconnect Bridge
  (EMIB) White Paper*. Santa Clara CA, 2017.
- TSMC: *Chip-on-Wafer-on-Substrate (CoWoS) Technology Brief*.
  Hsinchu, 2018.
- Karypis, G. & Kumar, V. (1998). *METIS — A Software Package
  for Partitioning Unstructured Graphs and Computing
  Fill-Reducing Orderings of Sparse Matrices*. University of
  Minnesota.
- Pellegrini, F. (2007). *Scotch and libScotch — Sparse Matrix
  Ordering and Parallel Graph Partitioning*. INRIA Bordeaux.

## 14. Audit batch identification

This page was produced as part of the **Antigravity audit, batch
B1, package 3** (per
`docs/internal/antigravity_inventory_2026-04-17.md`). B1 closes
with this commit. B2 (`bci_studio/`, `analog_bridge/`,
`asic_flow/`, plus the existing `chip_compiler/`) follows in
subsequent batches.
