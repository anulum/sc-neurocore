# `sc_neurocore.chiplet` — Multi-die package generation

## Scope

`sc_neurocore.chiplet` maps partitioned neuron networks onto multiple dies. It
models package topology, derives AER routes, analyses communication and thermal
constraints, and emits connected SystemVerilog plus XDC timing constraints.
`sc_neurocore.chiplet.hierarchical_partitioner` supplies the upstream graph
partition.

The package is research-tier. Generated RTL is an input to downstream FPGA or
ASIC verification and physical-design tools; it is not a foundry sign-off
artefact.

## Responsibility modules

The historical `sc_neurocore.chiplet.chiplet_gen` and
`sc_neurocore.chiplet.hierarchical_partitioner` import paths remain
compatibility facades. The 36 chiplet-generator objects are implemented by
focused, acyclic modules:

| Module | Responsibility |
|---|---|
| `topology` | Dies, interposer links, mesh/ring/star/torus construction, and 3D stacking |
| `routing` | AER routes, link-disjoint paths, timing, congestion, decorrelation seeds, and energy |
| `thermal` | Steady-state and implicit-Euler transient package thermal solving |
| `rtl` | Die wrappers, CDC bridges, routing tables, package top, and XDC emission |
| `link_protocols` | Clock-domain configuration, CRC-32, and credit flow control |
| `power` | Voltage-island ownership and sequenced power-gating RTL |
| `partition` | Neuron-to-die assignments translated into routing tables |

The package root continues to re-export 57 objects: 36 from the historical
chiplet-generator surface and 21 from `hierarchical_partitioner`.

```python
from sc_neurocore.chiplet import (
    ChipletGenerator,
    ChipletTopology,
    HierarchicalPartitioner,
    InterposerTech,
    RoutingTable,
    estimate_congestion,
    estimate_package_energy,
    make_torus,
    simulate_thermal,
)
```

Historical imports and pickle-qualified names remain valid:

```python
from sc_neurocore.chiplet.chiplet_gen import ChipletTopology

assert ChipletTopology.__module__ == "sc_neurocore.chiplet.chiplet_gen"
```

## Topology and interposer models

`InterposerLink.from_tech(source, destination, technology)` applies the
following default link contract:

| Technology | Latency (ns) | Jitter (ns) | Bandwidth (Gb/s) | BER |
|---|---:|---:|---:|---:|
| UCIe | 2.0 | 0.05 | 32 | 1e-15 |
| BoW | 1.5 | 0.03 | 16 | 1e-12 |
| EMIB | 1.0 | 0.02 | 64 | 1e-15 |
| CoWoS | 0.5 | 0.01 | 128 | 1e-16 |
| Organic | 5.0 | 0.5 | 8 | 1e-12 |
| Custom | 2.0 | 0.1 | 32 | 1e-15 |

Link identifiers must be non-negative. Latency and jitter must be finite and
non-negative; bandwidth and data width must be positive; BER must be in
`[0, 1]`. A measured custom thermal resistance may be supplied in K/W and must
be finite and positive.

`ChipletTopology.mesh_2d`, `.ring`, `.star`, and `make_torus` construct
deterministic layouts with non-zero per-die LFSR seeds. `add_3d_stack` adds
reciprocal links for TSV, hybrid-bonded, or coplanar associations.

## Routing and package analysis

`RoutingTable` maps a source neuron to a destination die, destination neuron,
and Q8.8 weight. `PartitionAssignment.to_routing_tables` omits local and
unmapped connections and creates tables only for source dies with cross-die
traffic.

`compute_decorrelation_seeds` assigns each directed link a deterministic
non-zero 16-bit seed using a golden-ratio sequence. `find_disjoint_paths`,
`adaptive_route`, and `bandwidth_aware_route` operate on the same directed
topology graph.

Communication energy uses the following package-model coefficients:

| Technology | Energy (pJ/bit) |
|---|---:|
| UCIe | 0.5 |
| BoW | 0.3 |
| EMIB | 0.2 |
| CoWoS | 0.1 |
| Organic | 2.0 |
| Custom | 0.5 |

`estimate_package_energy(topology, bits_per_link)` applies uniform traffic to
each directed link. `estimate_congestion` converts routed events at the
historical 200 MHz reference clock into capacity utilisation. Negative traffic
or event-rate inputs are rejected.

## Thermal model

`simulate_thermal` builds a symmetric inter-die conductance matrix and a
per-die ambient path. The steady state solves

\[
(D-G)T=P+g_{amb}T_{amb},
\]

where `G` is the off-diagonal bond conductance and `D` contains row sums plus
ambient conductance. Optional transients use implicit Euler:

\[
(C/\Delta t + D-G)T_{k+1}=C/\Delta t\,T_k + P + g_{amb}T_{amb}.
\]

The returned `PackageThermalReport` contains steady-state temperatures,
throttled die identifiers, the exact conductance matrix, and optional transient
time and temperature arrays. Empty topologies, non-finite values, negative
power, non-positive time steps, duplicate die identifiers, and invalid material
properties fail before solving.

## Generated RTL

`ChipletGenerator.generate(topology, routing, *, decorrelate_links=False)` emits:

- one die wrapper (`sc_chiplet_die_<d>`) per die, binding multi-fanout
  `(dst_die, dst_neuron, weight_q88)` arrays into `sc_aer_router` (requiring
  `local_spike_id < die.n_neurons`), mapping cross-die next hops along shortest
  directed topology paths, and arbitrating local and incoming interposer traffic
  onto local delivery or outgoing links with lossless `ready` backpressure
  (fixed priority: local router first, then incoming links in declaration order;
  not deadlock-free on directed link cycles if every bounded FIFO on a cycle
  fills);
- one asynchronous FIFO, stall-aware latency pipeline, and optional transparent
  per-link LFSR scrambler/descrambler (`decorrelate_links=True`, scrambling
  `s_tdata` before the CDC FIFO and descrambling after the latency pipeline from
  `DECOR_SEED`) per directed link;
- optional single-cycle primary-route lookup modules (`sc_chiplet_rtable_<d>`)
  sized by `die.n_neurons`, with `target_neuron` widened to the package AER
  width (`PKG_AER_W`) for heterogeneous dies and valid signed fixed-point
  literals;
- a package top (`sc_chiplet_top`) exposing per-die AER ingress/egress ports
  and connecting every die and bridge AXI-Stream port;
- XDC clocks and `-datapath_only` inter-clock link-delay constraints;
- fail-closed validation rejecting duplicate dies or directed links, narrow
  link payloads, out-of-range neuron IDs or signed weights, and unreachable
  routed destination dies.

`emit_crc32_sv` emits IEEE 802.3 normal and reflected CRC-32 feedback logic.
`emit_credit_controller_sv` emits a saturating receiver-credit controller.
`emit_power_gating_sv` isolates a voltage island before switch-off and waits
four cycles after restoration before de-isolating it.

## Hierarchical partitioner

`HierarchicalPartitioner` recursively bisects a correlation-aware CSR graph,
then applies KL refinement. Its objective combines edge cut, load balance, and
boundary stochastic correlation. Ghost-cell, boundary-synchronisation, LFSR
seed-allocation, migration, and MPI-rank mapping objects are exposed through
the same package root.

The historical module is a 104-line facade. Its 21 public definitions retain
their established import identity, qualified names, and pickle paths while ten
acyclic implementation modules own one responsibility each:

| Module | Responsibility |
|---|---|
| `hierarchical_graph` | Typed graph, edge, CSR, hierarchy, and deterministic seed models |
| `hierarchical_backend_runtime` | Lazy discovery and loading of installed backend runtimes |
| `hierarchical_backends` | Shared flat-buffer ABI encoding, validation, dispatch, and result decoding |
| `hierarchical_bisection` | Coarsening, deterministic spectral scoring, and recursive bisection |
| `hierarchical_refinement` | Python reference KL refinement and incremental repartitioning |
| `hierarchical_core` | Public partitioning orchestration and backend selection |
| `hierarchical_metrics` | Edge-cut, SCC-boundary, imbalance, and communication metrics |
| `hierarchical_boundary` | Ghost-cell discovery and SCC-budgeted boundary synchronisation |
| `hierarchical_balancing` | Runtime load metrics, migrations, and MPI rank mapping |
| `hierarchical_reporting` | Stable report model and deterministic report composition |

The KL refinement has real Rust, Julia, Go, Mojo, and Python implementations
behind a shared CSR-flat ABI. Backend output must decode to one complete,
duplicate-free partition map before it can replace the Python result. Auto
dispatch tries only installed maintained backends and falls back to the Python
reference; an explicitly requested unavailable or malformed backend fails
closed.

## Pipeline wiring

1. A network graph is partitioned across target dies.
2. `PartitionAssignment` derives cross-die routing tables.
3. `ChipletGenerator` emits connected RTL and constraints.
4. Timing, energy, congestion, and thermal functions produce package evidence.
5. Generated artefacts feed `sc_neurocore.asic_flow` and
   `sc_neurocore.uvm_gen`.

## Backend truth

The chiplet-control operations have no installed non-Python compute backend.
Historical files named for Rust, Julia, Go, and Mojo were non-executable
scaffolds and are absent. This is distinct from the real multi-language KL
refinement used by `hierarchical_partitioner`.

| Chiplet-control backend | Installed | Reason |
|---|:---:|---|
| Python | yes | Authoritative package-control and analysis path |
| Rust | no | Numerical operations are sub-ms; RTL generation is textual graph orchestration |
| Julia | no | Same boundary; first-call JIT also exceeds the numerical operation budget |
| Go | no | Same boundary; the former service contained empty functions |
| Mojo | no | Same boundary; the former kernel stored source text and returned zero |

`benchmarks/results/bench_chiplet.json` records these backends as unavailable
and exempt, rather than claiming that scaffold files are acceleration.

The partitioner is a different boundary. Its maintained implementations are
`engine/src/partition.rs`, `accel/julia/chiplet/kl_refine.jl`,
`accel/go/partition/partition.go`, and
`accel/mojo/partition/partition.mojo`. Historical files named
`hierarchical_partitioner` in generated acceleration directories were
non-executable mirrors and are absent; they are not counted as backends.

## Performance evidence

Reproduce the committed Python measurement with:

```bash
PYTHONPATH=src taskset -c 10 .venv/bin/python benchmarks/bench_chiplet.py \
  --json benchmarks/results/bench_chiplet.json
```

The committed run used Python 3.12.16, NumPy 2.5.4, Linux 7.0.0-42,
30 repeats per cell, and scheduler affinity `[10]`. One-minute load was 13.79,
so the numbers are diagnostic and are not isolated-core throughput claims.
The JSON includes the runner hash and every imported chiplet source hash.

| Operation | Size | Median (ms) | Minimum (ms) |
|---|---:|---:|---:|
| `make_torus` | 2×2 | 0.014 | 0.013 |
| `make_torus` | 4×4 | 0.053 | 0.049 |
| `make_torus` | 8×8 | 0.216 | 0.197 |
| `compute_decorrelation_seeds` | 16 links | 0.003 | 0.003 |
| `compute_decorrelation_seeds` | 64 links | 0.013 | 0.012 |
| `compute_decorrelation_seeds` | 256 links | 0.050 | 0.048 |
| `estimate_package_energy` | 4 dies | 0.001 | 0.001 |
| `estimate_package_energy` | 16 dies | 0.004 | 0.004 |
| `estimate_package_energy` | 64 dies | 0.017 | 0.017 |
| `simulate_thermal` | 4 dies | 0.030 | 0.027 |
| `simulate_thermal` | 16 dies | 0.108 | 0.069 |
| `simulate_thermal` | 64 dies | 0.271 | 0.209 |
| `ChipletGenerator.generate` | 2 dies | 0.300 | 0.263 |
| `ChipletGenerator.generate` | 8 dies | 2.203 | 1.721 |
| `ChipletGenerator.generate` | 32 dies | 35.707 | 27.061 |
| `HierarchicalPartitioner.partition` | V=50, P=2 | 1.220 | 0.879 |
| `HierarchicalPartitioner.partition` | V=100, P=4 | 2.562 | 1.754 |
| `HierarchicalPartitioner.partition` | V=200, P=4 | 4.258 | 3.454 |

A same-process, alternating-order comparison against the pre-refactor source
showed topology construction 26–33% faster, seed and energy operations within
approximately ±10%, full RTL generation within −7% to +7%, and thermal solving
22–38% slower because each die, power input, and solver boundary now validates
finite physical values. The comparison was pinned but not exclusive and ran
under high host load; it supports regression diagnosis, not a performance
promotion claim.

The maintained cross-language KL evidence is reproduced with:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_kl_refine.py \
  --label candidate --json benchmarks/results/bench_kl_refine.json
```

The source-bound parent (`d9c172c84`) and candidate runs produced the same
canonical partition SHA-256,
`258b5c0c54ea33758f60090e68b7ce2a7657800ec16a6c90bc0eb52fbbd3585f`,
for every workload and all five backends. The V=1000 medians are diagnostic:

| Backend | Parent (ms) | Candidate (ms) | Exact membership parity |
|---|---:|---:|:---:|
| Python | 287.309 | 113.720 | yes |
| Rust | 0.275 | 0.180 | yes |
| Julia | 0.473 | 0.260 | yes |
| Go | 0.767 | 0.360 | yes |
| Mojo | 0.495 | 0.240 | yes |

The committed result hashes the benchmark runner, all eleven Python
partitioner modules, every maintained kernel source, and the available compiled
binaries. The host used the powersave governor, affinity CPUs 0–11, and no
isolated CPU reservation, so timing is local regression evidence rather than a
throughput claim.

## Verification

The partitioner-owning exact cohort contains 111 tests and covers all 664
statements and 186 branches across the eleven `hierarchical*.py` modules, with
no misses or partial branches. The broader public-API, architecture, and
benchmark cohort contains 134 passing tests, and the complete chiplet package
contains 245 tests (including Icarus Verilog multi-die AER delivery,
contention/backpressure, heterogeneous-die routing-table, and end-to-end link
scrambling/descrambling co-simulations).
Architecture tests enforce the
acyclic dependency graph, one implementation owner per public object, dynamic
backend diagnostics, historical pickling, maintained kernel presence, false
mirror absence, the exact CI gate, and maximum file sizes of 258 production
lines and 291 test lines. Strict MyPy, complete public docstrings, and Ruff
cover the affected source, tests, and benchmark runner.

## Limitations

- The package remains research-tier; physical PPA and foundry sign-off happen
  in downstream flows.
- Chiplet-control functions are Python-only by measured architectural choice.
- Benchmark timings are host-load-sensitive and the committed run was not
  exclusive.

## References

- UCIe Consortium. *UCIe Specification 1.0*. 2022.
- Open Compute Project. *Bunch of Wires Specification*. 2022.
- Skadron, K. et al. “HotSpot: A Compact Thermal Modeling Methodology for
  Early-Stage VLSI Design.” *IEEE Transactions on VLSI Systems*, 2006.
- Karypis, G. and Kumar, V. *METIS: A Software Package for Partitioning
  Unstructured Graphs and Computing Fill-Reducing Orderings of Sparse
  Matrices*. University of Minnesota, 1998.
