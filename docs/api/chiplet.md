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

The historical `sc_neurocore.chiplet.chiplet_gen` import path remains a
compatibility facade. Its 36 public objects are implemented by focused,
acyclic modules:

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

`ChipletGenerator.generate(topology, routing)` emits:

- one die wrapper per die, including the local AER router and LFSR;
- one asynchronous FIFO, latency pipe, and decorrelator per directed link;
- optional per-die AER routing tables;
- a package top that connects every die and bridge AXI-Stream port;
- XDC clocks and link-delay constraints.

`emit_crc32_sv` emits IEEE 802.3 normal and reflected CRC-32 feedback logic.
`emit_credit_controller_sv` emits a saturating receiver-credit controller.
`emit_power_gating_sv` isolates a voltage island before switch-off and waits
four cycles after restoration before de-isolating it.

The modularisation preserves generated bytes for the canonical three-die EMIB
ring, including die wrappers, bridges, routing table, package top, XDC, CRC,
credit, and power-controller outputs.

## Hierarchical partitioner

`HierarchicalPartitioner` recursively bisects a correlation-aware CSR graph,
then applies KL refinement. Its objective combines edge cut, load balance, and
boundary stochastic correlation. Ghost-cell, boundary-synchronisation, LFSR
seed-allocation, migration, and MPI-rank mapping objects are exposed through
the same package root.

The KL refinement has real Rust, Julia, Go, Mojo, and Python implementations
behind a shared CSR-flat ABI. `benchmarks/bench_kl_refine.py` is the applicable
cross-language benchmark; this chiplet-control refactor does not change those
kernels.

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

## Performance evidence

Reproduce the committed Python measurement with:

```bash
PYTHONPATH=src taskset -c 10 .venv/bin/python benchmarks/bench_chiplet.py \
  --json benchmarks/results/bench_chiplet.json
```

The committed run used Python 3.12.3, NumPy 2.2.6, Linux 6.17.0-35,
30 repeats per cell, and scheduler affinity `[10]`. One-minute load was 17.65,
so the numbers are diagnostic and are not isolated-core throughput claims.
The JSON includes the runner hash and every imported chiplet source hash.

| Operation | Size | Median (ms) | Minimum (ms) |
|---|---:|---:|---:|
| `make_torus` | 2×2 | 0.037 | 0.030 |
| `make_torus` | 4×4 | 0.103 | 0.085 |
| `make_torus` | 8×8 | 0.516 | 0.331 |
| `compute_decorrelation_seeds` | 16 links | 0.008 | 0.006 |
| `compute_decorrelation_seeds` | 64 links | 0.039 | 0.026 |
| `compute_decorrelation_seeds` | 256 links | 0.134 | 0.083 |
| `estimate_package_energy` | 4 dies | 0.003 | 0.003 |
| `estimate_package_energy` | 16 dies | 0.012 | 0.011 |
| `estimate_package_energy` | 64 dies | 0.046 | 0.045 |
| `simulate_thermal` | 4 dies | 0.064 | 0.061 |
| `simulate_thermal` | 16 dies | 0.134 | 0.109 |
| `simulate_thermal` | 64 dies | 0.502 | 0.347 |
| `ChipletGenerator.generate` | 2 dies | 0.427 | 0.243 |
| `ChipletGenerator.generate` | 8 dies | 1.906 | 1.284 |
| `ChipletGenerator.generate` | 32 dies | 8.219 | 5.711 |
| `HierarchicalPartitioner.partition` | V=50, P=2 | 3.131 | 1.824 |
| `HierarchicalPartitioner.partition` | V=100, P=4 | 6.697 | 3.703 |
| `HierarchicalPartitioner.partition` | V=200, P=4 | 13.116 | 5.385 |

A same-process, alternating-order comparison against the pre-refactor source
showed topology construction 26–33% faster, seed and energy operations within
approximately ±10%, full RTL generation within −7% to +7%, and thermal solving
22–38% slower because each die, power input, and solver boundary now validates
finite physical values. The comparison was pinned but not exclusive and ran
under high host load; it supports regression diagnosis, not a performance
promotion claim.

## Verification

The focused chiplet cohort contains 172 tests and exercises package imports,
historical identities, pickling, topology validation, route selection, thermal
physics, byte-stable RTL generation, Icarus parsing and simulation, power-state
timing, and the hierarchical partitioner boundary. The nine refactored
production files cover 680 statements and 202 branches at 100%, with no misses
or partial branches. Strict MyPy and Ruff cover the affected source, tests, and
benchmark runner.

## Limitations

- The package remains research-tier; physical PPA and foundry sign-off happen
  in downstream flows.
- Chiplet-control functions are Python-only by measured architectural choice.
- Benchmark timings are host-load-sensitive and the committed run was not
  exclusive.
- `tests/test_debug/test_hierarchical_partitioner.py` remains historically
  located outside `tests/test_chiplet`; moving it is separate test-layout work.

## References

- UCIe Consortium. *UCIe Specification 1.0*. 2022.
- Open Compute Project. *Bunch of Wires Specification*. 2022.
- Skadron, K. et al. “HotSpot: A Compact Thermal Modeling Methodology for
  Early-Stage VLSI Design.” *IEEE Transactions on VLSI Systems*, 2006.
- Karypis, G. and Kumar, V. *METIS: A Software Package for Partitioning
  Unstructured Graphs and Computing Fill-Reducing Orderings of Sparse
  Matrices*. University of Minnesota, 1998.
