<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Math

Mathematical foundations: category theory, topological observables,
and differential geometry on coupling graphs.

## Category Theory

::: sc_neurocore.math.category_theory

## Topological Observables

Geometric invariants computed on SCPN coupling graphs: winding number
from phase dynamics, Ollivier-Ricci curvature on edges, sheaf
consistency defect, and connection curvature from parallel transport.

`ollivier_ricci_curvature` evaluates the graph-metric definition on the
non-negative coupling support: lazy random-walk measures are transported by
Wasserstein-1 distance over shortest-hop graph distances, and invalid matrices
or node indices fail closed before transport.

### Ollivier-Ricci curvature backends

The curvature solve is the one compute-bound observable in this module — each
node pair runs an exact successive-shortest-path min-cost flow to obtain the
Wasserstein-1 (earth-mover) distance between the two lazy random-walk measures.
It therefore carries a polyglot accelerator chain selected through the
`backend` argument (`"auto"`, `"rust"`, `"julia"`, `"go"`, `"mojo"`,
`"python"`). `"auto"` prefers the Rust path (shipped in the
`sc_neurocore_engine` wheel) and falls back to the pure-NumPy reference. Every
accelerator reproduces the NumPy reference to machine epsilon because all five
implementations share the same deterministic Bellman-Ford iteration order, so
the chosen augmenting paths — and the floating-point accumulation of the
transport cost — coincide.

```python
from sc_neurocore.math.topology import ollivier_ricci_curvature

# auto picks the fastest available compiled backend, else pure NumPy
kappa = ollivier_ricci_curvature(coupling_matrix, i=0, j=7)

# force a specific backend (raises if that backend is not built)
kappa_rust = ollivier_ricci_curvature(coupling_matrix, 0, 7, backend="rust")
```

The lighter observables (winding number, sheaf defect, connection curvature)
are single vectorised NumPy expressions for which NumPy is already the fastest
path; they are intentionally not accelerated.

#### Measured performance

Reproduce with `python benchmarks/bench_topology.py --json
benchmarks/results/bench_topology.json`. Workload: 15 curvature solves per
sweep over weighted random graphs of N = 20, 50, 100; median of 7 repeats.
These figures are **non-isolated** (loaded developer workstation, Python
3.12 / NumPy 2.3) and are functional/regression evidence, not isolated-core
release numbers.

| backend | median (ms) | speedup vs NumPy | parity Δ vs NumPy |
|---|---:|---:|---:|
| python (NumPy) | 3385.19 | 1.00× | 0 |
| mojo | 66.80 | 50.68× | 7.8e-16 |
| rust | 68.15 | 49.67× | 7.8e-16 |
| go | 90.40 | 37.45× | 7.8e-16 |
| julia | 93.16 | 36.34× | 3.3e-16 |

Mojo and Rust are within measurement noise of each other on this loaded host;
`"auto"` selects Rust because it ships in the wheel and needs no local build.

::: sc_neurocore.math.topology.winding_number

::: sc_neurocore.math.topology.ollivier_ricci_curvature

::: sc_neurocore.math.topology.sheaf_consistency_defect

::: sc_neurocore.math.topology.connection_curvature
