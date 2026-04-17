# Network Topology Analysis

**Module:** `sc_neurocore.topology`
**Source:** `src/sc_neurocore/topology/analyzer.py` — 155 LOC
**Status (v3.14.0):** 2 public symbols (`TopologyAnalyzer`,
`TopologyReport`); 7 tests pass; pure-Python; **disjoint from**
`sc_neurocore.network.topology` (the connectivity-generator module
documented in [`api/network.md`](network.md) §6) — same word, different
module.

This page covers the **post-hoc graph metrics** for an SNN
connectivity matrix: clustering coefficient, average path length,
small-world sigma, degree statistics, hub identification, degree
assortativity. The connectivity-generator side (Erdős–Rényi,
Watts–Strogatz, Barabási–Albert, ring, grid, all-to-all) lives
in a different module and is documented separately.

---

## 1. Public surface

`sc_neurocore.topology.__init__` re-exports 2 symbols:

| Symbol | Source | Role |
|--------|--------|------|
| `TopologyAnalyzer` | `analyzer.py:51` | Stateful analyser bound to one adjacency matrix |
| `TopologyReport` | `analyzer.py:21` | Dataclass holding all computed metrics |

### Naming caveat

The two `topology` modules in sc-neurocore have nothing to do with
each other beyond sharing a word:

| Module | Purpose | LOC |
|--------|---------|----:|
| `sc_neurocore.topology` (this page) | Graph metrics on a given adjacency matrix | 155 |
| `sc_neurocore.network.topology` | Connectivity generators (random, small-world, ring, grid, …) | 145 |

If you want to **generate** connectivity, use
`network.topology`. If you want to **measure** an existing
connectivity matrix, use this module. A doc-level cross-reference
clarifies the distinction; renaming either module would be a
breaking change. Tracked as task #40.

---

## 2. `TopologyReport`

```python
@dataclass
class TopologyReport:
    n_nodes: int = 0
    n_edges: int = 0
    density: float = 0.0
    clustering_coefficient: float = 0.0
    avg_path_length: float = 0.0
    small_world_sigma: float = 0.0
    degree_mean: float = 0.0
    degree_std: float = 0.0
    degree_max: int = 0
    modularity: float = 0.0
    assortativity: float = 0.0
    hub_neurons: list[int] = field(default_factory=list)

    def summary(self) -> str: ...
```

A POD struct populated by `TopologyAnalyzer.analyze()`. The
`summary()` method returns a 4-line human-readable banner that
includes the small-world flag (sigma > 1 → "YES").

**Note:** As of task #42, `modularity` is now populated by
`TopologyAnalyzer.analyze()` via Newman's Q with a connected-
components partition. See §6.1 for the formula and §3.1 for the
algorithm row.

---

## 3. `TopologyAnalyzer`

```python
class TopologyAnalyzer:
    def __init__(self, adjacency: np.ndarray, directed: bool = False):
        self.adj = (np.abs(adjacency) > 1e-10).astype(np.float64)
        np.fill_diagonal(self.adj, 0)
        self.directed = directed
        self.N = self.adj.shape[0]

    def analyze(self) -> TopologyReport: ...
```

The constructor binarises the input adjacency at threshold `1e-10`
(any non-zero entry becomes an edge) and zeros the diagonal (no
self-loops). The matrix is stored as `float64`; the `directed`
flag controls how undirected metrics symmetrise it during
analysis.

### 3.1 What `analyze()` populates

| Field | Algorithm | Cost (N nodes, k = mean degree) |
|-------|-----------|-------------------------------:|
| `n_nodes` | `adj.shape[0]` | O(1) |
| `n_edges` | `int(adj.sum()) // (1 if directed else 2)` | O(N²) |
| `density` | `n_edges / max_edges` | O(1) after edge count |
| `clustering_coefficient` | `_clustering()` — local Watts–Strogatz coefficient averaged over nodes | O(N · k³) |
| `avg_path_length` | `_avg_path_length()` — BFS from up to 100 source nodes | O(min(100, N) · (N + E)) |
| `degree_mean` / `_std` / `_max` | `adj.sum(axis=1)` | O(N²) |
| `hub_neurons` | top-5 by degree, `np.argsort(-degrees)[:5]` | O(N log N) |
| `small_world_sigma` | `(C/C_rand) / (L/L_rand)` per Humphries & Gurney 2008 | O(1) after C and L |
| `assortativity` | `np.corrcoef` over edge endpoints' degrees | O(E) |
| `modularity` | Newman 2006 Q with connected-components partition | O(N²) for the dense matrix step |

`C_rand` and `L_rand` are the Erdős–Rényi expected values:
- `C_rand ≈ k / N`
- `L_rand ≈ ln(N) / ln(k)`

with `k = max(degree_mean, 1)` and divide-by-zero guards for small
graphs. Sigma > 1 indicates small-world structure (high clustering
relative to random, with comparable path length).

### 3.2 Sampled BFS for `avg_path_length` (CONFIGURABLE since task #41)

`_avg_path_length` samples up to `self.n_path_samples` source
nodes rather than running an all-pairs BFS:

```python
cap = self.n_path_samples if self.n_path_samples > 0 else self.N
for src in range(min(self.N, cap)):
    dist = self._bfs(A, src)
    ...
```

The constructor accepts `n_path_samples: int = 100`. For
`N <= n_path_samples` the result is the true all-pairs average;
for larger graphs it samples the first `n_path_samples` source
nodes deterministically. Pass `n_path_samples=0` to disable the
cap and force full all-pairs (expensive at N >> 100).

### 3.3 Directed-graph handling

For `directed=True`, the constructor still zeroes the diagonal and
binarises, but `_clustering` and `_avg_path_length` symmetrise the
matrix internally via `np.maximum(adj, adj.T)`. This means
clustering and path length are computed on the **undirected
projection** even for directed input — a deliberate simplification.
Truly directed metrics (in/out degree, motif counts, weakly vs
strongly connected components) are not provided.

---

## 4. Performance — measured (this workstation)

Hardware: Intel i5-11600K, 32 GB DDR4, Python 3.12.3, NumPy 2.2.6.
Random Erdős–Rényi graph at p=0.1, undirected, single
`analyzer.analyze()` call:

| N | density | `analyze` wall |
|---:|--------:|---------------:|
| 50 | 0.096 | 25.4 ms |
| 200 | 0.098 | 238.4 ms |
| 500 | 0.099 | 1 065.9 ms |

Scaling: roughly `O(N²·k)` where `k = N·p`, dominated by the
`_clustering` per-node sub-graph extraction and the `_bfs` inner
loop on the sampled source nodes. The cap at 100 BFS sources keeps
path-length cost from blowing up but introduces sampling noise
above N ≈ 100.

---

## 5. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.topology import TopologyAnalyzer, TopologyReport` | `topology/__init__.py:10` | `tests/test_graph_topology.py` |
| `TopologyReport.summary()` 4-line banner | direct method call | covered transitively |
| `TopologyAnalyzer.analyze()` populates 11 fields | direct method call | `test_analyze_*` battery |

There is no integration with `sc_neurocore.network.Network` —
callers must extract the connectivity matrix manually from
projections (e.g. via `proj.indptr`, `proj.indices`, `proj.data`)
before constructing the analyser.

---

## 6. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | Both symbols re-exported and tested |
| 2 | Multi-angle tests | ✅ PASS | 7 tests in `test_graph_topology.py` covering construction, density, clustering, path length, small-world flag, hubs, summary |
| 3 | Rust path | ❌ FAIL | Pure Python; `_clustering` and `_bfs` are Python loops. For N ≥ 500 the analysis takes >1 s. Same Rustification scope as the network/topology generators (task #13). |
| 4 | Benchmarks | ✅ PASS | §4 measured this session |
| 5 | Performance docs | ✅ PASS | §4 |
| 6 | Documentation page | ✅ PASS | This page (upgraded from a 21-line stub) |
| 7 | Rules followed | ✅ PASS | SPDX header ✅. `modularity` field now populated by Newman 2006 Q with connected-components partition (closes task #42, see §6.1). `avg_path_length` sample cap is exposed via the `n_path_samples` constructor argument (closes task #41, see §3.2). British English clean. No `# noqa`, no `# type: ignore`. |

Net: **0 WARN, 1 FAIL.** Tasks #41 and #42 both closed in this
session; the remaining FAIL is the absence of a Rust path
(rolled into the broader Rustification effort, task #13).

### 6.1 `modularity` (FIXED by task #42)

`TopologyAnalyzer.analyze()` now populates the field via Newman's Q:

```
Q = (1 / 2m) · Σ_ij [A_ij − (k_i · k_j) / 2m] · δ(c_i, c_j)
```

By default the partition is the connected components of the
(undirected projection of the) graph — each component is its own
community. Users can pass an explicit per-node partition to
`TopologyAnalyzer._modularity(communities=[...])` to score any
candidate split. Reference:
Newman M. E. J. "Modularity and community structure in networks."
*PNAS* 103(23):8577-8582 (2006).

Verified expected values on hand-crafted graphs:
- single complete graph K₅ → Q = 0.0 (no community structure)
- two disjoint K₅ cliques → Q = 0.5 (Newman 2006 example)
- three equal disjoint cliques → Q = 2/3 (theoretical maximum 1 − 1/k)
- empty graph → Q = 0.0
- correct vs wrong vs singleton partitions on the same graph
  ranked correct > wrong > singleton

Algorithm cost is O(N²) for the dense `(A − expected)` step;
acceptable for N ≤ 10³, slow above. The connected-components
labeller is a Python BFS (also O(N²) worst case).

Regression: `tests/test_graph_topology.py::TestModularity` (6
tests).

---

## 7. Known issues

### 7.1 `modularity` (FIXED by task #42)

See §6.1. Field now populated by Newman's Q with connected-
components partition, with caller-overridable communities and 6
regression tests.

### 7.2 `avg_path_length` sample cap (FIXED by task #41)

The cap is now exposed as a constructor argument
(`n_path_samples: int = 100`). Pass `n_path_samples=0` for full
all-pairs (expensive at N >> 100), or any other positive integer
to override the default. See §3.2 and the regression class
`tests/test_graph_topology.py::TestPathSampleCap`.

### 7.3 Naming overlap with `sc_neurocore.network.topology`

Two unrelated modules share the name. New users encountering both
in the auto-API may not realise they do different things. Tracked
as task #40 (rename one, or add a doc-level Sphinx cross-reference
that can't be missed).

### 7.4 Directed-graph metrics use undirected projection

`_clustering` and `_avg_path_length` symmetrise via
`np.maximum(adj, adj.T)` even when `directed=True`. The metrics
for genuinely directed graphs (weakly/strongly connected
components, directed clustering, in/out assortativity) are not
provided. Consider documenting the limitation explicitly in the
class docstring.

### 7.5 `hub_neurons` is hard-coded to top-5

The `hub_neurons` field always contains exactly 5 entries (or
fewer if N < 5). No parameter controls this. For larger networks
where hubs span dozens of nodes, callers must compute their own
ranking from the degree distribution.

---

## 8. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_graph_topology.py -q
# 7 passed in 0.83s (verified 2026-04-17)
```

Coverage: 7 tests covering `TopologyAnalyzer` construction,
`analyze()` end-to-end, density correctness on a known graph,
clustering coefficient on a triangle, path length on a chain,
small-world detection on a Watts-Strogatz example, hub
identification.

Not covered:
- The `modularity = 0.0` dead-field anti-pattern (§6.1)
- The 100-BFS-source approximation behaviour at N > 100 (§7.2)
- Directed-graph metrics on a genuinely directed input (§7.4)
- Edge cases: empty graph, single-node graph, disconnected
  components

---

## 9. References

- Watts D. J., Strogatz S. H. "Collective dynamics of small-world
  networks." *Nature* 393:440-442 (1998). The clustering
  coefficient definition used here.
- Humphries M. D., Gurney K. "Network 'small-world-ness': a
  quantitative method for determining canonical network
  equivalence." *PLoS ONE* 3(4):e0002051 (2008). The
  small-world sigma definition.
- Newman M. E. J. *Networks: An Introduction* (Oxford UP, 2010).
  General reference for assortativity, modularity, BFS path
  length.
- Brandes U. "A faster algorithm for betweenness centrality."
  *J Mathematical Sociology* 25(2):163-177 (2001). Better path
  length algorithm if §7.2 is upgraded.

Internal:

- Connectivity generators (the *other* topology module):
  [`api/network.md`](network.md) §6
- Tutorial 83: see `docs/tutorials/83_topology.md` (existing
  reference from the previous stub version of this page)

---

## 10. Auto-rendered API

::: sc_neurocore.topology.analyzer
    options:
      show_root_heading: true
      show_source: true
      members:
        - TopologyAnalyzer
        - TopologyReport
