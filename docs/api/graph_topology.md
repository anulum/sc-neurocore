# Network Topology Analysis

Graph metrics for SNN connectivity structure.

- `TopologyAnalyzer` — Computes clustering coefficient, mean path length, small-world sigma, modularity, degree distribution from a connectivity matrix.
- `TopologyReport` — Dataclass with all computed metrics: `n_nodes`, `n_edges`, `density`, `clustering`, `mean_path_length`, `small_world_sigma`.

Small-world networks (sigma > 1) are common in biological neural circuits and tend to produce better SNN dynamics.

```python
from sc_neurocore.topology import TopologyAnalyzer
```

See [Tutorial 83: Network Topology Analysis](../tutorials/83_topology.md) for usage examples.

::: sc_neurocore.topology.analyzer
    options:
      show_root_heading: true
      members:
        - TopologyAnalyzer
        - TopologyReport
