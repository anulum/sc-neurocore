# Tutorial 83: Network Topology Analysis

Graph metrics for SNN connectivity: small-world coefficient, modularity,
centrality, clustering, path lengths.

## TopologyAnalyzer

```python
import numpy as np
from sc_neurocore.topology import TopologyAnalyzer

rng = np.random.RandomState(42)
W = (rng.random((100, 100)) < 0.1).astype(float)
np.fill_diagonal(W, 0)

analyzer = TopologyAnalyzer(W)
report = analyzer.analyze()

print(f"Nodes: {report.n_nodes}")
print(f"Edges: {report.n_edges}")
print(f"Clustering: {report.clustering:.3f}")
print(f"Mean path length: {report.mean_path_length:.2f}")
print(f"Small-world sigma: {report.small_world_sigma:.2f}")
```

## Metrics

| Metric | Meaning | Typical SNN Value |
|--------|---------|-------------------|
| Clustering coefficient | Local connectivity density | 0.1-0.3 |
| Mean path length | Average hops | 2-4 |
| Small-world sigma | >1 = small-world | 1.5-5.0 |

Small-world networks produce better SNN dynamics (high clustering + short paths).

## API Reference

::: sc_neurocore.topology
    options:
      show_root_heading: true
