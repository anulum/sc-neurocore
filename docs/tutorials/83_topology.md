# Tutorial 83: Network Topology Analysis

Analyse the graph structure of your SNN: small-world coefficient,
modularity, centrality, clustering, path lengths. These metrics predict
network dynamics and inform architecture choices — small-world networks
produce richer dynamics than random graphs.

## Why Topology Matters

The connectivity pattern of an SNN determines its computational
properties more than individual neuron parameters:

| Topology | Dynamics | Computation |
|----------|---------|-------------|
| Random (Erdős-Rényi) | Homogeneous, rapid synchrony | Poor memory, poor separation |
| Small-world (Watts-Strogatz) | Rich dynamics, edge-of-chaos | Good memory + separation |
| Scale-free (Barabási-Albert) | Hub-dominated, fragile | High throughput, low robustness |
| Modular | Community structure | Task-specific subnetworks |

SC-NeuroCore uses small-world connectivity by default — it produces
the best balance of local clustering and global integration.

## TopologyAnalyzer

```python
import numpy as np
from sc_neurocore.topology import TopologyAnalyzer

rng = np.random.RandomState(42)

# Random sparse connectivity (100 neurons, 10% connection probability)
W = (rng.random((100, 100)) < 0.1).astype(float)
np.fill_diagonal(W, 0)

analyzer = TopologyAnalyzer(W)
report = analyzer.analyze()

print(f"Nodes: {report.n_nodes}")
print(f"Edges: {report.n_edges}")
print(f"Density: {report.density:.3f}")
print(f"Clustering coefficient: {report.clustering:.3f}")
print(f"Mean path length: {report.mean_path_length:.2f}")
print(f"Small-world sigma: {report.small_world_sigma:.2f}")
print(f"Modularity: {report.modularity:.3f}")
```

## Key Metrics

| Metric | Formula | What It Means | Optimal for SNN |
|--------|---------|:------------:|:---------------:|
| Clustering (C) | Triangle count / triplet count | Local connectivity density | 0.1–0.3 |
| Mean path (L) | Average shortest-path hops | Global integration | 2–4 hops |
| Small-world (σ) | (C/C_rand) / (L/L_rand) | σ > 1 = small-world | 1.5–5.0 |
| Modularity (Q) | Community structure strength | Q > 0.3 = modular | 0.3–0.6 |
| Degree centrality | Connections / (N-1) | Hub detection | Uniform (no hubs) |
| Betweenness | Fraction of shortest paths through node | Bottleneck detection | Low max |

## Small-World Analysis

Networks with high clustering AND short path lengths are "small-world."
This is the optimal regime for spiking networks:

```python
# Generate small-world connectivity
from sc_neurocore.network.topology import generate_connectivity

# Watts-Strogatz small-world
W_sw = generate_connectivity(100, topology="small_world", p_rewire=0.1, k=10)
report_sw = TopologyAnalyzer(W_sw).analyze()

# Compare with random
W_rand = generate_connectivity(100, topology="random", p=0.1)
report_rand = TopologyAnalyzer(W_rand).analyze()

print(f"Small-world: C={report_sw.clustering:.3f}, L={report_sw.mean_path_length:.2f}, "
      f"σ={report_sw.small_world_sigma:.2f}")
print(f"Random:      C={report_rand.clustering:.3f}, L={report_rand.mean_path_length:.2f}, "
      f"σ={report_rand.small_world_sigma:.2f}")
```

## Modularity Detection

Find functional communities in your network:

```python
communities = analyzer.detect_communities()
print(f"Communities found: {len(communities)}")
for i, comm in enumerate(communities):
    print(f"  Community {i}: {len(comm)} neurons")
```

## Integration with Studio

The Network Canvas shows populations and projections. The topology
analyser operates on the underlying weight matrix — use it to verify
that your designed connectivity has the desired graph properties.

## References

- Watts & Strogatz (1998). "Collective dynamics of 'small-world'
  networks." Nature 393:440-442.
- Barabási & Albert (1999). "Emergence of Scaling in Random Networks."
  Science 286:509-512.
- Sporns (2011). "Networks of the Brain." MIT Press.
