# Tutorial 73: Spike-Native Graph Neural Networks

Graph processing with spike-based message passing. Unlike float-based GNNs,
messages are spike trains — enabling event-driven, power-proportional
computation on neuromorphic hardware.

## SpikeGNNLayer

```python
import numpy as np
from sc_neurocore.spike_gnn import SpikeGNNLayer

# 20-node graph with random connectivity
adj = (np.random.rand(20, 20) > 0.7).astype(float)
np.fill_diagonal(adj, 0)

# Node features: 16-dim per node
features = np.random.rand(20, 16)

# GNN layer: 16 -> 8 -> 3 (node classification)
gnn = SpikeGNNLayer([16, 8, 3], T=8)
node_out = gnn.forward(features, adj)
# shape: (20, 3) — per-node output

# Graph-level classification
predicted_class = gnn.graph_classify(features, adj)
```

## How Message Passing Works

1. Each node encodes its features as spike trains
2. Spikes propagate along edges (adjacency matrix)
3. Neighborhood aggregation via spike-domain summation
4. Output is spike rate vector per node

Computation is O(spikes * edges), not O(nodes * features). Sparse graphs
with low firing rates get massive speedups vs dense float GNNs.

## API Reference

::: sc_neurocore.spike_gnn
    options:
      show_root_heading: true
