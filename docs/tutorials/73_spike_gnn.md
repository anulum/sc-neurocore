# Tutorial 73: Spike-Native Graph Neural Networks

Graph processing with spike-based message passing.

```python
from sc_neurocore.spike_gnn import SpikeGNNLayer

gnn = SpikeGNNLayer([16, 8, 3], T=8)
node_out = gnn.forward(node_features, adjacency)
predicted_class = gnn.graph_classify(node_features, adjacency)
```
