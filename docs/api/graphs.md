# Graphs

Event-based graph neural network layer for spike-graph message passing.

- `StochasticGraphLayer` — GNN convolution where message passing happens via bitstreams. Takes adjacency matrix + per-node feature vectors, propagates through graph structure with SC arithmetic. Supports variable-topology graphs.

```python
import numpy as np
from sc_neurocore.graphs import StochasticGraphLayer

adj = (np.random.rand(20, 20) > 0.7).astype(float)
np.fill_diagonal(adj, 0)
layer = StochasticGraphLayer(adj_matrix=adj, n_features=16)
features = np.random.rand(20, 16)
output = layer.forward(features)
```

::: sc_neurocore.graphs.gnn
    options:
      show_root_heading: true
