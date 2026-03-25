# Spike GNN

Spike-based graph neural network: message passing with spike trains instead of float vectors.

- `SpikeGNNLayer` — Graph convolution where messages are spike trains. Neighborhood aggregation via spike-domain summation. Supports temporal spike patterns, not just rates.

Works with any graph topology (sparse adjacency matrix). Compatible with event-driven simulation for O(spikes) computation.

```python
from sc_neurocore.spike_gnn import SpikeGNNLayer
```

See [Tutorial 73: Spike GNN](../tutorials/73_spike_gnn.md).

::: sc_neurocore.spike_gnn
    options:
      show_root_heading: true
