# Spike GNN — Graph Neural Networks with Spike Messages

Graph neural networks where messages are spike trains instead of float vectors. Nodes are spiking neuron populations. Aggregation via normalized neighborhood summation, followed by LIF integration.

## Architecture

Each `SpikeGraphConv` layer performs:

1. **Message passing:** `h_agg = (A @ X) / deg` — aggregate neighbor features
2. **Linear projection:** `h_proj = h_agg @ W^T` — learned weight transform
3. **LIF integration:** Over T timesteps, rate-coded input drives LIF neurons. Output = spike counts per node.

`SpikeGNNLayer` stacks multiple `SpikeGraphConv` layers with inter-layer spike count normalization.

The implementation is deterministic for a fixed layer seed. `SpikeGraphConv.forward`
normalizes each node's neighbor aggregate by clipped in-degree, projects through
the learned weight matrix, runs a rate-coded LIF loop for `T` timesteps, and
returns non-negative spike counts with shape `(n_nodes, out_features)`.

## Components

- **`SpikeGraphConv`** — Single spike-based graph convolution layer.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `in_features` | (required) | Input dimension per node |
| `out_features` | (required) | Output dimension per node |
| `threshold` | 1.0 | LIF spike threshold |
| `tau_mem` | 10.0 | Membrane time constant |

- **`SpikeGNNLayer`** — Multi-layer spike GNN for graph classification.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `layer_dims` | (required) | [in, hidden, ..., out] dimensions |
| `threshold` | 1.0 | LIF threshold for all layers |
| `T` | 8 | Simulation timesteps per layer |

Methods: `forward(node_features, adjacency)`, `graph_classify(node_features, adjacency)`.

## Usage

```python
from sc_neurocore.spike_gnn.spike_gnn import SpikeGraphConv, SpikeGNNLayer
import numpy as np

# Single layer
conv = SpikeGraphConv(in_features=16, out_features=8)
adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
features = np.random.rand(3, 16)
output = conv.forward(features, adj, T=8)  # (3, 8) spike counts

# Multi-layer graph classifier
gnn = SpikeGNNLayer(layer_dims=[16, 8, 4], threshold=1.0, T=8)
label = gnn.graph_classify(features, adj)
print(f"Predicted class: {label}")
```

**Reference:** SGNNBench (2025) — 9 SGNN architectures benchmarked.

See [Tutorial 73: Spike GNN](../tutorials/73_spike_gnn.md).

::: sc_neurocore.spike_gnn
    options:
      show_root_heading: true
