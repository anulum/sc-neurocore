# Tutorial 73: Spike-Native Graph Neural Networks

Process graph-structured data with spike-based message passing. Unlike
float-based GNNs where messages are continuous vectors, here messages
are spike trains — enabling event-driven, power-proportional computation
on neuromorphic hardware.

## Why Spiking GNNs

Standard GNNs (GCN, GAT, GraphSAGE) use dense matrix operations.
Spiking GNNs replace these with spike-based message passing where
computation scales with activity, not graph size:

| Property | Float GNN | Spiking GNN |
|----------|-----------|-------------|
| Message type | Dense vector | Spike train |
| Compute per edge | O(d) multiply-adds | O(spikes) additions |
| Power | Constant per step | Proportional to activity |
| Hardware | GPU (dense) | Neuromorphic / FPGA (sparse) |

For sparse, event-driven graphs (sensor networks, social media streams,
molecular dynamics), spiking GNNs can be orders of magnitude more
energy-efficient.

## SpikeGNNLayer

```python
import numpy as np
from sc_neurocore.spike_gnn import SpikeGNNLayer

# 20-node graph with random sparse connectivity
rng = np.random.default_rng(42)
adj = (rng.random((20, 20)) > 0.7).astype(float)
np.fill_diagonal(adj, 0)

# Node features: 16 dimensions per node
features = rng.random((20, 16)).astype(np.float32)

# GNN layer stack: 16 → 8 → 3 (node classification)
gnn = SpikeGNNLayer(
    layer_dims=[16, 8, 3],
    T=8,              # simulation timesteps per message round
    threshold=1.0,
    tau_mem=20.0,     # membrane time constant
)

# Forward pass: spike-based message passing
node_output = gnn.forward(features, adj)
print(f"Input:  {features.shape}")    # (20, 16)
print(f"Output: {node_output.shape}")  # (20, 3) — per-node class scores

# Graph-level classification (global pooling over nodes)
predicted_class = gnn.graph_classify(features, adj)
print(f"Graph class: {predicted_class}")
```

## How Spike Message Passing Works

Each message-passing round proceeds in four steps:

1. **Encode:** Each node converts its feature vector to spike trains
   via rate coding. Feature value 0.8 → 80% firing rate over T steps.

2. **Propagate:** Spikes travel along edges (adjacency matrix). Each
   edge has a learnable weight that scales the incoming spike.

3. **Aggregate:** At each target node, incoming spikes from all
   neighbours are summed into a membrane potential (LIF dynamics).

4. **Decode:** Output spike counts per node form the next-layer input
   or final prediction.

```
For each message round:
  for each node v:
    membrane[v] = 0
    for each neighbour u of v:
      membrane[v] += weight[u,v] * spikes[u]
    if membrane[v] > threshold:
      spikes[v] = 1
      membrane[v] = reset
```

Computation is O(active_spikes × edges), not O(nodes × features).
For sparse graphs with low firing rates, this yields massive speedups.

## Multi-Layer GNN

Stack multiple message-passing rounds for deeper graph reasoning:

```python
# 3-layer GNN: 16 → 32 → 16 → 3
gnn = SpikeGNNLayer(
    layer_dims=[16, 32, 16, 3],
    T=8,
    threshold=1.0,
)

# Each layer performs one round of spike message passing
# The receptive field grows with depth: layer k sees k-hop neighbours
output = gnn.forward(features, adj)
```

## Applications

| Application | Graph Structure | Why Spiking |
|-------------|----------------|-------------|
| Sensor networks | Spatial proximity | Event-driven sensors + event-driven processing |
| Molecular property prediction | Atom bonds | Low activity = low power on edge devices |
| Social network analysis | Follower edges | Sparse updates, not every user active |
| Traffic prediction | Road segments | Event-driven: process only changed segments |
| Point cloud classification | k-NN graph | DVS camera output is already spike-based |

## FPGA Deployment

The spike-based message passing maps naturally to AER (Address-Event
Representation) routing on FPGA:

```python
# Export graph + weights for FPGA
nir = gnn.to_nir(adj)  # NIR format with graph topology

# In the Studio:
# 1. Import NIR on the Network Canvas
# 2. Populations = graph nodes, Projections = edges
# 3. Pipeline → ice40 for synthesis
```

## Comparison

| Feature | SC-NeuroCore | DGL | PyG | SpikeGCL |
|---------|:-----------:|:---:|:---:|:--------:|
| Spike-based messages | Yes | No | No | Yes |
| LIF neuron aggregation | Yes | No | No | Yes |
| FPGA deployment | Yes | No | No | No |
| Event-driven compute | Yes | No | No | Partial |
| Temporal message passing | Yes | No | No | Yes |

## References

- Zhu et al. (2022). "Spiking Graph Convolutional Networks." IJCAI 2022.
- Li et al. (2023). "SpikingGCN: Efficient and Accurate Spiking Graph
  Convolutional Networks." Neural Networks 167:519-531.
- Xu et al. (2021). "Exploiting Spiking Dynamics with Spatial-Temporal
  Feature Normalization in Graph Learning." IJCAI 2021.
