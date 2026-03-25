# Tutorial 66: SNN Compiler Optimization Passes

Optimize SNN computation graphs before hardware deployment.

## Build a Graph

```python
from sc_neurocore.snn_optimizer import SNNGraph, LayerNode, optimize

graph = SNNGraph(layers=[
    LayerNode("hidden1", 784, 256, weights_h1, firing_rates=rates_h1),
    LayerNode("hidden2", 256, 64, weights_h2, firing_rates=rates_h2),
    LayerNode("output", 64, 10, weights_out, firing_rates=rates_out),
])
```

## Run All Passes

```python
optimized, report = optimize(graph)
print(report.summary())
# SNN Optimizer: 215,104 -> 142,890 params (1.51x compression)
```

## Available Passes

| Pass | What It Does |
|------|-------------|
| dead_neuron_elimination | Remove neurons that never fire |
| layer_fusion | Merge adjacent silent layers |
| redundancy_elimination | Merge neurons with identical weights |

## Selective Passes

```python
optimized, report = optimize(graph, passes=["dead_neuron_elimination"])
```
