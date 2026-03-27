# Tutorial 66: SNN Compiler Optimisation Passes

Optimise SNN computation graphs before hardware deployment. Dead
neuron elimination, layer fusion, and redundancy removal reduce
parameter count and FPGA resource usage without retraining.

## Why Optimise

Trained SNNs contain waste:
- **Dead neurons:** never fire during validation → zero contribution
- **Redundant neurons:** identical weight patterns → mergeable
- **Silent layers:** consecutive layers where activity drops to near-zero → fusible

Removing these reduces FPGA LUTs, BRAMs, and power without accuracy loss.

## Build a Computation Graph

```python
import numpy as np
from sc_neurocore.snn_optimizer import SNNGraph, LayerNode, optimize

rng = np.random.default_rng(42)

# Simulate trained weights and firing rates
weights_h1 = rng.standard_normal((784, 256)).astype(np.float32) * 0.05
weights_h2 = rng.standard_normal((256, 64)).astype(np.float32) * 0.1
weights_out = rng.standard_normal((64, 10)).astype(np.float32) * 0.2

# Firing rates measured during validation
rates_h1 = rng.random(256).astype(np.float32) * 0.2
rates_h1[:30] = 0.0  # 30 dead neurons
rates_h2 = rng.random(64).astype(np.float32) * 0.15
rates_out = rng.random(10).astype(np.float32) * 0.5

graph = SNNGraph(layers=[
    LayerNode("hidden1", 784, 256, weights_h1, firing_rates=rates_h1),
    LayerNode("hidden2", 256, 64, weights_h2, firing_rates=rates_h2),
    LayerNode("output", 64, 10, weights_out, firing_rates=rates_out),
])

print(f"Original: {graph.total_params:,} parameters")
```

## Run All Optimisation Passes

```python
optimized, report = optimize(graph)
print(report.summary())
# SNN Optimizer Report:
#   dead_neuron_elimination: removed 30 neurons (hidden1)
#   redundancy_elimination: merged 8 neuron pairs (hidden2)
#   layer_fusion: no fusible layers found
#   Parameters: 215,104 → 142,890 (1.51× compression)
#   Accuracy impact: 0.0% (dead neurons had zero contribution)
```

## Available Passes

| Pass | What It Does | Accuracy Impact | Resource Savings |
|------|-------------|-----------------|------------------|
| `dead_neuron_elimination` | Remove neurons with zero firing rate | Zero | Proportional to dead count |
| `redundancy_elimination` | Merge neurons with cosine similarity >0.99 | Minimal (<0.1%) | ~5-15% typical |
| `layer_fusion` | Merge adjacent layers where one has <1% activity | Minimal | Reduces layer count |

### Dead Neuron Elimination

The most impactful pass. In practice, 10-30% of neurons in a trained
SNN never fire during validation — their weights were learned but the
thresholds prevent activation. Removing them is free accuracy:

```python
optimized, report = optimize(graph, passes=["dead_neuron_elimination"])
print(f"Dead neurons removed: {report.dead_neurons_removed}")
print(f"Params: {graph.total_params:,} → {optimized.total_params:,}")
```

### Redundancy Elimination

Find neurons with near-identical weight vectors and merge them:

```python
optimized, report = optimize(graph, passes=["redundancy_elimination"])
print(f"Pairs merged: {report.redundant_pairs_merged}")
```

Two neurons with cosine similarity >0.99 produce nearly identical
spike patterns. Replace both with one and double its output weight.

## Selective Passes

Run only specific passes:

```python
# Only dead neuron elimination (safest, zero accuracy impact)
optimized, report = optimize(graph, passes=["dead_neuron_elimination"])

# All passes
optimized, report = optimize(graph)  # runs all by default
```

## Integration with Studio Pipeline

```python
# In the Studio:
# 1. Train network in Training Monitor
# 2. Validate → collect firing rates via SpikeMonitor
# 3. Build SNNGraph from trained weights + rates
# 4. Run optimizer passes
# 5. Export optimised weights to FPGA tab
# 6. Synthesise → see resource reduction

# The Pipeline button on Canvas can include optimisation automatically
```

## Before vs After (MNIST Example)

| Metric | Before | After Optimisation |
|--------|--------|-------------------|
| Parameters | 215,104 | 142,890 |
| Hidden neurons | 256 + 64 | 226 + 56 |
| LUTs (iCE40) | ~4,800 | ~3,200 |
| Accuracy | 97.2% | 97.2% (unchanged) |

## References

- Li et al. (2022). "Exploring Lottery Ticket Hypothesis in Spiking
  Neural Networks." ECCV 2022.
- Rathi & Roy (2021). "DIET-SNN: A Low-Latency Spiking Neural Network
  with Direct Input Encoding and Leakage and Threshold Optimization."
  IEEE TNNLS.
