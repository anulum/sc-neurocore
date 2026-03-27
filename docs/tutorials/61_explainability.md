# Tutorial 61: SNN Explainability — Why Did It Fire?

Spiking neural networks make decisions through temporal spike patterns.
Understanding *which spikes at which times* drove a classification is
harder than in standard ANNs because the temporal dimension adds
complexity. SC-NeuroCore provides three explanation methods at
different granularities.

## Three Explanation Methods

| Method | What It Shows | Speed | Granularity |
|--------|--------------|-------|-------------|
| **Spike Attribution** | Weight-path importance | Fast (one pass) | Per spike × per timestep |
| **Temporal Saliency** | Output sensitivity to each spike | Slow (N perturbations) | Per spike × per timestep |
| **Causal Importance** | Neuron-level impact | Medium (N interventions) | Per neuron |

## Spike Attribution

Trace the contribution of each input spike through the network's weight
paths to the output. Fast — requires only one forward pass with trace
recording:

```python
import numpy as np
from sc_neurocore.explain import SpikeAttributor

attr = SpikeAttributor(decay=0.9)

# Input: 100 timesteps, 64 neurons
rng = np.random.default_rng(42)
input_spikes = (rng.random((100, 64)) < 0.1).astype(float)

# Weights: 2-layer network
weights = [
    rng.standard_normal((64, 32)).astype(np.float32) * 0.3,
    rng.standard_normal((32, 10)).astype(np.float32) * 0.3,
]

result = attr.attribute(input_spikes, weights, output_neuron=0)

# Top 5 most important input spikes
print("Most important spikes for output neuron 0:")
for t, n, score in result.top_k(5):
    print(f"  t={t:>3d}, input neuron={n:>2d}: importance={score:.4f}")

# Temporal profile: which timesteps matter most
temporal_importance = result.temporal_profile()
peak_time = np.argmax(temporal_importance)
print(f"Peak importance at t={peak_time}")
```

### How Attribution Works

At each timestep, the attribution propagates backward through weights:

```
attribution[t, i] = sum_j(w[i,j] * spike[t,i] * decay^(T-t))
```

Recent spikes matter more (exponential decay). Spikes through
high-weight paths matter more. The result is a per-spike importance
score that sums to the total output activation.

## Temporal Saliency

Remove each spike one at a time and measure the output change. More
accurate than attribution but slower — requires N perturbation runs:

```python
from sc_neurocore.explain import TemporalSaliency

def my_model(spikes):
    """Your SNN forward pass — returns output spike counts."""
    return np.random.rand(10)  # replace with actual model

sal = TemporalSaliency(run_fn=my_model)
result = sal.explain(input_spikes, output_neuron=0)

print(result.summary())
# Total spikes tested: 640
# Significant spikes: 45 (7.0%)
# Max saliency: 0.234 at (t=42, neuron=17)
# Mean saliency: 0.018

# Visualise as heatmap
saliency_map = result.saliency_map()  # shape: (100, 64)
```

### When to Use Saliency vs Attribution

- **Attribution** is fast but approximate (linear weight paths only)
- **Saliency** is exact but slow (one perturbation per spike)
- Use attribution for quick exploration, saliency for publication figures

## Causal Importance

Silence each neuron entirely (across all timesteps) and measure
classification accuracy change. Identifies which neurons are critical
for the network's decision:

```python
from sc_neurocore.explain import CausalImportance

ci = CausalImportance(run_fn=my_model)
result = ci.explain(input_spikes, output_neuron=0)

# Per-neuron importance scores
importance = result.importance_map()  # shape: (n_neurons,)
critical = np.argsort(importance)[-5:]
print(f"Top 5 critical neurons: {critical}")
print(f"Their importance scores: {importance[critical]}")

# Robustness: how many neurons can be silenced before accuracy drops 5%
print(f"Robustness: can silence {result.robustness_count(threshold=0.05)} neurons")
```

## Practical Example: MNIST Explanation

```python
# After training an MNIST SNN:
# 1. Run a test digit through the network
# 2. Attribute: which input pixels (spikes) drove the classification
# 3. Visualise: overlay importance on the original digit image

from sc_neurocore.explain import SpikeAttributor

attr = SpikeAttributor(decay=0.95)

# test_spikes: (25, 784) — 25 timesteps, 784 input pixels
result = attr.attribute(test_spikes, model_weights, output_neuron=predicted_class)

# Reshape importance to 28x28 image
pixel_importance = result.spatial_importance()  # sum over time
importance_image = pixel_importance.reshape(28, 28)
# Now visualise with matplotlib — high-importance pixels show where
# the network "looked" to make its decision
```

## FPGA Deployment

Attribution can run on-chip for real-time explanations:

```python
# Export attribution logic for FPGA
# Each synapse has a trace register that accumulates importance
# Final trace values are the per-input importance scores
attr.export_fpga("explain_ice40.v", target="ice40")
```

## Comparison

| Feature | SC-NeuroCore | snnTorch | Norse | Captum |
|---------|:-----------:|:--------:|:-----:|:------:|
| Spike attribution | Yes | No | No | ANN only |
| Temporal saliency | Yes | No | No | ANN only |
| Causal importance | Yes | No | No | ANN only |
| Per-spike granularity | Yes | — | — | No (per-sample) |
| FPGA-compatible | Yes | — | — | No |

## References

- Binder et al. (2016). "Layer-wise Relevance Propagation for Neural
  Networks with Local Renormalization Layers." ICANN 2016.
- Kim et al. (2022). "Exploring Temporal Information Dynamics in Spiking
  Neural Networks." AAAI 2022.
- Fang et al. (2023). "SpikingJelly Interpretability Toolkit."
