# Tutorial 61: SNN Explainability — Why Did It Fire?

Three methods to explain SNN decisions at the spike level.

## Spike Attribution

Trace which input spikes contributed most to the output:

```python
from sc_neurocore.explain import SpikeAttributor

attr = SpikeAttributor(decay=0.9)
result = attr.attribute(input_spikes, weights, output_neuron=0)

# Top 5 most important spikes
for t, n, score in result.top_k(5):
    print(f"  t={t}, neuron={n}: importance={score:.3f}")
```

## Perturbation Saliency

Remove each spike and measure output change:

```python
from sc_neurocore.explain import TemporalSaliency

sal = TemporalSaliency(run_fn=my_model)
result = sal.explain(input_spikes, output_neuron=0)
print(result.summary())
```

## Causal Importance

Silence each neuron and measure classification impact:

```python
from sc_neurocore.explain import CausalImportance

ci = CausalImportance(run_fn=my_model)
result = ci.explain(input_spikes, output_neuron=0)
# result.importance_map[0] = per-neuron importance scores
```

## Comparison

| Method | What It Shows | Speed | Granularity |
|--------|--------------|-------|-------------|
| Attribution | Weight-path importance | Fast (one pass) | Per spike per timestep |
| Saliency | Output sensitivity | Slow (N perturbations) | Per spike per timestep |
| Causal | Neuron-level impact | Medium (N interventions) | Per neuron |
