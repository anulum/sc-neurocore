# AI-Optimized Neuron Models

Nine novel neuron models designed for AI workloads, not biological simulation.

**Modules**: `sc_neurocore.neurons.models.arcane_neuron`, `sc_neurocore.neurons.models.ai_optimized`

## ArcaneNeuron

Five coupled subsystems in a single coherent ODE:

1. **FAST** (tau=5ms): spike timing, immediate sensory processing
2. **WORKING** (tau=200ms): working memory via sustained activity
3. **DEEP** (tau=10s): long-term context accumulation, personality drift
4. **GATE**: learned attention over inputs, modulated by confidence
5. **PREDICTOR**: forward model of own future state, fires on surprise

The deep compartment accumulates identity: it changes only when the
neuron encounters genuine novelty (prediction errors), not routine
input. Confidence modulates the threshold (confident = lower threshold
= faster responses) and the learning rate (uncertain = learn faster).

```python
from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron

neuron = ArcaneNeuron()
for t in range(1000):
    spike = neuron.step(current=0.5)

print(neuron.identity_state)      # deep compartment value
print(neuron.confidence)           # 1 - mean(novelty_history)
print(neuron.meta_learning_rate)   # lr_base * (1 + eta * novelty)
```

Rust implementation: `ArcaneNeuron` in `network_runner.rs` (included
in the 111-model NetworkRunner).

## AI-Optimized Models

| Model | Equation Summary | Use Case |
|-------|-----------------|----------|
| `MultiTimescaleNeuron` | Three-compartment (fast/medium/slow), slow compartment modulates threshold | Temporal context accumulation |
| `AttentionGatedNeuron` | Sigmoid gate with learned key/query weights | Selective input filtering |
| `PredictiveCodingNeuron` | Fires only on prediction errors | Novelty detection |
| `SelfReferentialNeuron` | Introspects spike history, auto-regulates dynamics | Stable autonomous firing |
| `CompositionalBindingNeuron` | Phase-coding with amplitude, in-phase = bound | Variable binding |
| `DifferentiableSurrogateNeuron` | Trainable alpha/beta/theta for surrogate gradients | SNN training |
| `ContinuousAttractorNeuron` | Ring attractor, Mexican-hat connectivity | Continuous working memory |
| `MetaPlasticNeuron` | Error-trace-driven meta-learning rate | Adaptive learning speed |

All models share the standard `step(current) -> int` / `reset()` interface.

```python
from sc_neurocore.neurons.models.ai_optimized import (
    MultiTimescaleNeuron,
    AttentionGatedNeuron,
    PredictiveCodingNeuron,
)

mtn = MultiTimescaleNeuron()
for t in range(200):
    spike = mtn.step(current=0.8)
print(f"Slow compartment: {mtn.v_slow:.4f}")
```
