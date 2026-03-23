<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 34: ArcaneNeuron — Self-Referential Cognition

ArcaneNeuron is the flagship AI-optimized neuron model in SC-NeuroCore. It couples
five subsystems in a single coherent ODE: fast processing, working memory, deep
context accumulation, learned attention gating, and a forward self-model (predictor).
No equivalent exists in any other toolkit.

## The Five Compartments

```
Input ──► GATE ──► FAST (tau=5ms) ──► WORKING (tau=200ms) ──► DEEP (tau=10s)
            ↑                                                       │
            └──── CONFIDENCE ◄──── NOVELTY ◄──── PREDICTOR ◄───────┘
```

| Compartment | Time constant | Function |
|-------------|---------------|----------|
| **Fast** | 5 ms | Spike timing, immediate sensory processing |
| **Working** | 200 ms | Working memory via sustained activity |
| **Deep** | 10 s | Long-term context, personality drift, identity |
| **Gate** | learned | Selective attention over inputs |
| **Predictor** | learned | Forward model of own future state |

## Why It Matters

Traditional neuron models (LIF, HH, Izhikevich) are memoryless — they process
input without context. ArcaneNeuron's deep compartment accumulates identity:
it changes only when the neuron encounters genuine novelty (prediction errors),
not routine input. This means:

- **Confidence modulates threshold**: confident neuron = lower threshold = faster responses
- **Novelty modulates learning**: uncertain neuron = higher learning rate = faster adaptation
- **The gate prevents irrelevant input** from reaching the fast compartment
- **The predictor models the neuron's own behavior** — when reality deviates from prediction, the neuron knows it encountered something new

## 1. Basic Usage

```python
from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron

neuron = ArcaneNeuron()

# Step with constant input
for t in range(2000):
    spike = neuron.step(current=0.8)
    if spike:
        state = neuron.get_state()
        print(f"  t={t}: spike, confidence={neuron.confidence():.3f}, "
              f"novelty={neuron.novelty():.3f}, deep={neuron.identity_state():.4f}")
```

## 2. Observing Identity Accumulation

The deep compartment changes slowly — only on genuine novelty:

```python
neuron = ArcaneNeuron()

# Phase 1: routine input (constant)
for t in range(5000):
    neuron.step(current=0.5)
deep_after_routine = neuron.identity_state()

# Phase 2: novel input (sudden change)
for t in range(5000):
    neuron.step(current=2.0)
deep_after_novel = neuron.identity_state()

# Phase 3: return to routine
for t in range(5000):
    neuron.step(current=0.5)
deep_after_return = neuron.identity_state()

print(f"Deep after routine: {deep_after_routine:.4f}")
print(f"Deep after novel:   {deep_after_novel:.4f}")  # changed
print(f"Deep after return:  {deep_after_return:.4f}")  # retains change
```

The deep compartment retains the effect of novel experiences even after the input
returns to baseline. This is the mechanism for identity persistence.

## 3. Confidence and Threshold Dynamics

```python
neuron = ArcaneNeuron()

# Predictable input → confidence rises → threshold drops → faster firing
for t in range(3000):
    neuron.step(current=0.7)

confident_threshold = neuron.get_state()["effective_threshold"]
confident_rate = neuron.confidence()

# Unpredictable input → confidence falls → threshold rises → cautious firing
import numpy as np
for t in range(3000):
    neuron.step(current=np.random.uniform(0, 2.0))

uncertain_threshold = neuron.get_state()["effective_threshold"]
uncertain_rate = neuron.confidence()

print(f"After predictable: confidence={confident_rate:.3f}, threshold={confident_threshold:.3f}")
print(f"After random:      confidence={uncertain_rate:.3f}, threshold={uncertain_threshold:.3f}")
```

## 4. Meta-Learning Rate

The learning rate increases when the neuron encounters surprise:

```python
neuron = ArcaneNeuron()

# Stable regime
for _ in range(2000):
    neuron.step(current=0.5)
stable_lr = neuron.meta_learning_rate()

# Inject surprise
for _ in range(100):
    neuron.step(current=5.0)
surprised_lr = neuron.meta_learning_rate()

print(f"Stable learning rate:    {stable_lr:.4f}")
print(f"Surprised learning rate: {surprised_lr:.4f}")  # higher
```

## 5. The Core Equations

```
gate = sigmoid(w_g @ [I, v_fast, v_work, confidence])
I_eff = gate * I

dv_fast/dt = (-v_fast + I_eff - w_inh * spike_rate) / tau_fast
dv_work/dt = (-v_work + alpha_w * v_fast * spike) / tau_work
dv_deep/dt = (-v_deep + alpha_d * v_work * novelty) / tau_deep

prediction = w_pred @ [v_fast, v_work, v_deep]
surprise = |v_fast - prediction|
novelty = sigmoid(kappa * (surprise - baseline))

confidence = 1 - mean(novelty_history)
threshold_eff = theta * (1 + gamma * v_deep) * (1 - delta * confidence)

meta_lr = lr_base * (1 + eta * novelty)

spike when v_fast >= threshold_eff
```

## 6. Comparison with Other Models

| Property | LIF | Izhikevich | HH | ArcaneNeuron |
|----------|-----|------------|-----|-------------|
| State variables | 1 | 2 | 4 | 5+ |
| Memory | None | Reset | None | Deep compartment |
| Attention | No | No | No | Learned gate |
| Self-model | No | No | No | Predictor |
| Confidence | No | No | No | Yes |
| Adaptive learning | No | No | No | Meta-learning rate |
| Identity | No | No | No | Deep context (tau=10s) |

## 7. Using in Networks

ArcaneNeuron works with the Population-Projection-Network engine:

```python
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor

pop = Population(ArcaneNeuron, n=50, label="arcane")
mon = SpikeMonitor(pop)

net = Network(pop, mon)
net.run(duration=2.0, dt=0.001)
print(f"Spikes: {mon.count()}")
```

## Further Reading

- [API: AI-Optimized Neurons](../api/ai_neurons.md) — all 9 AI neuron models
- [Tutorial 32: Identity Substrate](32_identity_substrate.md) — persistent SNN using ArcaneNeuron
- [Tutorial 17: Custom Neuron Models](17_custom_neuron_models.md) — build your own
- Reference: Šotek & Arcane Sapience 2026 (original design)
