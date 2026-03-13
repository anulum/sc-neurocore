<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Online Learning with STDP and Reward-Modulated STDP

Train synapses using spike-timing-dependent plasticity — no
backpropagation, no gradient computation, no external optimizer.
STDP is local, online, and maps directly to hardware.

**Prerequisites**: `pip install sc-neurocore matplotlib`

## 1. STDP basics

Spike-timing-dependent plasticity strengthens synapses when
pre-synaptic activity precedes post-synaptic spikes (LTP)
and weakens them when post fires without recent pre activity (LTD).

Bi & Poo (1998) measured an asymmetric learning window:

- Pre→Post (Δt > 0): potentiation, magnitude decays with Δt
- Post→Pre (Δt < 0): depression, ~50% weaker than LTP

SC-NeuroCore's `StochasticSTDPSynapse` implements this as a
trace-based rule operating on individual bitstream steps.

## 2. Single synapse: observe weight drift

```python
from sc_neurocore import StochasticSTDPSynapse
import numpy as np

syn = StochasticSTDPSynapse(
    w_min=0.0, w_max=1.0, w=0.5,
    learning_rate=0.01, window_size=5, ltd_ratio=0.5,
    length=64, seed=42,
)

# Correlated pre/post → expect potentiation
weights_corr = [syn.w]
for t in range(500):
    pre = 1 if np.random.rand() < 0.3 else 0
    post = pre  # perfect correlation
    syn.process_step(pre, post)
    weights_corr.append(syn.w)

print(f"Correlated: w started {weights_corr[0]:.3f} → ended {weights_corr[-1]:.3f}")
```

With perfect pre→post correlation, the synapse should strengthen
toward `w_max=1.0`.

## 3. Anticorrelated activity: depression

```python
syn_anti = StochasticSTDPSynapse(
    w_min=0.0, w_max=1.0, w=0.5,
    learning_rate=0.01, window_size=5, ltd_ratio=0.5,
    length=64, seed=7,
)

weights_anti = [syn_anti.w]
for t in range(500):
    pre = 1 if np.random.rand() < 0.3 else 0
    post = 1 - pre  # anti-correlated
    syn_anti.process_step(pre, post)
    weights_anti.append(syn_anti.w)

print(f"Anti-correlated: w started {weights_anti[0]:.3f} → ended {weights_anti[-1]:.3f}")
```

Anti-correlated activity drives the weight toward `w_min=0.0`.

## 4. Plot the learning curves

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(weights_corr, label="Correlated (LTP)")
ax.plot(weights_anti, label="Anti-correlated (LTD)")
ax.set_xlabel("Time step")
ax.set_ylabel("Synaptic weight")
ax.set_title("STDP Weight Dynamics")
ax.legend()
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig("stdp_learning_curves.png", dpi=150)
```

## 5. Reward-modulated STDP (R-STDP)

Standard STDP is unsupervised — it finds correlations but has no
notion of task performance. R-STDP adds a third factor: a global
reward signal that gates weight updates via an eligibility trace.

The eligibility trace $e(t)$ accumulates Hebbian coincidences:

$$e(t+1) = \gamma \cdot e(t) + \text{Hebbian}(t) - \alpha \cdot \text{Anti-Hebbian}(t)$$

Weights change only when `apply_reward(R)` is called:

$$\Delta w = \eta \cdot R \cdot e$$

This decouples the *what* (trace accumulation) from the *when*
(reward signal), enabling reinforcement-style learning.

```python
from sc_neurocore import RewardModulatedSTDPSynapse

rsyn = RewardModulatedSTDPSynapse(
    w_min=0.0, w_max=1.0, w=0.5,
    learning_rate=0.01, length=64, seed=42,
)

# Phase 1: accumulate eligibility (no weight change yet)
for _ in range(100):
    rsyn.process_step(pre_bit=1, post_bit=1)

print(f"Eligibility trace: {rsyn.eligibility_trace:.3f}")
print(f"Weight before reward: {rsyn.w:.3f}")

# Phase 2: positive reward → potentiate
rsyn.apply_reward(reward=1.0)
print(f"Weight after +1 reward: {rsyn.w:.3f}")
```

## 6. R-STDP for pattern selection

Reward the synapse when it contributes to a correct output,
punish when it contributes to an error:

```python
rsyn_select = RewardModulatedSTDPSynapse(
    w_min=0.0, w_max=1.0, w=0.5,
    learning_rate=0.005, length=64, seed=42,
)

# Simulate: pre-synaptic neuron fires when "target pattern" is present
# Reward when output matches target, punish otherwise
np.random.seed(42)
weights = [rsyn_select.w]
for trial in range(200):
    target_present = np.random.rand() < 0.5
    pre_rate = 0.8 if target_present else 0.2

    # Run 20 timesteps per trial
    for _ in range(20):
        pre = 1 if np.random.rand() < pre_rate else 0
        post = 1 if np.random.rand() < 0.4 else 0  # fixed post rate
        rsyn_select.process_step(pre, post)

    # Reward: +1 if target present and synapse is strong, -1 otherwise
    output_strength = rsyn_select.w * pre_rate
    if target_present and output_strength > 0.3:
        rsyn_select.apply_reward(1.0)
    elif not target_present and output_strength > 0.3:
        rsyn_select.apply_reward(-0.5)

    weights.append(rsyn_select.w)

print(f"R-STDP final weight: {rsyn_select.w:.3f}")
```

## 7. Comparison: STDP vs R-STDP

| Property | STDP | R-STDP |
|----------|------|--------|
| Learning signal | Spike timing only | Timing + reward |
| Supervision | Unsupervised | Reinforcement |
| Update timing | Every timestep | On reward signal |
| Hardware cost | 2 comparators | 2 comparators + trace register |
| Use case | Feature extraction | Decision making |

## 8. Practical considerations

**Weight bounds**: Both synapse types clamp weights to `[w_min, w_max]`
after every update. Verify bounds hold under extreme conditions:

```python
from sc_neurocore import StochasticSTDPSynapse

syn = StochasticSTDPSynapse(
    w_min=0.0, w_max=1.0, w=0.99,
    learning_rate=0.1, length=64,
)
for _ in range(1000):
    syn.process_step(pre_bit=1, post_bit=1)
assert 0.0 <= syn.w <= 1.0, "Weight escaped bounds"
print(f"Stress test passed: w={syn.w:.4f}")
```

**Bitstream length**: Shorter bitstreams (length=32-64) suffice for
STDP because the synapse samples one bit per step, not the full
stream. The weight bitstream is re-encoded after each update.

**Window size**: `window_size=5` means the synapse remembers the last
5 pre-synaptic bits. Larger windows capture longer-range correlations
but increase memory and false-positive coincidences.

## What you learned

- STDP potentiates correlated synapses, depresses anti-correlated ones
- R-STDP adds a reward-gated eligibility trace for task-directed learning
- Both operate on single bitstream steps — no matrix operations needed
- Weight bounds are enforced at every update
- Window size controls the temporal span of spike-timing detection

## Next steps

- Combine STDP with `SCLearningLayer` for a complete learning layer
- Use R-STDP in a reservoir computing setup (`SCRecurrentLayer` + readout)
- Compare learning speed at different bitstream lengths
