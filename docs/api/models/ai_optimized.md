# AI-Optimized Spiking Neuron Models

**Module:** `sc_neurocore.neurons.models.ai_optimized`
**Reference:** Original SC-NeuroCore designs (Šotek 2024–2026)
**Family:** Novel architectures designed for AI workloads, not biological simulation
**Models:** 8 distinct neuron types in one module

---

## Overview

This module contains **8 original neuron model designs** created for AI and
machine learning applications. Unlike the biophysical models (HH, AdEx, etc.)
which reproduce experimental data from specific cell types, these models are
designed from first principles to solve computational problems:

1. **MultiTimescaleNeuron** — 3-compartment memory for temporal context
2. **AttentionGatedNeuron** — learned sigmoid gate for input selection
3. **PredictiveCodingNeuron** — fires only on prediction errors
4. **SelfReferentialNeuron** — introspects on own spike history
5. **CompositionalBindingNeuron** — phase coding for variable binding
6. **DifferentiableSurrogateNeuron** — trainable surrogate gradient
7. **ContinuousAttractorNeuron** — ring attractor for continuous WM
8. **MetaPlasticNeuron** — self-regulating meta-learning rate

All use `math` (not numpy) for transcendental functions and are fully
pipeline-compatible (`step(current) → int`).

---

## 1. MultiTimescaleNeuron

### Equations

$$\frac{dv_{fast}}{dt} = \frac{-v_{fast} + I}{\tau_{fast}}$$
$$\frac{dv_{medium}}{dt} = \frac{-v_{medium} + \alpha \cdot \text{spike}}{\tau_{medium}}$$
$$\frac{dv_{slow}}{dt} = \frac{-v_{slow} + \beta \cdot v_{medium}}{\tau_{slow}}$$
$$\theta_{eff} = \theta_{base} - \gamma \cdot v_{slow}$$

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_fast` | 5 ms | Fast integration |
| `tau_medium` | 200 ms | Medium context |
| `tau_slow` | 10,000 ms (10 s) | Slow context |
| `alpha` | 10.0 | Spike → medium gain |
| `beta` | 0.05 | Medium → slow gain |
| `gamma` | 0.3 | Slow → threshold modulation |

### Key insight

Three cascaded timescales (5 ms → 200 ms → 10 s) create a **temporal
hierarchy:** the fast compartment processes current input, the medium
compartment tracks recent spike history (~200 ms), and the slow compartment
accumulates contextual information over seconds. The slow variable
modulates the threshold — neurons with rich recent history become more
excitable (lower threshold).

**Application:** Sequential tasks where context from seconds ago matters
(speech, navigation, reinforcement learning).

---

## 2. AttentionGatedNeuron

### Equations

$$\text{gate} = \sigma(w_{key} \cdot I + w_{query} \cdot v)$$
$$\frac{dv}{dt} = \frac{-v + \text{gate} \cdot I}{\tau}$$

### Key insight

Each neuron learns **which input magnitudes to attend to** via key/query
weights. The gate is a sigmoid of the key-value alignment (input × key +
state × query) — a single-neuron implementation of the transformer
attention mechanism.

**Application:** Feature selection, input filtering, saliency detection.

---

## 3. PredictiveCodingNeuron

### Equations

$$\frac{d\text{pred}}{dt} = \frac{I - \text{pred}}{\tau_{pred}}$$
$$\text{surprise} = |I - \text{pred}|$$
$$\frac{dv}{dt} = \frac{-v + \text{surprise}}{\tau}$$

### Key insight

The neuron maintains a running prediction of its input (exponential
average with τ_pred=50 ms). It fires only when the input deviates from
the prediction — a **surprise detector**. Constant input → silence.
Changing input → spikes.

**Application:** Anomaly detection, change detection, novelty-based
exploration in RL.

---

## 4. SelfReferentialNeuron

### Equations

$$\text{rate} = \frac{\text{count(recent\_spikes)}}{\text{window}}$$
$$\tau_{eff} = \tau \cdot (1 + \text{rate} / \text{target\_rate})$$

### Key insight

The neuron monitors its own spike history (last 50 steps) and adjusts
its dynamics to match a target firing rate. High recent activity → slower
integration (self-stabilisation). Low activity → faster response. This is
a **homeostatic mechanism** implemented at the single-neuron level.

**Application:** Self-regulating networks, homeostatic learning, robust
representations.

---

## 5. CompositionalBindingNeuron

### Equations

$$\frac{d\phi}{dt} = \omega$$
$$\frac{dA}{dt} = \frac{-A + I}{\tau}$$
$$\text{spike} = \text{int}(A \cdot \cos(\phi) > \theta)$$

### Key insight

The neuron oscillates at frequency ω and fires when the product of
amplitude (input-driven) and phase (cosine) exceeds threshold. Two
neurons with the **same phase** represent bound concepts (e.g., "red" +
"circle" = "red circle"). Phase offset encodes different bindings.

**Application:** Variable binding in symbolic AI, compositional
representations, relational reasoning.

---

## 6. DifferentiableSurrogateNeuron

### Equations

$$\text{spike} = \text{int}(v \geq \theta)$$
$$v_{t+1} = \alpha \cdot v_t \cdot (1 - \text{spike}) + I$$
$$\sigma'(v) = \frac{1}{(1 + \beta|v - \theta|)^2}$$

### Key insight

All three parameters (α decay, β steepness, θ threshold) are **trainable.**
The surrogate gradient is accessible via `surrogate_grad()`. Unlike
SuperSpikeNeuron (which fixes α and β), this model makes them learnable —
enabling the network to optimise its own spiking dynamics during training.

**Application:** Meta-learning of neuron dynamics, neural architecture
search for SNNs.

---

## 7. ContinuousAttractorNeuron

### Equations

$$u_i += \frac{-u_i + f\left(\sum_j w_{ij} u_j\right) + I}{\tau} \cdot dt$$
$$w_{ij} = A \exp\left(-\frac{d_{ij}^2}{2\sigma_e^2}\right) - B$$
$$f(x) = \frac{\max(0,x)^2}{1 + \max(0,x)^2}$$

### Key insight

A ring of 16 units with **Mexican hat connectivity** (local excitation,
global inhibition) creates a continuous attractor — an activity bump that
can be positioned anywhere on the ring and persists without input.

**Application:** Head direction cells, path integration, continuous
working memory (angular position).

---

## 8. MetaPlasticNeuron

### Equations

$$\frac{dv}{dt} = \frac{-v + I}{\tau}$$
$$\text{error\_trace} += \frac{-\text{error\_trace} + |\text{reward} - \text{expected}|}{\tau_{meta}} \cdot dt$$
$$\text{meta\_lr} = \frac{lr_0}{1 + \exp(-\kappa(\text{error\_trace} - \text{target\_error}))}$$

### Key insight

The neuron adjusts its own learning rate based on prediction error. High
error → fast learning (explore). Low error → slow learning (exploit).
This implements the **Xu & Bhatt (2002) meta-learning** principle at the
single-neuron level.

**Application:** Continual learning, non-stationary environments,
learn-to-learn.

---

## Design Philosophy

### Why novel designs?

Existing neuron models (HH, LIF, Izhikevich) are constrained by
biological plausibility. The AI-optimized models relax this constraint
to explore what computations are possible with spiking dynamics:

- **MultiTimescale:** Biological neurons have ≤3 timescales (membrane,
  adaptation, slow AHP). This model extends to 10 s context — beyond
  any single neuron but matching circuit-level dynamics.
- **PredictiveCoding:** Implements Karl Friston's Free Energy Principle
  at the single-neuron level — normally a network-level theory.
- **CompositionalBinding:** Implements vector symbolic architecture
  (Kanerva 2009) in a spiking framework.

### All use math, not numpy

The module uses `math.exp`, `math.cos`, `math.sin` instead of `numpy`.
This eliminates numpy overhead for single-neuron computation and enables
pure-Python deployment without numpy dependency.

---

## Pipeline Compatibility

All 8 models have `step(current) → int` interface. All are compatible
with Population, Network, SpikeMonitor, Projection.

**Exception:** ContinuousAttractorNeuron internally manages 16 units.
As a Population member, each "neuron" is a 16-unit ring — a
population-within-a-population.

---

## Performance

| Model | steps/s | Notes |
|-------|---------|-------|
| MultiTimescale | ~500K | 3 linear updates |
| AttentionGated | ~400K | 1 exp (math.exp) |
| PredictiveCoding | ~500K | 1 abs |
| SelfReferential | ~200K | deque sum per step |
| CompositionalBinding | ~400K | 1 cos (math.cos) |
| DifferentiableSurrogate | ~500K | 1 abs (surrogate optional) |
| ContinuousAttractor | ~10K | 16-unit ring, O(n²) weights |
| MetaPlastic | ~500K | 1 exp in update_meta |

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| MultiTimescale | 3 | 3-compartment cascade, threshold modulation, slow context |
| AttentionGated | 2 | gate opens/closes, selective amplification |
| PredictiveCoding | 3 | surprise detection, constant input silence, change triggers |
| SelfReferential | 2 | rate homeostasis, history tracking |
| CompositionalBinding | 2 | phase-amplitude product, phase binding |
| DifferentiableSurrogate | 2 | trainable params, surrogate gradient |
| ContinuousAttractor | 2 | bump formation, bump position |
| MetaPlastic | 2 | meta-lr adapts, error-driven |
| **Total** | **18** | |

See `tests/test_model_ai_optimized.py`. No bugs found.

---

## Findings

1. **All 8 models pipeline-compatible:** step(current)→int works for all.

2. **MultiTimescale 10s context unique:** No other model in SC-NeuroCore
   has a timescale exceeding 1 s. τ_slow=10,000 ms provides unprecedented
   contextual memory.

3. **PredictiveCoding detects change:** Constant input → silence. Step
   change → burst of spikes. Exactly implements surprise detection.

4. **SelfReferential homeostasis:** Rate self-regulates toward target_rate
   via τ_eff modulation.

5. **CompositionalBinding uses phase:** Two neurons with same ω and phase
   represent bound concepts — a spiking implementation of VSA.

6. **ContinuousAttractor forms bumps:** Mexican hat connectivity creates
   stable activity bumps on the 16-unit ring.

7. **MetaPlastic adjusts lr:** High error → lr increases. Low error →
   lr decreases. Matches meta-learning theory.

8. **All use math, not numpy:** Eliminates numpy overhead for single-
   neuron computation.
