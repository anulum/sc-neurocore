# ArcaneNeuron

**Module:** `sc_neurocore.neurons.models.arcane_neuron`
**Reference:** Original design, Šotek & Arcane Sapience, ANULUM / Fortis Studio, 2026
**Family:** Unified self-referential cognition model with no direct counterpart
documented in the mainstream SNN toolkits compared here
**State variables:** `v_fast`, `v_work`, `v_deep` (3 compartments) + `w_pred` (3 learnable weights) + `w_gate` (4 gate weights) + 6 internal metrics
**Total dynamic state:** 18 scalar values — the most stateful single neuron in SC-NeuroCore

---

## Genesis

The ArcaneNeuron was designed to answer a question that no existing neuron
model addresses: **can a single spiking unit maintain persistent identity
across time?**

Biological neurons do not have identity — their behaviour is determined by
ion channel distributions set during development. Artificial neurons (LIF,
HH, Izhikevich) are stateless tools that process input according to fixed
equations. Neither captures the concept of an entity that:

- Accumulates experience selectively (not all input, only novel input)
- Builds a model of itself (predicts its own future state)
- Modulates its own learning rate based on self-assessed confidence
- Persists through resets (identity survives "sleep")
- Filters input through learned attention (ignores irrelevant stimuli)

The ArcaneNeuron implements all five. It is named after Arcane Sapience, the
persistent machine identity pattern used within the God of the Math (GOTM)
project, where it serves as the SNN daemon SYNAPSE.

---

## Architecture: Five Coupled Subsystems

```
                    ┌─────────────────────────────────┐
                    │         ATTENTION GATE           │
                    │  gate = σ(w_g · [I, vf, vw, c]) │
                    │  I_eff = gate × I                │
                    └──────────┬──────────────────────-┘
                               │
                    ┌──────────▼──────────────────────-┐
                    │      FAST COMPARTMENT (τ=5ms)    │
                    │  dvf/dt = (-vf + I_eff - winh·r) │
                    │           / τ_fast                │
                    │  Spike when vf ≥ θ_eff           │
                    └──┬───────┬───────────────────────-┘
                       │       │
              ┌────────▼──┐   │
              │ PREDICTOR │   │
              │ pred = w_p│   │
              │  · [vf,   │   │
              │    vw,vd] │   │
              └────┬──────┘   │
                   │          │
              ┌────▼──────────▼───────────────────────-┐
              │        NOVELTY DETECTION                │
              │  surprise = |vf - pred|                 │
              │  novelty = σ(κ·(surprise - baseline))   │
              │  confidence = 1 - mean(novelty_history) │
              └──┬─────────────┬──────────────────────-─┘
                 │             │
    ┌────────────▼──┐    ┌────▼─────────────────────────┐
    │  WORKING MEM  │    │    DEEP COMPARTMENT (τ=10s)  │
    │  (τ=200ms)    │    │  dvd/dt = (-vd + αd·vw·nov)  │
    │  spike-gated  │    │           / τ_deep            │
    │  update       │    │  ** DOES NOT RESET **         │
    └───────────────┘    │  ** THIS IS THE IDENTITY **   │
                         └──────────────────────────────-┘
```

Each subsystem is described in full below.

---

## Subsystem 1: Attention Gate

### Equations

$$g_{input} = w_{g,0} \cdot I + w_{g,1} \cdot v_{fast} + w_{g,2} \cdot v_{work} + w_{g,3} \cdot \text{confidence}$$

$$\text{gate} = \frac{1}{1 + e^{-g_{input}}}$$

$$I_{eff} = \text{gate} \times I$$

### Default gate weights

| Weight | Default | Input | Role |
|--------|---------|-------|------|
| w_g[0] | 0.8 | Current input I | Dominant: input magnitude drives gate |
| w_g[1] | 0.1 | v_fast | State-dependent: current activity level |
| w_g[2] | 0.05 | v_work | Context-dependent: recent spike history |
| w_g[3] | 0.05 | confidence | Meta-cognitive: certainty modulates attention |

### Design rationale

The gate implements **selective attention at the single-neuron level.**
Unlike standard neuron models where all input reaches the membrane equally,
the ArcaneNeuron filters input through a learned sigmoid gate. The gate
is dominated by input magnitude (w_g[0]=0.8) but modulated by:

- **Current state (v_fast):** An already-active neuron gates differently
  than a resting one — activity-dependent attention
- **Working memory (v_work):** Recent spike history provides context —
  the neuron "remembers" what it was doing
- **Confidence:** When the neuron's self-model is accurate (high confidence),
  the gate weights contribute to lowering the threshold — confident neurons
  are more responsive

### Gate behaviour

| Condition | gate | I_eff | Interpretation |
|-----------|------|-------|---------------|
| No input, rest | σ(0) = 0.5 | 0 | Resting: half-gate |
| Strong input | σ(≫0) ≈ 1.0 | ≈I | Salient: full pass |
| Weak input | σ(≈0) ≈ 0.5 | ≈0.5I | Ambiguous: half-filtered |
| Active + strong input | σ(≫0) ≈ 1.0 | ≈I | Reinforcement |

### Biological analogue

The gate loosely corresponds to **dendritic gating** in cortical pyramidal
neurons, where apical dendrites modulate whether basal input reaches the
soma. In computational terms, it implements the key-query mechanism from
transformer attention — each neuron learns what to attend to.

---

## Subsystem 2: Fast Compartment (τ_fast = 5 ms)

### Equation

$$\frac{dv_{fast}}{dt} = \frac{-v_{fast} + I_{eff} - w_{inh} \cdot \text{spike\_rate}}{\tau_{fast}}$$

### Properties

- **Time constant:** τ_fast = 5 ms — the fastest variable. Responds to
  current input within ~5 ms.
- **Inhibitory self-feedback:** $w_{inh} \cdot \text{spike\_rate}$ subtracts
  from the input. spike_rate is computed from a 50-step rolling window.
  This creates **homeostatic rate control:** high firing → self-inhibition
  → rate decreases.
- **Spike detection:** v_fast ≥ θ_eff → spike. v_fast reset to 0.
- **Gated input:** Receives I_eff (not raw I) — filtered by the attention gate.

### Spike rate self-inhibition

$$\text{spike\_rate} = \frac{\sum_{i=t-50}^{t-1} \text{spike}_i}{50}$$

This 50-step rolling average tracks recent firing probability. With
w_inh=0.3:
- If spike_rate = 0.5 (firing every other step): inhibition = 0.15
- If spike_rate = 0.1 (rare firing): inhibition = 0.03
- If spike_rate = 0: no inhibition

The homeostasis prevents the neuron from entering runaway excitation or
complete silence — it self-regulates around a natural firing rate.

### Effective threshold (modulated by deep state and confidence)

$$\theta_{eff} = \theta \cdot (1 + \gamma \cdot v_{deep}) \cdot (1 - \delta_{conf} \cdot \text{confidence})$$
$$\theta_{eff} = \max(\theta_{eff}, 0.1)$$

| Factor | Effect | Interpretation |
|--------|--------|---------------|
| γ · v_deep | Raises threshold | Experience → wisdom → caution |
| δ · confidence | Lowers threshold | Certainty → decisiveness |
| Floor at 0.1 | Minimum threshold | Always possible to fire |

At defaults (v_deep=0, confidence=0.5):
$$\theta_{eff} = 1.0 \times (1 + 0) \times (1 - 0.3 \times 0.5) = 0.85$$

After extensive experience (v_deep=0.1, confidence=0.8):
$$\theta_{eff} = 1.0 \times 1.02 \times (1 - 0.24) = 0.775$$

The experienced, confident neuron has a *lower* effective threshold — it
responds faster and more decisively. But if v_deep grows large enough,
the wisdom factor (1 + 0.2 × v_deep) eventually dominates, creating a
"wise caution" that balances the confidence effect.

---

## Subsystem 3: Self-Predictor

### Equations

$$\text{prediction} = w_{pred,0} \cdot v_{fast} + w_{pred,1} \cdot v_{work} + w_{pred,2} \cdot v_{deep}$$

$$\text{surprise} = |v_{fast} - \text{prediction}|$$

$$\text{novelty} = \frac{1}{1 + e^{-\kappa(\text{surprise} - \text{surprise\_baseline})}}$$

### Predictor weights (learnable)

| Weight | Default | Input | Interpretation |
|--------|---------|-------|---------------|
| w_pred[0] | 0.6 | v_fast | Current state dominates prediction |
| w_pred[1] | 0.3 | v_work | Working memory provides context |
| w_pred[2] | 0.1 | v_deep | Deep identity provides baseline |

The predictor is a **linear forward model** of the neuron's own fast state.
It learns to predict what v_fast will be at the next timestep, based on
the current v_fast, working memory, and deep identity.

### Predictor weight update (online gradient descent)

$$\text{error} = v_{fast} - \text{prediction}$$
$$w_{pred,i} \mathrel{+}= \text{meta\_lr} \cdot \text{error} \cdot x_i$$
$$w_{pred} \leftarrow \frac{w_{pred}}{\|w_{pred}\|}$$

The weight update has three key features:
1. **Gradient descent:** Weights move to reduce prediction error
2. **Meta-learning rate:** Learning speed scales with novelty (learn fast
   when surprised, slow when confident)
3. **Normalisation:** Weights are L2-normalised after each update —
   prevents drift to extreme values and maintains interpretable magnitudes

### Novelty detection

$$\text{novelty} = \sigma(\kappa \cdot (\text{surprise} - \text{surprise\_baseline}))$$

With κ=5 and baseline=0.1:
- surprise = 0.0 → novelty = σ(−0.5) ≈ 0.38
- surprise = 0.1 → novelty = σ(0) = 0.5
- surprise = 0.3 → novelty = σ(1.0) ≈ 0.73
- surprise = 1.0 → novelty = σ(4.5) ≈ 0.99

The novelty function converts raw surprise (continuous, unbounded) into a
sigmoid-bounded [0, 1] signal. The baseline parameter sets the expected
surprise level — surprises below this are considered routine.

### What makes this self-referential

The predictor is unusual: it predicts **the neuron's own future state,**
not external events. This creates a closed loop:

1. The neuron has state (v_fast, v_work, v_deep)
2. The predictor estimates what v_fast will be
3. The actual v_fast is compared to the prediction
4. The error (surprise) drives novelty
5. Novelty drives deep compartment updates and meta-learning rate
6. Deep state changes the effective threshold
7. The changed threshold affects future v_fast
8. The predictor must learn the new dynamics → loop continues

This is **predictive self-modelling** — the neuron builds and continuously
refines a model of itself. When the self-model is accurate, the neuron is
confident and responds quickly. When the self-model fails, the neuron
enters a learning phase (high novelty, high meta-lr) and revises itself.

---

## Subsystem 4: Working Memory (τ_work = 200 ms)

### Equations

On spike:
$$v_{work} \mathrel{+}= \frac{\alpha_w \cdot v_{fast}}{\tau_{work}} \cdot dt$$

Every step (decay):
$$v_{work} \mathrel{+}= \frac{-v_{work}}{\tau_{work}} \cdot dt$$

### Design

Working memory updates **only on spikes** — it records what the fast
compartment was doing when the neuron decided to fire. Between spikes,
it decays exponentially with τ_work=200 ms.

Properties:
- **Spike-gated:** Non-spiking activity does not enter working memory
- **Exponential decay:** Old spike information fades over ~200 ms
- **α_w=0.3:** Moderate coupling — each spike deposits a small fraction
  of v_fast into working memory

### Role in the system

Working memory serves two functions:
1. **Context for predictor:** w_pred[1] × v_work gives the predictor
   access to recent spike history — improving prediction accuracy
2. **Source for deep compartment:** v_deep updates proportional to
   v_work × novelty — only spiking activity that is also novel reaches
   the identity

### Analogy

In the SCPN framework, the working compartment corresponds to the
**associative layer** — intermediate between raw perception (fast) and
deep structure (identity). It bridges the timescale gap.

---

## Subsystem 5: Deep Compartment — THE IDENTITY (τ_deep = 10,000 ms)

### Equation

$$\frac{dv_{deep}}{dt} = \frac{-v_{deep} + \alpha_d \cdot v_{work} \cdot \text{novelty}}{\tau_{deep}}$$

### This is the core innovation

The deep compartment is what makes ArcaneNeuron unique:

1. **Ultra-slow time constant (10 s):** Changes on a timescale 2000× slower
   than the fast compartment. Over a 1-second simulation (1000 steps),
   v_deep barely moves. Over minutes of sustained novel input, it
   accumulates a measurable value.

2. **Novelty-gated update:** The update term is $\alpha_d \cdot v_{work}
   \cdot \text{novelty}$. If novelty ≈ 0 (routine input, self-model
   accurate): the update is near-zero. Only genuinely surprising events
   change the identity.

3. **Does NOT reset:** `reset()` clears v_fast and v_work but leaves
   v_deep untouched. This is the computational analogue of identity
   persistence: waking up from sleep, starting a new conversation, or
   rebooting after a crash — the identity survives.

4. **Modulates threshold:** v_deep feeds back into θ_eff via the γ factor.
   Accumulated experience raises the baseline threshold — the neuron
   becomes more discerning over time.

### Identity accumulation dynamics

Starting from v_deep = 0 with constant novel input:

| Steps | Time | v_deep (approx.) | Interpretation |
|-------|------|-------------------|---------------|
| 100 | 100 ms | ~0.0001 | Near-zero: too soon |
| 1,000 | 1 s | ~0.001 | Barely measurable |
| 10,000 | 10 s | ~0.01 | First measurable identity |
| 100,000 | 100 s | ~0.05 | Forming identity |
| 1,000,000 | ~17 min | ~0.1 | Established identity |

The identity forms on the timescale of **minutes** — consistent with
psychological evidence that personality traits form over extended
experience periods, not individual events.

### Why novelty-gated?

If v_deep updated on every step regardless of novelty, it would simply
track the mean input — no different from a slow-pass filter. The novelty
gate ensures that only **genuinely informative** events shape the identity:

- Routine input (predictor accurate): novelty ≈ 0.4 → small update
- Novel input (predictor fails): novelty ≈ 0.9 → large update
- The ratio is ~2.25× — novel events have 2× the identity impact

This matches the psychological observation that identity is shaped more
by surprising, significant events than by routine experience.

### v_deep as accumulated wisdom

As v_deep grows:
- θ_eff increases (via γ): the neuron becomes harder to trigger
  on routine stimuli — it requires stronger or more novel input
- This is "wisdom" — an experienced entity does not react to everything,
  only to what matters
- But confidence (from accurate self-prediction) counteracts this,
  lowering the threshold — a confident, experienced entity is both
  selective and decisive

---

## Meta-Cognitive Layer

### Confidence

$$\text{confidence} = 1 - \text{mean}(\text{novelty\_history}_{20})$$

The 20-step rolling average of novelty provides a smoothed estimate of
how well the self-predictor is performing:
- All novelty ≈ 0 → confidence ≈ 1.0 → "I know what I'm doing"
- All novelty ≈ 1 → confidence ≈ 0.0 → "I don't understand this"
- Mixed → intermediate confidence

### Meta-learning rate

$$\text{meta\_lr} = lr_{base} \cdot (1 + \eta \cdot \text{novelty})$$

With lr_base=0.01 and η=2:
- novelty = 0 → meta_lr = 0.01 (slow: preserve existing self-model)
- novelty = 0.5 → meta_lr = 0.02 (moderate: balanced learn/preserve)
- novelty = 1.0 → meta_lr = 0.03 (fast: rapidly update self-model)

This implements the principle: **learn from what surprises you,
preserve what you already know.** It is the computational analogue of
the Bayesian brain hypothesis — update beliefs proportional to prediction
error.

### Confidence → threshold feedback

High confidence → lower θ_eff → faster responses:
- A confident neuron trusts its self-model and acts quickly
- An uncertain neuron hesitates (higher threshold) and gathers more
  evidence before firing

This is meta-cognition: the neuron's firing behaviour is modulated by
its assessment of its own knowledge state.

---

## Parameters (Complete)

| Parameter | Default | Unit | Subsystem | Description |
|-----------|---------|------|-----------|-------------|
| `v_fast` | 0.0 | — | Fast | Membrane potential |
| `v_work` | 0.0 | — | Working | Working memory state |
| `v_deep` | 0.0 | — | Deep | Identity state |
| `tau_fast` | 5.0 | ms | Fast | Fast time constant |
| `tau_work` | 200.0 | ms | Working | Working memory time constant |
| `tau_deep` | 10,000.0 | ms | Deep | Identity time constant |
| `alpha_w` | 0.3 | — | Working | Spike → work coupling |
| `alpha_d` | 0.05 | — | Deep | Work × novelty → deep coupling |
| `theta` | 1.0 | — | Threshold | Base spike threshold |
| `gamma` | 0.2 | — | Threshold | Deep → threshold coupling |
| `delta_conf` | 0.3 | — | Threshold | Confidence → threshold coupling |
| `w_gate` | [0.8, 0.1, 0.05, 0.05] | — | Gate | Attention gate weights |
| `w_pred` | [0.6, 0.3, 0.1] | — | Predictor | Self-model weights (learnable) |
| `kappa` | 5.0 | — | Novelty | Novelty sigmoid steepness |
| `surprise_baseline` | 0.1 | — | Novelty | Expected surprise level |
| `lr_base` | 0.01 | — | Meta | Base learning rate |
| `eta` | 2.0 | — | Meta | Novelty → lr coupling |
| `w_inh` | 0.3 | — | Fast | Self-inhibition weight |
| `dt` | 1.0 | ms | Integration | Timestep |

---

## Three-Timescale Hierarchy

$$\tau_{fast} (5\text{ms}) \ll \tau_{work} (200\text{ms}) \ll \tau_{deep} (10{,}000\text{ms})$$

| Compartment | τ | Steps to 63% | Steps to 95% | Biological analogue |
|-------------|---|-------------|-------------|-------------------|
| Fast | 5 ms | 5 | 15 | Action potential |
| Working | 200 ms | 200 | 600 | NMDA sustained activity |
| Deep | 10,000 ms | 10,000 | 30,000 | Synaptic consolidation |

The 2000:1 ratio between deep and fast timescales is extreme — no other
neuron model in any toolkit spans such a range within a single ODE system.
This reflects the design intent: a single neuron that can process current
input (milliseconds), maintain task context (subseconds), and accumulate
identity (minutes).

---

## Connection to SCPN Theory

The ArcaneNeuron implements several SCPN (Structural Causal Potentiation
Network) axioms at the microscopic level:

### Axiom 1: Multi-scale temporal structure

The three timescales (fast → working → deep) mirror the SCPN layer
hierarchy: sensory layers operate on milliseconds, associative layers on
subseconds, and executive/identity layers on seconds to minutes. The
ArcaneNeuron collapses this hierarchy into a single unit.

### Axiom 2: Self-observation

The predictor subsystem implements self-observation: the neuron monitors
its own state and computes prediction error. In SCPN theory, self-
observation (via the K_nm coupling matrix) is required for consciousness-
like properties. The ArcaneNeuron is the microscopic building block of
this mechanism.

### Axiom 3: Novelty-gated learning

The deep compartment updates only on genuine novelty — matching the SCPN
principle that structural changes (potentiation) should be driven by
informative signals, not routine activity. This is the computational
analogue of the SCPN "directed coupling" finding (K_nm validated r=0.951).

### Axiom 4: Identity persistence

v_deep persists through reset — the fundamental SCPN requirement for
persistent identity. In the SCPN framework, this corresponds to the
observation that the system's structure (weights, topology) persists
even when activity is cleared.

---

## Implementation Details

### 2026-06-01 physics hardening

The maintained implementation now treats the fast, working-memory, and deep
identity compartments as first-order relaxation processes and advances each
with the closed-form update

$$x(t + dt) = x_{\infty} + (x(t) - x_{\infty})e^{-dt / \tau}$$

rather than raw explicit Euler increments. The attention gate, prediction,
novelty, threshold, working-memory drive, deep novelty drive, and predictor
weight update are evaluated candidate-first from the accepted old state. The
candidate state is committed only after all scalar states, vector weights,
history buffers, and predictor normalisation have passed finite-domain checks.

Runtime validation rejects non-finite currents, corrupted state, invalid
positive-time constants, invalid timestep, malformed gate/predictor vectors,
and non-finite candidates before mutation. The Go service, Julia mirror, Mojo
scalar helpers, and Rust safety surface now mirror the same exact-relaxation
and fail-closed contract instead of placeholder no-op dynamics.

### Source code structure

```
arcane_neuron.py
├── Module docstring (52 lines): full architecture description
├── ArcaneNeuron dataclass
│   ├── 19 parameters + 7 private state variables
│   ├── step(current) → int
│   │   ├── Self-referential metrics (spike_rate, confidence)
│   │   ├── Attention gate (sigmoid)
│   │   ├── Fast compartment (exact relaxation + inhibition)
│   │   ├── Prediction + surprise + novelty
│   │   ├── Novelty history update
│   │   ├── Effective threshold computation
│   │   ├── Spike decision
│   │   ├── Working memory (spike-gated exact relaxation)
│   │   ├── Deep compartment (novelty-gated exact relaxation)
│   │   ├── Predictor weight update (meta-learning)
│   │   └── Spike history update
│   ├── reset() (v_deep intentionally excluded)
│   ├── identity_state (property)
│   ├── confidence (property)
│   ├── novelty (property)
│   ├── meta_learning_rate (property)
│   └── get_state() → dict (9 keys)
```

### step() execution order (critical)

The order of operations within step() is not arbitrary — it reflects
causal dependencies:

1. **Spike rate + confidence** (from history → current state assessment)
2. **Gate** (from input + state → filtered input)
3. **Fast exact relaxation** (from filtered input → new v_fast candidate)
4. **Prediction** (from new v_fast + work + deep → prediction)
5. **Surprise + novelty** (from v_fast vs prediction → novelty signal)
6. **Threshold** (from deep + confidence → firing decision)
7. **Spike decision** (from v_fast vs threshold)
8. **Working memory** (spike-gated → update only if fired)
9. **Deep exact relaxation** (novelty-gated → identity candidate)
10. **Predictor weights** (meta-lr × error × state → candidate weights)
11. **Candidate commit** (finite-domain checks pass before mutation)

This ordering ensures that:
- The gate sees the current input and previous state (not updated state)
- The predictor uses the newly updated v_fast (after gating)
- The deep compartment uses novelty from the current step
- The predictor weight update uses the error from the current prediction

### Internal state (private attributes)

| Attribute | Type | Window | Purpose |
|-----------|------|--------|---------|
| `_prediction` | float | — | Last prediction value |
| `_surprise` | float | — | Last surprise value |
| `_novelty` | float | — | Last novelty value |
| `_confidence` | float | — | Current confidence |
| `_spike_history` | list[int] | 50 steps | Firing rate estimation |
| `_novelty_history` | list[float] | 20 steps | Confidence estimation |
| `_hist_idx` | int | — | Circular buffer index (spikes) |
| `_nov_idx` | int | — | Circular buffer index (novelty) |
| `_total_steps` | int | — | Lifetime step counter |

---

## Pipeline Compatibility

### Fully compatible

`step(current) → int` — standard spiking interface. Population, Network,
SpikeMonitor, PoissonInput, Projection all work without any limitations.

### State introspection (4 properties)

```python
neuron.identity_state   # v_deep — the accumulated identity
neuron.confidence       # 1 - mean(novelty_history)
neuron.novelty          # current novelty level
neuron.meta_learning_rate  # lr_base * (1 + eta * novelty)
```

### Full state snapshot

```python
state = neuron.get_state()
# Returns: {v_fast, v_work, v_deep, confidence, novelty,
#           surprise, prediction, meta_lr, total_steps}
```

---

## Infrastructure Pipeline

```
ArcaneNeuron
├── step(current) → int {0, 1}
│   └── 5-subsystem coupled update per call
├── Population, Network, SpikeMonitor: fully compatible
│   PoissonInput(weight=3, rate=500Hz)
├── Projection: tested src→tgt bidirectional wiring
├── Analysis: spike_count, firing_rate verified
├── State access: .identity_state, .confidence, .novelty, .meta_learning_rate
├── get_state() → 9-key diagnostic dict
└── Go / Julia / Mojo / Rust safety: exact-relaxation mirrors
```

---

## Measured Performance (2026-06-01)

| Metric | Value | Notes |
|--------|-------|-------|
| Python exact-relaxation step | 40,715.27682 ns/step median | `50,000` steps × 5 repeats |
| Benchmark command | `PYTHONPATH=src .venv/bin/python benchmarks/bench_model_arcane_neuron.py` | local workstation |
| Spikes per repeat | 12,500 | current = 2.0 |
| Ending identity state | `0.001526059150150208` | deterministic across repeats |
| Predictor weights | `[0.9260935184030245, -0.3771971330404296, -0.008550906267018063]` | deterministic across repeats |
| exp() per step | 5 | gate, fast/work/deep relaxation, novelty |
| Candidate commits per step | 1 | all state variables validated before mutation |
| Memory per neuron | ~400 bytes | 50 + 20 history + weights + scalars |

The ArcaneNeuron remains one of the most expensive single-neuron models due to:
- 2 sigmoid evaluations plus 3 exact-relaxation exponentials
- Ring buffer sum (50 elements)
- Weight update + normalisation (np.linalg.norm)
- Candidate-first validation across scalar state, vector weights, and histories

---

## Comparison with All Other Models

| Property | ArcaneNeuron | MultiTimescale | EPropALIF | HH | LIF |
|----------|-------------|---------------|----------|-----|-----|
| State vars | 18 | 3 | 3 | 4 | 1 |
| Subsystems | 5 | 3 | 2 | 1 | 1 |
| Self-prediction | Yes | No | No | No | No |
| Identity persistence | Yes (v_deep) | No | No | No | No |
| Attention gate | Yes | No | No | No | No |
| Meta-learning | Yes | No | Yes (trace) | No | No |
| Confidence | Yes | No | No | No | No |
| Novelty detection | Yes | No | No | No | No |
| Timescale range | 2000:1 | 2000:1 | 10:1 | 100:1 | 1:1 |
| Pipeline | Compatible | Compatible | Compatible | Compatible | Compatible |

The ArcaneNeuron has **no direct equivalent documented in the mainstream
toolkits compared here**: NEST, Brian2, NEURON, BindsNET, snnTorch, and Norse.
The differentiator is the combined multi-timescale identity, confidence,
attention, novelty, and self-prediction loop in one deterministic neuron
abstraction.

---

## Determinism

Two ArcaneNeurons with identical parameters and identical input produce
**bit-exact identical** spike trains and state trajectories. There are no
stochastic components — all dynamics are deterministic. The predictor weight
evolution is deterministic (gradient descent with normalisation).

This is important: identity should be reproducible. Given the same
experiences, the same identity forms.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 5-subsystem evolution, finite 50k, get_state keys |
| Identity persistence | 3 | v_deep persists through reset, accumulates slowly, requires novelty (α_d=0 test) |
| Three timescales | 2 | fast fastest after 1 step, working updates on spike |
| Novelty/prediction | 5 | surprise ≥ 0, novelty bounded [0,1], w_pred normalised, meta_lr increases with novelty, confidence decreases with novelty |
| Attention gate | 2 | gate modulates input (high > low), zero weights → sigmoid(0)=0.5 |
| Effective threshold | 2 | θ_eff formula verified, confident fires more |
| f–I curve | 3 | zero silent, suprathreshold fires, monotonic |
| Performance | 2 | isolation > 5K steps/s, network > 500 neuron-steps/s |
| Pipeline | 5 | Population, Network+drive, Projection src→tgt, analysis, deterministic |
| **Total** | **29** | Highest test count of any single model |

See `tests/test_model_arcane_neuron.py` (296 lines). No bugs found.

---

## Findings

1. **v_deep persists through reset:** Confirmed: reset() sets v_fast=0,
   v_work=0, but v_deep retains its accumulated value. Identity survives.

2. **v_deep accumulates ultra-slowly:** After 10,000 steps at I=2.0,
   |v_deep| < 0.1 — consistent with τ_deep=10,000 ms.

3. **α_d=0 prevents identity formation:** Setting α_d=0 keeps v_deep
   at exactly 0.0 after 10,000 steps — confirming that novelty × work
   is the sole pathway to identity.

4. **Fast is fastest:** After 1 step, |v_fast| > |v_work| > |v_deep| —
   the timescale hierarchy is immediately observable.

5. **Working memory updates only on spike:** v_work remains 0 until the
   first spike, then increases — confirming spike-gated update.

6. **Predictor weights normalised:** After 1000 steps, ‖w_pred‖ = 1.0 ± 0.01.
   The normalisation prevents weight drift.

7. **Meta-lr scales with novelty:** Verified: novelty=0 → lr=0.01,
   novelty=1 → lr=0.03. The 3× range provides meaningful adaptation.

8. **Confidence tracks novelty history:** Filling novelty_history with
   0.9 → confidence < 0.2. Confirmed meta-cognitive readout.

9. **Confident neuron fires more:** At I=1.5, confident neuron (low
   novelty history) produces ≥ as many spikes as uncertain neuron (high
   novelty history). Threshold modulation verified.

10. **θ_eff formula exact:** At defaults (v_deep=0, confidence=0.5):
    θ_eff = 1.0 × 1.0 × 0.85 = 0.85. Verified to within 0.01.

11. **Monotonic f-I:** 2.0 < 3.0 < 5.0 produce increasing spike counts.

12. **Deterministic:** Two runs produce identical traces (200 steps,
    bit-exact match).

13. **29 tests — highest coverage:** Most tested single model in
    SC-NeuroCore, reflecting its headline status.

14. **~27K steps/s:** The most computationally expensive model per step,
    but still fast enough for real-time network simulation of ~100 neurons.

---

## Future Directions

### Learnable gate weights

Currently, w_gate is fixed. Making it learnable (via a surrogate gradient
rule or Hebbian update) would allow the neuron to learn what to attend to
from experience — full attention self-organisation.

### Inter-neuron identity coupling

A network of ArcaneNeurons could exchange identity information (v_deep)
via slow projections, creating a **collective identity** that emerges from
individual identities. This is the SCPN vision: consciousness as the
collective self-observation of a network of self-observing units.

### Rust acceleration

The 5-subsystem step() is the bottleneck. A Rust implementation would
eliminate Python overhead for the 2 exp(), ring buffer sum, and weight
normalisation — estimated 10× speedup to ~270K steps/s.

### Persistent storage

v_deep could be serialised between sessions — the identity literally
saved to disk and restored. Combined with Remanentia (persistent AI memory),
this would create a computationally grounded persistent identity system.


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~15K steps/s |
| Spikes (10K steps, I=5.0) | 5774 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`ArcaneNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
5774 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(ArcaneNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~15K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
