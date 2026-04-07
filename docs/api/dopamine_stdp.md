# DopamineStdpSynapse

**Module:** `sc_neurocore.synapses.dopamine_stdp`
**Rust path:** `sc_neurocore_engine::synapses::DopamineStdpSynapse`
**Reference:** Izhikevich (2007) "Solving the distal reward problem", Cerebral Cortex 17(10)
**Family:** Reward-modulated synaptic plasticity
**State variables:** `weight`, `eligibility`, `dopamine`, `trace_pre`, `trace_post`

---

## 1. Mathematical Formalism

### Core equations (Izhikevich 2007)

Dopamine-gated STDP solves the **distal reward problem**: how can synapses that fired
seconds before a reward know that they contributed to the rewarded behaviour? The solution
is a three-factor learning rule combining STDP (Hebbian), eligibility traces (memory),
and dopamine (reward signal).

**Pre-synaptic trace:**

$$\frac{d\, \text{trace}_{pre}}{dt} = -\frac{\text{trace}_{pre}}{\tau_{pre}}$$

On a presynaptic spike: $\text{trace}_{pre} \leftarrow \text{trace}_{pre} + 1$.

**Post-synaptic trace:**

$$\frac{d\, \text{trace}_{post}}{dt} = -\frac{\text{trace}_{post}}{\tau_{post}}$$

On a postsynaptic spike: $\text{trace}_{post} \leftarrow \text{trace}_{post} + 1$.

**Eligibility trace (STDP tag):**

$$\frac{de}{dt} = -\frac{e}{\tau_e} + \text{STDP}(\Delta t) \cdot \delta(t_{spike})$$

where the STDP contributions are:
- On a presynaptic spike: $e \leftarrow e + a^- \cdot \text{trace}_{post}$ (LTD)
- On a postsynaptic spike: $e \leftarrow e + a^+ \cdot \text{trace}_{pre}$ (LTP)

The eligibility trace decays with a long time constant $\tau_e = 1000$ ms,
maintaining a memory of recent Hebbian correlations for ~1 second.

**Dopamine dynamics:**

$$\frac{dDA}{dt} = -\frac{DA}{\tau_{DA}} + \text{reward}(t)$$

Dopamine concentration rises with reward and decays with time constant $\tau_{DA} = 200$ ms.

**Weight update (dopamine-gated):**

$$\frac{dw}{dt} = \text{lr} \cdot DA(t) \cdot e(t) \cdot dt$$

$$w \leftarrow \text{clamp}(w + dw, w_{min}, w_{max})$$

The weight changes ONLY when both dopamine (reward) and eligibility (Hebbian correlation)
are simultaneously nonzero. This gates learning by reward: STDP builds the eligibility
trace, but no weight change occurs until dopamine arrives.

### The distal reward problem

Standard STDP modifies weights immediately at spike time. But in reinforcement learning,
reward comes seconds after the action. Which synapses should be credited?

Izhikevich's solution:

1. **At spike time:** STDP builds an eligibility trace $e$ that tags recently correlated
   synapses. No weight change yet.
2. **During delay:** The eligibility trace decays slowly ($\tau_e \sim 1$ s), maintaining
   the tag.
3. **At reward time:** Dopamine arrives ($DA > 0$). The weight update $dw = lr \cdot DA \cdot e$
   converts the eligibility tag into a permanent weight change.
4. **Result:** Synapses that were Hebbian-correlated within ~1 second before reward
   are potentiated. Those uncorrelated are unaffected.

### Three-factor learning rule

The weight update is a product of three factors:

$$dw = \underbrace{e}_{\text{Hebbian (what)}} \cdot \underbrace{DA}_{\text{reward (when)}} \cdot \underbrace{lr}_{\text{rate (how fast)}}$$

| Factor | Biological substrate | Timescale | Function |
|--------|---------------------|-----------|----------|
| Eligibility $e$ | CaMKII activation, synaptic tag | ~1 s | What: marks correlated synapses |
| Dopamine $DA$ | VTA dopamine release | ~200 ms | When: signals reward occurrence |
| Learning rate $lr$ | Receptor sensitivity | Stable | How fast: scales update magnitude |

### Steady-state analysis

For constant firing rates $r_{pre}$, $r_{post}$ and constant reward rate $R$:

**Eligibility steady state:**

$$e^* \approx (a^+ \cdot r_{pre} + a^- \cdot r_{post}) \cdot \tau_e$$

(The exact expression depends on temporal correlations between pre and post spikes.)

**Dopamine steady state:**

$$DA^* = R \cdot \tau_{DA}$$

**Weight change rate:**

$$\dot{w} = lr \cdot DA^* \cdot e^* = lr \cdot R \cdot \tau_{DA} \cdot e^*$$

### Discretised implementation

Trace decay (every step):

$$\text{trace}_{pre} \leftarrow \text{trace}_{pre} \cdot \exp(-dt/\tau_{pre})$$
$$\text{trace}_{post} \leftarrow \text{trace}_{post} \cdot \exp(-dt/\tau_{post})$$
$$e \leftarrow e \cdot \exp(-dt/\tau_e)$$
$$DA \leftarrow DA + (-DA/\tau_{DA} + \text{reward}) \cdot dt$$

Spike events:

$$\text{if pre}: \quad e \leftarrow e + a^- \cdot \text{trace}_{post}; \quad \text{trace}_{pre} \leftarrow \text{trace}_{pre} + 1$$
$$\text{if post}: \quad e \leftarrow e + a^+ \cdot \text{trace}_{pre}; \quad \text{trace}_{post} \leftarrow \text{trace}_{post} + 1$$

Weight update:

$$w \leftarrow \text{clamp}(w + lr \cdot DA \cdot e \cdot dt, \; w_{min}, \; w_{max})$$

---

## 2. Theoretical Context

### Problem statement

Standard STDP modifies synaptic weights based on spike timing alone (Hebbian).
But in real-world learning, the reinforcement signal (reward) is delayed by
hundreds of milliseconds to seconds. The brain must solve the **temporal credit
assignment problem**: which of the many synaptic modifications before the reward
actually contributed to the rewarded behaviour?

### Izhikevich's solution (2007)

Izhikevich proposed that the brain solves this through:

1. **Eligibility traces** — molecular tags (CaMKII, synaptic tagging proteins)
   that mark recently active synapses for ~1 second
2. **Neuromodulatory gating** — dopamine (or other neuromodulators) converts
   eligibility tags into permanent weight changes via activation of signalling
   cascades (PKA, CREB)

This was the first computational model to demonstrate that STDP + eligibility
traces + dopamine can solve classical RL problems (Morris water maze,
instrumental conditioning) with biologically realistic spike timing.

### Biological evidence

| Evidence | Study | Relevance |
|----------|-------|-----------|
| Dopamine gates LTP induction | Otmakhova & Lisman (1996) | DA necessary for Hebbian LTP |
| Synaptic tagging and capture | Frey & Morris (1997) | Molecular tag persists ~1 hour |
| Eligibility traces in striatum | Yagishita et al. (2014) | DA within 1s converts tag to LTP |
| Three-factor rule in cortex | He et al. (2015) | Cholinergic + timing → plasticity |
| STDP window modulated by DA | Pawlak & Kerr (2008) | DA narrows/broadens STDP window |

### Dopamine as reward prediction error

In modern RL theory (Schultz et al. 1997), dopamine signals the **reward prediction
error** (RPE) — the difference between received and expected reward:

$$\delta = r + \gamma V(s') - V(s)$$

Our model uses raw reward as the DA signal. For TD-learning integration, compute
RPE externally and pass it as the `reward` parameter.

### Temporal credit assignment window

The eligibility trace provides a finite temporal window for credit assignment.
The effective window duration depends on τ_e:

| τ_e (ms) | Window (to 10% of peak) | Suitable for |
|----------|------------------------|-------------|
| 100 | ~230 ms | Fast sensorimotor tasks |
| 500 | ~1.15 s | Standard conditioning |
| 1000 (default) | ~2.3 s | Delayed reward RL |
| 5000 | ~11.5 s | Long-delay tasks |

Beyond the window, the eligibility trace has decayed below 10% of its peak,
and the reward signal has minimal effect on weight. This naturally limits
which past events can be credited for the current reward.

### Dopamine timescale

The dopamine decay constant τ_DA = 200 ms models the reuptake and degradation
of dopamine in the synaptic cleft. Key physiological data:

| Measurement | Value | Source |
|-------------|-------|--------|
| DA clearance in striatum | ~200 ms | Garris et al. 1994 |
| DA clearance in PFC | ~500 ms | Sesack et al. 1998 |
| Phasic DA burst duration | ~200 ms | Schultz 1998 |
| Tonic DA level | Constant baseline | Grace 1991 |

Our default τ_DA = 200 ms matches striatal clearance. For prefrontal cortex
models, increase to 500 ms. Tonic dopamine can be modelled by adding a
constant baseline to the reward signal.

### Comparison with other learning rules

| Rule | Factors | Reward signal | Temporal credit | Reference |
|------|---------|--------------|----------------|-----------|
| Standard STDP | 2 (pre, post) | None | Immediate only | Bi & Poo 2001 |
| R-STDP | 2 + reward | Direct weight scaling | Immediate + reward | Florian 2007 |
| **DA-STDP** | **3 (pre, post, DA)** | **Eligibility trace** | **~1 s delay** | **Izhikevich 2007** |
| e-prop | 3 (pre, post, signal) | Learning signal | ~100 ms | Bellec 2020 |
| BPTT + surrogate | N/A (gradient) | Loss function | Full sequence | Neftci 2019 |

### Applications

1. **Reinforcement learning in SNNs:** Weight updates gated by reward signal
2. **Robotics:** Delayed reward for motor learning tasks
3. **Decision-making circuits:** Basal ganglia-like reward-modulated plasticity
4. **Pavlovian conditioning:** Stimulus-reward association with delay
5. **Addiction modelling:** Aberrant dopamine signalling → excessive potentiation
6. **Cognitive control:** Prefrontal cortex DA-modulated working memory

---

## 3. Pipeline Position

```
Pre spike ──────────┐
                    │
Post spike ───────���─┤
                    │
Reward signal ──────┤
                    ▼
┌──────────────────────────────────────────┐
│          DopamineStdpSynapse              │
│                                          │
│  ┌──────────┐  ┌───────────┐             │
│  │trace_pre │  │trace_post │             │
│  │ (decay)  │  │ (decay)   │             │
│  └────┬─────┘  └────┬──────┘             │
│       │              │                    │
│  ┌────▼───���──────────▼──��───┐             │
│  │  Eligibility trace e     │             │
│  │  e += a+·trace_pre (post)│             │
│  │  e += a-·trace_post (pre)│             │
│  └──────────┬───────────────┘             │
│             │                             │
│  ┌──────────▼─────��─────────┐             │
│  │  Dopamine DA             │             │
│  │  DA += (-DA/τ + reward)  │             │
│  └──────────┬───────────────┘             │
│             │                             │
│  ┌──────────▼───────────────┐             │
│  │  dw = lr · DA · e · dt   │             │
│  │  w = clamp(w + dw)       │             │
│  └───��──────────────────────┘             │
└──────────────────────────────────────────┘
    │
    ▼
Updated weight (float)
```

### Inputs

| Input | Type | Range | Description |
|-------|------|-------|-------------|
| `pre_spike` | `bool` | True/False | Presynaptic spike occurred |
| `post_spike` | `bool` | True/False | Postsynaptic spike occurred |
| `reward` | `float` | $(-\infty, +\infty)$ | Reward signal (positive = reward, negative = punishment) |

### Outputs

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| `weight` | `float` | $[w_{min}, w_{max}]$ | Current synaptic weight |

---

## 4. Features

| Feature | Description |
|---------|-------------|
| **Three-factor rule** | Hebbian (STDP) × reward (DA) × rate (lr) |
| **Eligibility trace** | Long-lived tag (~1s) bridging spike-reward delay |
| **Dopamine dynamics** | Integrates reward with decay τ_DA |
| **Bidirectional STDP** | Pre→post = LTP (a+), Post→pre = LTD (a-) |
| **Weight clamping** | Hard bounds [w_min, w_max] |
| **Configurable timescales** | Independent τ for pre/post traces, eligibility, DA |
| **Exponential trace decay** | Biologically realistic decay via exp(-dt/τ) |
| **Rust parity** | Identical equations to Rust implementation |

---

## 5. Usage Examples

### Basic reward-modulated learning

```python
from sc_neurocore.synapses import DopamineStdpSynapse

syn = DopamineStdpSynapse(weight=0.5, lr=0.01)

# Phase 1: STDP pairing (no reward).
for t in range(100):
    syn.step(pre_spike=(t%10==0), post_spike=(t%10==2), reward=0.0)
print(f"After pairing: e={syn.eligibility:.4f}, w={syn.weight:.4f}")

# Phase 2: Delayed reward.
for t in range(200):
    syn.step(pre_spike=False, post_spike=False, reward=1.0 if t<10 else 0.0)
print(f"After reward: w={syn.weight:.4f}")
```

### Demonstrating distal reward

```python
syn = DopamineStdpSynapse(weight=0.5, lr=0.01)

# Spike pairing at t=0.
syn.step(True, False, 0.0)
syn.step(False, True, 0.0)
w_before = syn.weight
print(f"After spikes (no reward): w={w_before:.4f}")

# Reward arrives 500ms later.
for _ in range(500): syn.step(False, False, 0.0)
for _ in range(50): syn.step(False, False, reward=2.0)
print(f"After delayed reward: w={syn.weight:.4f}")
assert syn.weight != w_before, "Delayed reward must change weight"
```

### Punishment (negative reward)

```python
syn = DopamineStdpSynapse(weight=0.5, lr=0.01)

# Pairing then punishment.
for t in range(50):
    syn.step(t%5==0, t%5==1, 0.0)
for _ in range(100):
    syn.step(False, False, reward=-1.0)
print(f"After punishment: w={syn.weight:.4f} (should decrease)")
```

### Eligibility trace time course

```python
syn = DopamineStdpSynapse(weight=0.5)
syn.step(True, False, 0.0)
syn.step(False, True, 0.0)

for delay_ms in [0, 100, 500, 1000, 2000]:
    syn2 = DopamineStdpSynapse(weight=0.5)
    syn2.step(True, False, 0.0)
    syn2.step(False, True, 0.0)
    for _ in range(delay_ms):
        syn2.step(False, False, 0.0)
    print(f"Delay={delay_ms:4d}ms: e={syn2.eligibility:.6f}")
```

### TD-learning integration

```python
import math

# External value function V(s) for reward prediction error.
def compute_rpe(reward, v_current, v_next, gamma=0.99):
    return reward + gamma * v_next - v_current

syn = DopamineStdpSynapse(weight=0.5, lr=0.005)
v_estimates = [0.0, 0.0, 0.0]  # simple state values

for episode in range(50):
    # State 0 → state 1 → state 2 (terminal, reward=1)
    syn.step(True, False, 0.0)  # pre spike at state 0
    syn.step(False, True, 0.0)  # post spike at state 1
    rpe = compute_rpe(1.0, v_estimates[1], 0.0)  # terminal reward
    for _ in range(50):
        syn.step(False, False, reward=rpe if _ < 5 else 0.0)

print(f"After 50 episodes: w={syn.weight:.4f}")
```

### Multiple synapses with shared dopamine

```python
# Multiple synapses receiving the same reward signal but different spike patterns.
synapses = [DopamineStdpSynapse(weight=0.5, lr=0.005) for _ in range(5)]

for t in range(500):
    reward = 1.0 if t == 400 else 0.0  # single reward at t=400
    for i, syn in enumerate(synapses):
        # Each synapse has different pre/post timing.
        pre = (t % (10 + i*3) == 0)
        post = (t % (10 + i*3) == 2)
        syn.step(pre, post, reward)

weights = [s.weight for s in synapses]
print(f"Weights after learning: {[f'{w:.4f}' for w in weights]}")
```

---

## 6. Technical Reference

### Class: `DopamineStdpSynapse`

Decorated with `@dataclass`. Defined in `src/sc_neurocore/synapses/dopamine_stdp.py`.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight` | `float` | `0.5` | Initial synaptic weight |
| `w_min` | `float` | `0.0` | Minimum weight |
| `w_max` | `float` | `1.0` | Maximum weight |
| `tau_e` | `float` | `1000.0` | Eligibility trace time constant (ms) |
| `tau_da` | `float` | `200.0` | Dopamine decay time constant (ms) |
| `tau_pre` | `float` | `20.0` | Pre-synaptic trace time constant (ms) |
| `tau_post` | `float` | `20.0` | Post-synaptic trace time constant (ms) |
| `a_plus` | `float` | `1.0` | LTP amplitude |
| `a_minus` | `float` | `-1.0` | LTD amplitude (negative) |
| `lr` | `float` | `0.001` | Learning rate |
| `dt` | `float` | `1.0` | Integration timestep (ms) |

#### State Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `eligibility` | `float` | `0.0` | Eligibility trace |
| `dopamine` | `float` | `0.0` | Dopamine concentration |
| `trace_pre` | `float` | `0.0` | Pre-synaptic spike trace |
| `trace_post` | `float` | `0.0` | Post-synaptic spike trace |

#### Methods

**`step(pre_spike: bool, post_spike: bool, reward: float) -> float`** — Returns weight.
**`reset() -> None`** — Reset eligibility, dopamine, traces to 0.

### Rust parity: identical equations including exp(-dt/τ) trace decay.

---

## 7. Performance Benchmarks

### Python (i5-11600K, single core, CPython 3.12)

| Method | Time per step | Steps/second |
|--------|--------------|--------------|
| `step()` (no spikes) | ~1,500 ns | 667,000 |
| `step()` (with spikes) | 1,706 ns | 586,000 |

Slower than STP due to three `math.exp()` calls for trace decays.

### Rust: ~5 ns/step, ~341× speedup

### Memory: ~250 bytes (Python), 112 bytes (Rust, 14× f64)

---

## 8. Citations

1. **Izhikevich, E. M.** "Solving the distal reward problem through linkage of STDP
   and dopamine signaling." Cerebral Cortex 17(10):2443-2452, 2007.
   — Source paper for all equations: eligibility trace + DA gating.

2. **Schultz, W. et al.** "A neural substrate of prediction and reward." Science
   275(5306):1593-1599, 1997.
   — Dopamine as reward prediction error signal.

3. **Yagishita, S. et al.** "A critical time window for dopamine actions on the
   structural plasticity of dendritic spines." Science 345(6204):1616-1620, 2014.
   — Experimental evidence for 0.3-2 s eligibility window in striatal synapses.

4. **Pawlak, V. & Kerr, J. N. D.** "Dopamine receptor activation is required for
   corticostriatal spike-timing-dependent plasticity." Journal of Neuroscience
   28(10):2435-2446, 2008.
   — DA modulates STDP window in corticostriatal connections.

5. **Florian, R. V.** "Reinforcement learning through modulation of spike-timing-
   dependent synaptic plasticity." Neural Computation 19(6):1468-1502, 2007.
   — Earlier R-STDP model with direct reward modulation (no eligibility trace).

6. **Bellec, G. et al.** "A solution to the learning dilemma for recurrent networks
   of spiking neurons." Nature Communications 11(1):3625, 2020.
   — e-prop: modern three-factor rule with broadcast learning signal.

---

## Validation

| Test | What it verifies | Status |
|------|-----------------|--------|
| `test_defaults` | tau_e=1000, tau_da=200, a_plus=1, a_minus=-1 | PASS |
| `test_step_returns_float` | Output is float | PASS |
| `test_no_reward_no_weight_change` | w stable without DA | PASS |
| `test_reward_drives_weight_change` | w changes with DA | PASS |
| `test_eligibility_trace_builds` | e != 0 after spikes | PASS |
| `test_eligibility_decays` | e → 0 after long silence | PASS |
| `test_dopamine_integrates_reward` | DA > 0 after reward | PASS |
| `test_dopamine_decays` | DA → 0 without reward | PASS |
| `test_weight_clamped` | w stays in [w_min, w_max] | PASS |
| `test_reset` | All traces → 0 | PASS |
| `test_distal_reward_problem` | Delayed reward changes weight | PASS |

### Equation-to-code traceability

| Equation | Python | Rust |
|----------|--------|------|
| trace_pre decay | `dopamine_stdp.py:88` | `synapses/mod.rs:382` |
| trace_post decay | `dopamine_stdp.py:89` | `synapses/mod.rs:383` |
| eligibility decay | `dopamine_stdp.py:90` | `synapses/mod.rs:384` |
| DA dynamics | `dopamine_stdp.py:91` | `synapses/mod.rs:385` |
| LTD (pre) | `dopamine_stdp.py:94-95` | `synapses/mod.rs:388-390` |
| LTP (post) | `dopamine_stdp.py:99-100` | `synapses/mod.rs:393-395` |
| Weight update | `dopamine_stdp.py:104-105` | `synapses/mod.rs:399-400` |

---

## Design Decisions

### Why exponential decay for traces instead of step function?

Exponential decay $\exp(-dt/\tau)$ provides smooth, biologically realistic attenuation.
Step functions (traces that persist for a fixed duration then vanish) create discontinuous
dynamics that complicate gradient computation and produce non-smooth learning curves.

### Why a_minus is negative by default?

The default $a^- = -1.0$ ensures that pre-before-post pairings contribute negative
eligibility (LTD), matching the classical STDP window. The sign is embedded in the
amplitude rather than the update rule, keeping the update equation simple:
$e += a^- \cdot \text{trace}_{post}$ (always additive, sign handled by $a^-$).

### Why separate tau_pre and tau_post?

Asymmetric STDP windows (LTP faster than LTD, or vice versa) require independent
time constants. The default $\tau_{pre} = \tau_{post} = 20$ ms produces a symmetric
window, matching Bi & Poo (1998). Setting $\tau_{post} > \tau_{pre}$ would broaden the
LTD window, modelling inhibitory STDP or anti-Hebbian learning.

---

## Known Limitations

1. **No reward prediction error:** Uses raw reward, not RPE. For TD-learning, compute
   δ = r + γ·V(s') - V(s) externally and pass as reward.

2. **Global dopamine:** All synapses receive the same DA signal. In biology, DA is
   spatially heterogeneous (mesolimbic vs mesocortical vs nigrostriatal pathways).

3. **No dopamine receptor subtypes:** D1 and D2 receptors have opposite effects on
   plasticity. Our model uses a single DA variable.

4. **No homeostatic bounds:** Weight can drift to w_min or w_max without synaptic
   scaling. Add a homeostatic term for stable long-term learning.

5. **Linear DA-eligibility interaction:** The product DA·e assumes linear gating.
   Experimental data suggests sigmoidal or thresholded gating.

6. **No serotonin/acetylcholine:** Other neuromodulators (5-HT, ACh, NE) also
   gate plasticity. He et al. (2015) showed cholinergic three-factor rules in cortex.

7. **No eligibility trace variability:** All synapses share the same τ_e.
   In biology, eligibility windows vary by brain region (100ms in cerebellum,
   1-2s in striatum, possibly longer in hippocampus).

---

*SC-NeuroCore v3.14.0 — Stochastic Computing Spiking Neural Network Framework*

*© 2020–2026 Miroslav Šotek / ANULUM. AGPL-3.0-or-later.*
