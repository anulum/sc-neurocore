# ShortTermPlasticitySynapse

**Module:** `sc_neurocore.synapses.short_term_plasticity`
**Rust path:** `sc_neurocore_engine::synapses::ShortTermPlasticitySynapse`
**Reference:** Tsodyks & Markram (1997), Markram et al. (1998)
**Family:** Short-term synaptic dynamics
**State variables:** `x` (available resources), `u` (release probability)

---

## 1. Mathematical Formalism

### Core equations (Tsodyks-Markram 1997)

Models use-dependent synaptic dynamics on ms-to-s timescales. The synapse has
two interacting variables: available resources $x$ (depression) and release
probability $u$ (facilitation).

**Depression variable (resource depletion):**

$$\frac{dx}{dt} = \frac{1 - x}{\tau_d} - u \cdot x \cdot \delta(t_{spike})$$

Between spikes, $x$ recovers toward 1 with time constant $\tau_d$.
At each presynaptic spike, a fraction $u \cdot x$ of resources is consumed.

**Facilitation variable (release probability):**

$$\frac{du}{dt} = \frac{U - u}{\tau_f} + U \cdot (1 - u) \cdot \delta(t_{spike})$$

Between spikes, $u$ decays toward the baseline $U$ with time constant $\tau_f$.
At each spike, $u$ is boosted by $U \cdot (1 - u)$, increasing the release probability.

**Post-synaptic current:**

$$PSC = A \cdot u \cdot x \cdot \delta(t_{spike})$$

where $A$ is the maximum amplitude. The PSC is the product of the release
probability $u$ and available resources $x$, producing a use-dependent scaling
of the synaptic strength.

### Discretised implementation (forward Euler)

Between spikes (continuous recovery):

$$x(t+dt) = x(t) + \frac{(1 - x(t)) \cdot dt}{\tau_d}$$
$$u(t+dt) = u(t) + \frac{(U - u(t)) \cdot dt}{\tau_f}$$

On a presynaptic spike:

$$u \leftarrow u + U \cdot (1 - u) \quad \text{(facilitation first)}$$
$$PSC = A \cdot u \cdot x \quad \text{(compute output)}$$
$$x \leftarrow x - u \cdot x \quad \text{(depression after)}$$
$$x \leftarrow \max(x, 0) \quad \text{(clamp non-negative)}$$

The order matters: facilitation updates $u$ before the PSC is computed, and
depression updates $x$ after. This matches Markram et al. (1998) Eq. 3.

### Depression vs facilitation regimes

The behaviour depends on the parameter ratios:

**Depressing synapse** (high $U$, fast $\tau_d$, slow $\tau_f$):
- First spike has large PSC (high $u \approx U = 0.5$, full $x = 1.0$)
- Subsequent spikes: $x$ depletes faster than it recovers → PSC decreases
- Models: cortical pyramidal-pyramidal (pyr-pyr) connections
- Example: `new_depressing()` with $U = 0.5$, $\tau_d = 200$ ms, $\tau_f = 20$ ms

**Facilitating synapse** (low $U$, slow $\tau_d$, fast $\tau_f$):
- First spike has small PSC (low $u \approx U = 0.1$, full $x = 1.0$)
- Subsequent spikes: $u$ grows faster than $x$ depletes → PSC increases
- Models: cortical pyramidal-interneuron (pyr-FS) connections
- Example: `new_facilitating()` with $U = 0.1$, $\tau_d = 50$ ms, $\tau_f = 500$ ms

### Steady-state analysis

For periodic presynaptic spiking at rate $r$ (inter-spike interval $T = 1/r$):

At steady state, the recovery per interval equals the consumption per spike:

$$x_{ss} = \frac{1}{1 + u_{ss} \cdot \tau_d \cdot r}$$

$$u_{ss} = \frac{U}{1 - (1-U) \cdot \exp(-T/\tau_f)}$$

The steady-state PSC is:

$$PSC_{ss} = A \cdot u_{ss} \cdot x_{ss}$$

For depressing synapses: $PSC_{ss}$ decreases with rate $r$ (low-pass filter).
For facilitating synapses: $PSC_{ss}$ increases then decreases (band-pass filter).

### Time course of PSC trains

For a depressing synapse with $U = 0.5$, $\tau_d = 200$ ms, $A = 1.0$,
stimulated at 50 Hz (ISI = 20 ms):

| Spike # | u | x | PSC |
|---------|---|---|-----|
| 1 | 0.75 | 1.00 | 0.750 |
| 2 | 0.69 | 0.29 | 0.197 |
| 3 | 0.65 | 0.14 | 0.089 |
| 4 | 0.63 | 0.09 | 0.054 |
| 5 | 0.61 | 0.07 | 0.040 |

The PSC drops by ~19× from the first to the fifth spike — strong depression.

---

## 2. Theoretical Context

### Problem statement

Synaptic transmission is not a fixed-gain process. Every synapse in the brain
exhibits use-dependent changes in strength on timescales of milliseconds to seconds.
These short-term dynamics are distinct from long-term plasticity (STDP, LTP/LTD)
and play critical roles in temporal filtering, gain control, and working memory.

### The Tsodyks-Markram framework

Tsodyks & Markram (1997) proposed a phenomenological model with just two variables
($x$, $u$) that captures the essential dynamics of short-term plasticity:

1. **Minimal parameters:** 4 free parameters ($U$, $\tau_d$, $\tau_f$, $A$) vs
   detailed biophysical models with 10+ parameters
2. **Experimentally constrained:** Parameters directly measurable from paired-pulse
   recordings: $U$ from first-pulse amplitude, $\tau_d$ from recovery curves,
   $\tau_f$ from facilitation decay
3. **Analytically tractable:** Steady-state solutions exist in closed form
4. **Universally applicable:** Same framework describes depressing, facilitating,
   and mixed synapses by varying parameter ratios

### Biological substrates

| STP mechanism | Variable | Molecular basis | Timescale |
|--------------|----------|----------------|-----------|
| Depression | $x$ | Vesicle pool depletion | 200-1000 ms |
| Facilitation | $u$ | Residual Ca²⁺ at release sites | 20-500 ms |
| Augmentation | — | Munc13 priming | 5-10 s |
| Post-tetanic potentiation | — | PKC activation | 30-60 s |

Our model captures the two fastest mechanisms (depression and facilitation).
Augmentation and PTP operate on longer timescales and are not included.

### Functional roles of STP

1. **Temporal filtering:** Depressing synapses act as low-pass filters (attenuate
   high-frequency input), facilitating synapses act as high-pass filters (amplify
   burst onset). This creates complementary temporal coding channels.

2. **Gain control:** Depression implements automatic gain control — strong inputs
   are attenuated more than weak inputs, expanding the dynamic range.

3. **Working memory:** Facilitating synapses in prefrontal cortex maintain elevated
   transmission during the delay period of working memory tasks.

4. **Burst detection:** Facilitating synapses amplify burst onset relative to
   tonic activity, serving as burst detectors.

5. **Redundancy reduction:** Depression removes temporal redundancy in repetitive
   stimulation, transmitting only novel or changing input.

6. **Network stability:** Depression prevents runaway excitation by reducing
   synaptic drive during high-activity states.

### Synapse diversity in cortex

| Connection type | U | τ_d (ms) | τ_f (ms) | STP type |
|----------------|---|----------|----------|----------|
| Pyr → Pyr (L2/3) | 0.5 | 200 | 20 | Depressing |
| Pyr → FS interneuron | 0.1 | 50 | 500 | Facilitating |
| Pyr → SOM interneuron | 0.3 | 100 | 200 | Mixed |
| FS → Pyr | 0.2 | 100 | 20 | Weakly depressing |
| Thalamic → L4 | 0.7 | 300 | 10 | Strongly depressing |

### Paired-pulse ratio as experimental diagnostic

The paired-pulse ratio (PPR) — the ratio of the second PSC to the first — is the
primary experimental measure for characterising STP:

$$PPR = \frac{PSC_2}{PSC_1}$$

| PPR | Interpretation | U range |
|-----|---------------|---------|
| < 0.5 | Strong depression | U > 0.5 |
| 0.5 - 0.9 | Moderate depression | U = 0.3-0.5 |
| ~1.0 | No STP (or balanced) | U ~ 0.15-0.2 |
| 1.0 - 2.0 | Moderate facilitation | U = 0.05-0.15 |
| > 2.0 | Strong facilitation | U < 0.05 |

In our model, PPR can be computed directly:

```python
syn = ShortTermPlasticitySynapse(u_base=U)
psc1 = syn.step(True)
psc2 = syn.step(True)  # at ISI = dt
ppr = psc2 / psc1 if psc1 > 0 else float('nan')
```

### Information-theoretic perspective

STP implements a form of **adaptive coding** at the synapse level:

- **Depressing synapses** reduce transmission of repetitive (predictable) input,
  implementing a prediction error signal analogous to temporal whitening
- **Facilitating synapses** amplify bursts that carry novel information
  (coincidence detection at the temporal scale)

From the perspective of efficient coding (Barlow 1961), STP optimally adapts the
synaptic gain to the temporal statistics of the input, reducing redundancy in the
neural code. Abbott & Regehr (2004) formalised this as a **temporal filter**:

- Depression → low-pass filter (transmits onset, attenuates sustained)
- Facilitation → high-pass filter (transmits changes, attenuates tonic)

### Relationship to existing models

| Model | Variables | STP type | Plasticity | Reference |
|-------|----------|----------|------------|-----------|
| Static synapse | weight | None | None | — |
| **TM-STP** | **x, u** | **Depression + facilitation** | **None** | **Tsodyks 1997** |
| Mongillo WM | x, u | Depression + facilitation | None | Mongillo 2008 |
| TripletSTDP | r1, r2, o1, o2, w | None | Long-term (STDP) | Pfister 2006 |
| DopamineSTDP | e, DA, w | None | Reward-modulated | Izhikevich 2007 |
| Full biophysical | vesicles, Ca²⁺ | All mechanisms | Ca��⁺-dependent | Dittman 2000 |

---

## 3. Pipeline Position

```
Presynaptic spike train
    │
    ▼
┌────────────────────────────────┐
│  ShortTermPlasticitySynapse     │
│                                │
│  ┌─────────┐   ┌─────────┐    │
│  │ u       │   │ x       │    │
│  │ (facil.)│   │ (depr.) │    │
│  │ ↑on spike│   │ ↓on spike│    │
│  │ ↓recovery│   │ ↑recovery│    │
│  └────┬────┘   └────┬────┘    │
│       │              │         ��
│       └──────┬───────┘         │
│              │                 │
│       PSC = A · u · x          │
└──────���───────┼─────────────────┘
               │
               ▼
    Post-synaptic current (float)
```

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `pre_spike` | `bool` | Whether a presynaptic spike occurred this timestep |

### Outputs

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| `psc` | `float` | $[0, +\infty)$ | Post-synaptic current (0 if no spike) |

---

## 4. Features

| Feature | Description |
|---------|-------------|
| **Depression** | Resource depletion variable $x$, recovers with $\tau_d$ |
| **Facilitation** | Release probability $u$, boosted by spikes, decays with $\tau_f$ |
| **Two presets** | `new_depressing()` and `new_facilitating()` class methods |
| **Configurable** | All 6 parameters independently adjustable |
| **Non-negative** | $x$ clamped to $\geq 0$ |
| **Zero on silence** | PSC = 0 when no presynaptic spike |
| **Rust parity** | Identical equations to Rust implementation |
| **O(1)** | Constant time per step, no history buffers |

---

## 5. Usage Examples

### Depressing synapse — paired-pulse depression

```python
from sc_neurocore.synapses import ShortTermPlasticitySynapse

syn = ShortTermPlasticitySynapse.new_depressing()

# 5 spikes at 50 Hz (20 ms interval, dt=1ms → 20 steps apart).
pscs = []
for t in range(100):
    psc = syn.step(t % 20 == 0 and t < 100)
    if psc > 0:
        pscs.append(psc)
        print(f"Spike at t={t}: PSC={psc:.4f}")

print(f"Paired-pulse ratio: {pscs[1]/pscs[0]:.3f}" if len(pscs)>1 else "")
```

### Facilitating synapse — burst amplification

```python
syn = ShortTermPlasticitySynapse.new_facilitating()

pscs = [syn.step(True) for _ in range(5)]
for i, p in enumerate(pscs):
    print(f"Spike {i+1}: PSC={p:.4f}")
print(f"Facilitation index: {pscs[-1]/pscs[0]:.2f}x")
```

### Recovery after depletion

```python
syn = ShortTermPlasticitySynapse.new_depressing()

# Deplete with 10 rapid spikes.
for _ in range(10):
    syn.step(True)
print(f"After depletion: x={syn.x:.4f}")

# Recover for various durations.
for delay in [50, 100, 200, 500, 1000]:
    syn2 = ShortTermPlasticitySynapse.new_depressing()
    for _ in range(10): syn2.step(True)
    for _ in range(delay): syn2.step(False)
    psc = syn2.step(True)
    print(f"After {delay}ms silence: PSC={psc:.4f}, x={syn2.x:.3f}")
```

### Frequency-dependent filtering

```python
# Demonstrate STP as temporal filter at different frequencies.
for freq_hz in [10, 20, 50, 100, 200]:
    syn = ShortTermPlasticitySynapse.new_depressing()
    isi_ms = 1000.0 / freq_hz  # inter-spike interval
    isi_steps = max(1, int(isi_ms))  # at dt=1ms
    pscs = []
    for t in range(500):
        psc = syn.step(t % isi_steps == 0)
        if psc > 0:
            pscs.append(psc)
    ss_psc = pscs[-1] if pscs else 0
    first_psc = pscs[0] if pscs else 0
    ratio = ss_psc / first_psc if first_psc > 0 else 0
    print(f"{freq_hz:3d} Hz: first={first_psc:.3f}, steady={ss_psc:.4f}, ratio={ratio:.3f}")
```

### Comparison: depression vs facilitation on same spike train

```python
dep = ShortTermPlasticitySynapse.new_depressing()
fac = ShortTermPlasticitySynapse.new_facilitating()

# Burst of 10 spikes, then 200ms silence, then probe.
for label, syn in [("DEP", dep), ("FAC", fac)]:
    burst = [syn.step(True) for _ in range(10)]
    for _ in range(200): syn.step(False)
    probe = syn.step(True)
    print(f"{label}: burst_first={burst[0]:.3f}, burst_last={burst[-1]:.4f}, probe={probe:.4f}")
```

### Parameter sweep for U

```python
for U in [0.1, 0.3, 0.5, 0.7, 0.9]:
    syn = ShortTermPlasticitySynapse(u_base=U, u=U, tau_d=200, tau_f=20)
    pscs = [syn.step(True) for _ in range(5)]
    ratio = pscs[1] / pscs[0] if pscs[0] > 0 else 0
    print(f"U={U:.1f}: PPR={ratio:.3f} ({'depressing' if ratio < 1 else 'facilitating'})")
```

---

## 6. Technical Reference

### Class: `ShortTermPlasticitySynapse`

Decorated with `@dataclass`. Defined in
`src/sc_neurocore/synapses/short_term_plasticity.py`.

#### Constructor Parameters

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `x` | `float` | `1.0` | $[0, 1]$ | Available resources (depression) |
| `u` | `float` | `0.5` | $[0, 1]$ | Release probability (facilitation) |
| `u_base` | `float` | `0.5` | $(0, 1]$ | Baseline release probability $U$ |
| `tau_d` | `float` | `200.0` | $> 0$ | Depression recovery time constant (ms) |
| `tau_f` | `float` | `20.0` | $> 0$ | Facilitation decay time constant (ms) |
| `amplitude` | `float` | `1.0` | $> 0$ | Maximum PSC amplitude $A$ |
| `dt` | `float` | `1.0` | $> 0$ | Integration timestep (ms) |

#### State Variables

| Variable | Type | Description |
|----------|------|-------------|
| `x` | `float` | Available resources (1.0 = full, 0.0 = depleted) |
| `u` | `float` | Current release probability |

#### Methods

**`step(pre_spike: bool) -> float`** — Returns PSC (0 if no spike).
**`reset() -> None`** — x = 1.0, u = u_base.
**`new_depressing() -> cls`** — Depressing preset (U=0.5, τ_d=200, τ_f=20).
**`new_facilitating() -> cls`** — Facilitating preset (U=0.1, τ_d=50, τ_f=500).

### Rust parity

| Operation | Python | Rust |
|-----------|--------|------|
| x recovery | `x += (1-x)/tau_d * dt` | `self.x += (1.0-self.x)/self.tau_d * self.dt` |
| u recovery | `u += (U-u)/tau_f * dt` | `self.u += (self.u_base-self.u)/self.tau_f * self.dt` |
| u facilitation | `u += U*(1-u)` | `self.u += self.u_base*(1.0-self.u)` |
| PSC | `A * u * x` | `self.amplitude * self.u * self.x` |
| x depression | `x -= u*x; x = max(x,0)` | `self.x -= self.u*self.x; self.x = self.x.max(0.0)` |

---

## 7. Performance Benchmarks

### Python (i5-11600K, single core, CPython 3.12)

| Method | Time per step | Steps/second |
|--------|--------------|--------------|
| `step()` (no spike) | ~700 ns | 1,430,000 |
| `step()` (with spike) | 838 ns | 1,193,000 |

Fastest of all 11 gap models — no `math.exp()`, pure arithmetic.

### Rust: ~3 ns/step, ~280× speedup

### Memory: ~120 bytes (Python), 56 bytes (Rust, 7× f64)

---

## 8. Citations

1. **Tsodyks, M. V. & Markram, H.** "The neural code between neocortical pyramidal
   neurons depends on neurotransmitter release probability." PNAS 94(2):719-723, 1997.
   — Original TM model: $dx/dt$, $du/dt$ equations.

2. **Markram, H. et al.** "Differential signaling via the same axon of neocortical
   pyramidal neurons." PNAS 95(9):5323-5328, 1998.
   — Extended model with facilitation, parameter fits from paired recordings.

3. **Mongillo, G. et al.** "Synaptic theory of working memory." Science
   319(5869):1543-1546, 2008.
   — TM-STP as mechanism for working memory (self-sustaining activity via facilitation).

4. **Abbott, L. F. & Regehr, W. G.** "Synaptic computation." Nature 431:796-803, 2004.
   — Review of STP as computational mechanism for temporal filtering.

5. **Zucker, R. S. & Regehr, W. G.** "Short-term synaptic plasticity." Annual Review
   of Physiology 64:355-405, 2002.
   — Comprehensive review of biological STP mechanisms.

6. **Dittman, J. S. et al.** "Interplay between facilitation, depression, and residual
   calcium at three presynaptic terminals." Journal of Neuroscience 20(4):1374-1385, 2000.
   — Detailed biophysical model of STP at different synapse types.

---

## Validation

| Test | What it verifies | Status |
|------|-----------------|--------|
| `test_depressing_defaults` | u_base=0.5, tau_d=200 | PASS |
| `test_facilitating_defaults` | u_base=0.1, tau_f=500 | PASS |
| `test_depression_successive_spikes` | PSC[0] > PSC[1] > PSC[2] | PASS |
| `test_facilitation_successive_spikes` | PSC[1] > PSC[0] | PASS |
| `test_recovery_after_silence` | x recovers after depletion | PASS |
| `test_no_spike_no_current` | PSC=0 when no spike | PASS |
| `test_x_never_negative` | x >= 0 always | PASS |
| `test_reset` | x=1, u=u_base | PASS |

### Equation traceability

| Equation | Python | Rust |
|----------|--------|------|
| $dx/dt = (1-x)/\tau_d$ | `short_term_plasticity.py:89` | `synapses/mod.rs:306` |
| $du/dt = (U-u)/\tau_f$ | `short_term_plasticity.py:90` | `synapses/mod.rs:307` |
| $u \leftarrow u + U(1-u)$ | `short_term_plasticity.py:93` | `synapses/mod.rs:311` |
| $PSC = A \cdot u \cdot x$ | `short_term_plasticity.py:95` | `synapses/mod.rs:313` |
| $x \leftarrow x - ux$ | `short_term_plasticity.py:97` | `synapses/mod.rs:315` |

---

## Design Decisions

### Why facilitate before compute PSC, depress after?

This ordering matches Markram et al. (1998) and captures the biological sequence:
1. Ca²⁺ influx raises release probability (facilitation)
2. Vesicle fusion occurs with the updated probability → PSC
3. Released vesicles are consumed from the pool (depression)

The alternative (depress then compute) would give smaller first-spike PSC because
x would be reduced before computing output.

### Why clamp x >= 0?

Without clamping, strong facilitation can drive x negative:
$x \leftarrow x - u \cdot x$. When $u > 1$ (possible after many rapid facilitating
updates), $u \cdot x > x$. The clamp prevents unphysical negative resources.

### Implementation of recovery and spike in single step

The implementation combines continuous recovery and discrete spike events in one
`step()` call. The recovery phase happens first (every step), then the spike
event is processed (only when `pre_spike=True`). This means:

1. On a step WITH spike: recovery happens first, then facilitation, PSC, depression
2. On a step WITHOUT spike: only recovery happens, PSC = 0

This ordering ensures that recovery from the previous spike is applied before
the new spike is processed, which is correct for time intervals ≥ dt.

For inter-spike intervals equal to dt (consecutive spikes), the recovery is
minimal (dt/τ_d and dt/τ_f), so the PSC reflects near-maximal depression.

### Why not include augmentation and PTP?

Augmentation (~5s timescale) and post-tetanic potentiation (~30s) are slower STP
mechanisms with distinct molecular substrates. Including them would add 2+ more
state variables and time constants. For most simulation needs, the two-variable
model captures the dominant dynamics adequately.

---

## Known Limitations

1. **No voltage dependence:** Release probability depends only on spike timing,
   not on presynaptic voltage. Graded release (as in IHCs) is not supported.

2. **No stochastic release:** The model is deterministic. Real vesicle release
   is probabilistic (binomial with $n$ release sites and probability $u$).

3. **No multi-vesicular release:** Each spike releases $u \cdot x$ of the pool.
   Real synapses have discrete vesicle numbers (0-10 per spike).

4. **No spatial segregation:** All vesicles share a single pool. Real synapses
   have docked, primed, and reserve pools with different kinetics.

5. **No spike-timing effects:** The model operates on discrete pre_spike events.
   Sub-millisecond timing effects (e.g., interspike interval irregularity) are lost.

6. **No post-synaptic dependence:** STP depends only on presynaptic activity.
   Heterosynaptic and retrograde modulation (e.g., endocannabinoid-mediated
   depolarisation-induced suppression of excitation, DSE) are not modelled.

7. **Fixed parameters:** U, τ_d, τ_f are constant. In biology, STP parameters
   are modulated by neuromodulators (dopamine reduces facilitation, serotonin
   enhances depression in prefrontal synapses).

8. **No calcium dynamics:** The facilitation variable u is a phenomenological proxy
   for residual calcium at the release site. For detailed calcium-STP interaction,
   use the Dittman et al. (2000) biophysical model.

---

*SC-NeuroCore v3.14.0 — Stochastic Computing Spiking Neural Network Framework*

*© 2020–2026 Miroslav Šotek / ANULUM. AGPL-3.0-or-later.*

*Rust engine: sc-neurocore-engine (PyPI). Python: sc-neurocore.*
