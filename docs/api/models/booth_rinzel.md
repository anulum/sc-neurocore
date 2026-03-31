# BoothRinzelNeuron

**Module:** `sc_neurocore.neurons.models.booth_rinzel`
**Reference:** Booth, Rinzel & Kiehn, J. Neurophysiol. 78(6), 1997 (model from Booth & Rinzel 1995 preprint)
**Family:** Multi-compartment biophysical (2-compartment bistable motoneuron)
**State variables:** `vs` (soma voltage), `vd` (dendrite voltage), `h` (Na⁺ inactivation), `n` (K⁺ activation), `q` (dendritic Ca²⁺ gating), `ca` (intracellular Ca²⁺)

---

## Equations

### Soma compartment

$$C_m \frac{dV_s}{dt} = -I_{Na}(V_s) - I_K(V_s) - I_{L,s}(V_s) - \frac{g_c(V_s - V_d)}{p} + \frac{I}{p}$$

### Dendrite compartment

$$C_m \frac{dV_d}{dt} = -I_{Ca}(V_d) - I_{K(Ca)}(V_d) - I_{L,d}(V_d) - \frac{g_c(V_d - V_s)}{1-p}$$

### Six ionic currents

**Soma:**
$$I_{Na} = g_{Na} \, m_\infty^3 \, h \, (V_s - E_{Na})$$
$$I_K = g_K \, n^4 \, (V_s - E_K)$$
$$I_{L,s} = g_L \, (V_s - E_L)$$

**Dendrite:**
$$I_{Ca} = g_{Ca} \, s_\infty^2 \, (V_d - E_{Ca})$$
$$I_{K(Ca)} = g_{K(Ca)} \, \chi \, (V_d - E_K), \quad \chi = \min([Ca]/250, 1)$$
$$I_{L,d} = g_L \, (V_d - E_L)$$

### Soma-dendrite coupling

$$I_{coup,s} = \frac{g_c(V_s - V_d)}{p}, \quad I_{coup,d} = \frac{g_c(V_d - V_s)}{1-p}$$

The coupling parameter $p = 0.5$ represents the fraction of total membrane
area attributed to the soma. Current conservation: $p \cdot I_{coup,s} +
(1-p) \cdot I_{coup,d} = 0$.

### Boltzmann activation functions

| Function | Midpoint (mV) | Slope (mV) | Compartment |
|----------|-------------|-----------|-------------|
| m_∞ | −35 | 7.8 | Soma (Na⁺ activation) |
| h_∞ | −55 | 7.0 | Soma (Na⁺ inactivation) |
| n_∞ | −28 | 15 | Soma (K⁺ activation) |
| s_∞ | −22 | 5.0 | Dendrite (Ca²⁺ activation) |
| q_∞ | −35 | 2.0 | Dendrite (Ca²⁺ gating, steep) |

### Gating dynamics

$$\tau_h = \frac{30}{\exp((V_s+50)/15) + \exp(-(V_s+50)/16) + 10^{-12}}$$
$$\tau_n = \frac{7}{\exp((V_s+40)/40) + \exp(-(V_s+40)/50) + 10^{-12}}$$
$$\tau_q = 400 \text{ ms (constant)}$$

### Ca²⁺ dynamics

$$\frac{d[Ca]}{dt} = f_{Ca} \cdot (-\alpha_{Ca} \cdot I_{Ca} - k_{Ca} \cdot [Ca])$$

### K(Ca) activation (linear with saturation)

$$\chi = \min([Ca]/250, 1)$$

Linear activation with ceiling at 1.0. Half-activation at [Ca] = 125.

### 4 sub-steps per call

Forward Euler with 4 sub-steps at dt=0.025 ms. Each call integrates
0.1 ms of biological time.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `vs` | −65.0 | mV | Soma potential |
| `vd` | −65.0 | mV | Dendrite potential |
| `h` | 0.9 | — | Na⁺ inactivation gate |
| `n` | 0.0 | — | K⁺ activation gate |
| `q` | 0.0 | — | Ca²⁺ dendritic gate |
| `ca` | 0.0 | µM | Intracellular Ca²⁺ |
| `p` | 0.5 | — | Soma area fraction |
| `gc` | 0.1 | mS/cm² | Coupling conductance |
| `g_na` | 120.0 | mS/cm² | Na⁺ conductance (soma) |
| `g_k` | 20.0 | mS/cm² | K⁺ delayed rectifier (soma) |
| `g_ca` | 14.0 | mS/cm² | Ca²⁺ conductance (dendrite) |
| `g_kca` | 5.0 | mS/cm² | Ca²⁺-activated K⁺ (dendrite) |
| `g_l` | 0.51 | mS/cm² | Leak (both compartments) |
| `e_na` | 55.0 | mV | Na⁺ reversal |
| `e_k` | −80.0 | mV | K⁺ reversal |
| `e_ca` | 80.0 | mV | Ca²⁺ reversal |
| `e_l` | −60.0 | mV | Leak reversal |
| `c_m` | 1.0 | µF/cm² | Membrane capacitance |
| `alpha_ca` | 0.009 | — | Ca²⁺ influx coupling |
| `k_ca` | 0.18 | ms⁻¹ | Ca²⁺ clearance rate |
| `f_ca` | 0.0025 | — | Ca²⁺ dynamics scaling |
| `dt` | 0.025 | ms | Sub-step timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

---

## Analytical Properties

### Bistability (the core feature)

The Booth-Rinzel model is the canonical **bistable motoneuron model:**

1. **Down state (quiescent):** V_s ≈ −65 mV, V_d ≈ −65 mV, Ca ≈ 0.
   Both compartments rest near E_L. The dendritic Ca²⁺ channels are
   deactivated (s_∞(−65) ≈ 0).

2. **Up state (plateau):** V_s oscillates (spiking), V_d ≈ −40 mV
   (Ca²⁺ plateau), Ca elevated. The dendritic Ca²⁺ current maintains
   the plateau, which drives the soma via coupling.

3. **Bistability:** Both states are stable for a range of input currents.
   A brief depolarising pulse can switch from down to up state, and a
   brief hyperpolarising pulse can switch back.

### Two-compartment division of labour

| Compartment | Channels | Role |
|-------------|----------|------|
| **Soma** | Na⁺ + K⁺ + leak | Action potential generation |
| **Dendrite** | Ca²⁺ + K(Ca) + leak | Plateau potential / bistability |

The soma produces fast Na⁺/K⁺ spikes (like HH). The dendrite produces
slow Ca²⁺ plateaus (like the AvRon cardiac model). Together, they create
a motoneuron that can fire continuously (up state) or rest (down state).

### Coupling conductance gc

$$g_c = 0.1 \text{ mS/cm²}$$

This is weak coupling relative to the ionic conductances (g_Na=120). The
weak coupling means:
- Soma and dendrite can have substantially different voltages
- The soma can spike while the dendrite is at plateau
- The dendrite can be at plateau while the soma is subthreshold

### Area fraction p = 0.5

Equal area split: soma = 50%, dendrite = 50%. Current and coupling terms
are divided by p (soma) or 1−p (dendrite). With p=0.5: symmetric.

### Ca²⁺ accumulates during plateau

During the dendritic plateau (V_d ≈ −40):
- s_∞(−40) = 1/(1+exp(−(−40+22)/5)) = 1/(1+exp(−3.6)) ≈ 0.96 (nearly full)
- I_Ca = 14 × 0.96² × (−40 − 80) = 14 × 0.92 × (−120) ≈ −1548 µA/cm²
  (strong inward current)
- Ca accumulates: dCa/dt = f_ca × (−α_ca × I_Ca − k_ca × Ca) > 0

The rising Ca activates K(Ca): χ = Ca/250 → I_K(Ca) outward → eventually
terminates the plateau.

---

## Behaviour

### Motoneuron plateau potentials

The model reproduces **plateau potentials** observed in spinal motoneurons:
- A brief excitatory input triggers a sustained firing state (up state)
- The firing continues after the input ceases (self-sustaining)
- A brief inhibitory input can terminate the firing (return to down state)

This is the neural basis of **persistent inward currents (PICs)** in
motoneurons, which contribute to:
- Sustained muscle contraction
- Motor unit recruitment/derecruitment
- Spinal cord injury pathophysiology (spasticity from PIC dysregulation)

### Very slow model (364.69s for 15 tests)

The model runs 4 sub-steps per call, each with:
- 8 Boltzmann evaluations (8 exp() per sub-step)
- 6 ionic current computations
- 6 state variable updates
- 4 clip operations

Total: 32 exp() + 24 currents + 24 state updates + 16 clips per step().
This makes it one of the slowest models in the library.

### Two compartments differ

Verified: V_s and V_d take different values during spiking. The soma
produces action potentials while the dendrite maintains a plateau.

### Ca²⁺ accumulates during firing

Verified: Ca > 0 after sustained drive. The Ca²⁺ influx through dendritic
Ca²⁺ channels builds up intracellular concentration.

### Gating variables bounded

All gates (h, n, q) are clipped to [0, 1] after each sub-step. V_s and
V_d are clipped to [−200, 100]. Ca is clipped to ≥ 0. These clips ensure
numerical robustness for the stiff 2-compartment system.

---

## Comparison with Related Models

| Property | BoothRinzel | NeuroGridNeuron | MainenSejnowski | TwoCompartmentLIF |
|----------|-----------|----------------|-----------------|-------------------|
| Compartments | 2 (soma+dend) | 2 (soma+dend) | 2 (soma+axon) | 2 (soma+dend) |
| State vars | 6 | 2 | 5 | 2 |
| Sub-steps | 4 | 10 | 20 | 1 |
| Bistable | Yes (plateau) | No | No | No |
| Ca²⁺ | Yes (dendritic) | No | No | No |
| K(Ca) | Yes (linear sat.) | No | No | No |
| Cell type | Motoneuron | Silicon neuron | Cortical | Generic |
| Speed | ~250 steps/s | ~500 steps/s | ~700 steps/s | ~500K steps/s |

BoothRinzel is the most biophysically detailed 2-compartment model in
SC-NeuroCore — the only one with dendritic Ca²⁺ channels and bistability.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
15/15 PASSED in 364.69s (6 min 4 s — slowest test file in the library)
├── TestBoothRinzelIsolation: 9 tests
│   ├── construction (vs=-65, vd=-65, h=0.9, n=0, q=0, ca=0)
│   ├── step() → int {0,1}
│   ├── spikes under drive
│   ├── two compartments differ (vs ≠ vd during spiking)
│   ├── Ca²⁺ accumulates
│   ├── bistability (up/down states)
│   ├── numerical stability (long run)
│   ├── gating bounded ([0,1])
│   └── reset()
├── TestBoothRinzelNetwork: 3 tests
│   ├── Population(n=10)
│   ├── Network + PoissonInput → spikes
│   └── Projection (pop→pop) → spike_trains extractable
└── TestBoothRinzelAnalysis: 3 tests
    ├── firing_rate > 0
    ├── spike_count > 0
    └── isi all > 0, all finite
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | 6 state vars initialised |
| step() → int {0,1} | ✓ PASS | 4 sub-steps per call |
| Two compartments evolve | ✓ PASS | vs ≠ vd during activity |
| Ca²⁺ accumulates | ✓ PASS | Ca > 0 after drive |
| Bistability | ✓ PASS | Up/down states demonstrated |
| State finite | ✓ PASS | All 6 vars finite (50k steps) |
| Gating bounded | ✓ PASS | h, n, q ∈ [0, 1] |
| reset() | ✓ PASS | All 6 vars restored |
| Population(n=10) | ✓ PASS | 10 instances |
| Network + PoissonInput | ✓ PASS | Spikes produced |
| Projection(pop→pop) | ✓ PASS | spike_trains extractable |
| firing_rate | ✓ PASS | > 0 Hz |
| spike_count | ✓ PASS | > 0 |
| isi | ✓ PASS | all > 0, all finite |

### Network configuration tested

- Population: 10 BoothRinzelNeurons (very slow — each has 4 sub-steps)
- PoissonInput: rate=500Hz, weight high enough for spiking
- Projection: self-recurrent, accepted by Network
- SpikeMonitor: records spikes
- Duration: long enough for spikes (model is very slow)

### Slowest model in the library

364.69s for 15 tests — primarily due to the network test which simulates
many timesteps of a 10-neuron BoothRinzel population. Each timestep
requires 4 sub-steps × 10 neurons × (8 exp + 6 currents + 6 state updates)
= 800+ floating-point operations per network step.

**ALL 15 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**

---

## Numerical Considerations

- **4 sub-steps:** dt=0.025ms, 4 sub-steps → 0.1ms biological per call.
  Required due to stiff soma dynamics (fast Na⁺ with g_Na=120).
- **8 exp() per sub-step:** 32 exp() per call. Plus _safe_exp clips
  arguments to [−500, 500].
- **Gate clipping:** h, n, q clipped to [0, 1] every sub-step. Prevents
  Euler overshoot in the coupled 2-compartment system.
- **Voltage clipping:** V_s, V_d clipped to [−200, 100]. Prevents runaway
  from the large Na⁺ conductance (g_Na=120).
- **Ca clipping:** Ca ≥ 0 (concentration non-negative).
- **tau_h, tau_n denominators:** +1e-12 prevents division by zero.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/booth_rinzel.py` — 110 lines.
- **Six state variables:** vs, vd, h, n, q, ca.
- **_safe_exp():** Static method with np.clip(x, −500, 500) before exp().
- **4 sub-steps in step():** Inner loop with all gates, currents, voltages.
- **Dataclass:** Uses `@dataclass` with 23 parameters.
- **Rust wiring:** Compatible (6 f64 state vars, sub-stepping in native code
  would give significant speedup).

---

## Performance

| Metric | Python | Rust (estimated) |
|--------|--------|-----------------|
| Isolation | ~250 steps/s | ~2.5K steps/s |
| Network (10n) | very slow | — |

One of the slowest models — 4 sub-steps × 8 exp() = 32 exp() per call.
Rust acceleration would eliminate Python overhead in the inner loop.

---

## Test Coverage Summary

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 9 | construction, binary, spikes, 2-compartment, Ca²⁺, bistability, finite, gating, reset |
| Network | 3 | Population, Network+spikes, Projection+spike_trains |
| Analysis | 3 | firing_rate, spike_count, isi |
| **Total** | **15** | **ALL PASSED (364.69s)** |

---

## Findings (Measured 2026-03-31)

1. **15/15 tests PASSED in 364.69s.** Slowest test file. No failures.

2. **Bistability verified:** Model demonstrates distinct up (spiking) and
   down (silent) states at the same input level.

3. **Two compartments differ:** V_s and V_d take different values during
   activity — soma spikes, dendrite plateaus.

4. **Ca²⁺ accumulates:** Ca > 0 after sustained drive, confirming
   dendritic Ca²⁺ channel function.

5. **Gating bounded:** h, n, q all in [0, 1] after clipping. No Euler
   overshoot leaks through.

6. **Network pipeline functional:** Population + PoissonInput + Projection
   all work. Spikes recorded by SpikeMonitor.

7. **Analysis toolkit works:** firing_rate, spike_count, isi all produce
   valid results from the model's spike train.

8. **Slowest model:** 32 exp() per call makes this the computationally
   most expensive model per step in the library. Rust acceleration
   recommended for network-scale simulations.

9. **Only bistable model:** Unique in SC-NeuroCore — no other model
   has persistent up/down states at the same input level.

10. **Spinal motoneuron application:** Directly models persistent inward
    currents (PICs) relevant to motor control and spinal cord injury.
