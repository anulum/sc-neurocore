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
numerical safety for the stiff 2-compartment system.

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
    currents (PICs) in spinal motoneurons.

---

## Theoretical Context

### Persistent inward currents (PICs) in motoneurons

Booth, Rinzel & Kiehn (1997) developed this model to explain the
role of persistent inward currents (PICs) in spinal motoneuron
bistability. PICs are sustained depolarising currents — primarily
L-type Ca²⁺ currents in dendrites — that can maintain a neuron
in a depolarised "up-state" without ongoing synaptic input.

PICs are critical for:
- **Self-sustained firing:** Motoneurons continue firing after a
  brief synaptic input, maintaining muscle contraction
- **Plateau potentials:** Dendritic Ca²⁺ PICs produce sustained
  depolarisations that outlast the synaptic drive
- **Gain amplification:** PICs amplify synaptic currents 2–5×

### Two-compartment architecture

The soma-dendrite separation is essential for PIC-mediated bistability:
- **Soma** ($V_s$): Generates Na⁺/K⁺ action potentials (fast spiking)
- **Dendrite** ($V_d$): Hosts Ca²⁺ PICs and K(Ca) channels (slow plateau)
- **Coupling** ($g_c$): Electrotonic coupling between compartments

The fraction $p = 0.5$ represents equal soma/dendrite membrane area.
The coupling conductance $g_c = 0.1$ mS/cm² is weak — allowing the
dendrite to maintain a plateau independently of somatic spiking.

### Amyotrophic lateral sclerosis (ALS)

Motoneuron disease (ALS) involves progressive degeneration of
spinal motoneurons. The Booth-Rinzel model predicts that:
- Reduced dendritic Ca²⁺ conductance ($g_{Ca}$) → loss of PICs →
  loss of self-sustained firing → muscle weakness
- Increased K(Ca) → premature plateau collapse → fatigue
- The model provides a framework for studying how PIC dysfunction
  contributes to ALS symptomatology

### Spasticity and PIC dysregulation

After spinal cord injury, motoneurons develop enhanced PICs due to
loss of descending serotonergic modulation. The Booth-Rinzel model
predicts that increased $g_{Ca}$ or reduced K(Ca) produces
persistent plateaus that are difficult to terminate — matching
the clinical presentation of spasticity (involuntary sustained
muscle contraction).

---

## Usage Examples

### Example 1: Bistable up/down state transition

```python
from sc_neurocore.neurons.models.booth_rinzel import BoothRinzelNeuron

neuron = BoothRinzelNeuron()

# Phase 1: silent (down-state)
for _ in range(10000):
    neuron.step(0.0)
v_down = neuron.vs

# Phase 2: trigger up-state with brief input
for _ in range(5000):
    neuron.step(10.0)

# Phase 3: remove input — plateau maintained?
spikes = sum(neuron.step(0.0) for _ in range(20000))
print(f"Down-state V: {v_down:.1f} mV")
print(f"Spikes after input removed: {spikes}")
```

### Example 2: Dendritic Ca²⁺ plateau

```python
from sc_neurocore.neurons.models.booth_rinzel import BoothRinzelNeuron

neuron = BoothRinzelNeuron()
for _ in range(50000):
    neuron.step(5.0)

print(f"Soma V: {neuron.vs:.1f} mV")
print(f"Dendrite V: {neuron.vd:.1f} mV")
print(f"Ca²⁺: {neuron.ca:.6f}")
print(f"V_s ≠ V_d: {abs(neuron.vs - neuron.vd) > 1.0}")
```

### Example 3: Motoneuron pool with recurrent excitation

```python
from sc_neurocore.network import Network, Population, Projection
from sc_neurocore.neurons.models.booth_rinzel import BoothRinzelNeuron
from sc_neurocore.input import PoissonInput
from sc_neurocore.monitors import SpikeMonitor
from sc_neurocore.analysis import spike_count

pool = Population(BoothRinzelNeuron, n=5)
drive = PoissonInput(rate=200.0, weight=5.0, dt=0.001, seed=42)

net = Network()
net.add_population("motoneurons", pool)
net.add_input("descending", drive, target="motoneurons")

mon = SpikeMonitor()
net.add_monitor("spikes", mon, source="motoneurons")

net.run(duration=2.0)
print(f"Total spikes: {spike_count(mon)}")
```

---

## Technical Reference

### Rust parity

| Aspect | Python | Rust | Status |
|--------|--------|------|--------|
| State variables | vs, vd, h, n, q, ca | same | **EXACT** |
| g_k | 20.0 | 20.0 | **EXACT** (fixed from 100.0) |
| m_inf | −35/7.8 | same | **EXACT** (fixed from −30/9.0) |
| h_inf | −55/7.0 | same | **EXACT** (fixed from −45/7.0) |
| n_inf | −28/15 | same | **EXACT** (fixed from −30/10) |
| s_inf (m_ca) | −22/5.0 | same | **EXACT** (fixed from −20/9.0) |
| q_inf | Boltzmann (Vd+35)/2 | same | **EXACT** (fixed from ca/(ca+2)) |
| K(Ca) | chi=min(ca/250,1) | same | **EXACT** (fixed from q) |
| tau_h | voltage-dependent | same | **EXACT** (fixed from constant 1.0) |
| tau_n | voltage-dependent | same | **EXACT** (fixed from constant 3.0) |
| tau_q | 400 ms | same | **EXACT** (fixed from 100.0) |
| alpha_ca × f_ca | 0.009 × 0.0025 | same | **EXACT** (fixed from 0.13) |
| k_ca | 0.18 | same | **EXACT** (fixed from 0.075) |
| Clipping | v,gates bounded | same | **EXACT** (added) |

**Parity verified:** commit b7134296 corrected 12 Rust defects.

### Source files

| File | Lines | Description |
|------|-------|-------------|
| `src/sc_neurocore/neurons/models/booth_rinzel.py` | 110 | Python reference |
| `engine/src/neurons/multi_compartment/booth_rinzel.rs` | (dedicated) | Rust implementation |
| `tests/test_model_booth_rinzel.py` | ~150 | 15 tests |

---

## Performance Benchmarks

### Criterion benchmarks (local i5-11600K, measured 2026-04-05)

| Metric | Value |
|--------|-------|
| Test | `booth_rinzel_1k_steps` |
| Median | 256 µs |
| Per-step | 0.256 µs (256 ns) |
| Throughput | ~3.9 Mstep/s |

Two-compartment model with 4 sub-steps × 8 exp() = 32 exp() per
call. Higher cost than single-compartment models but still 15×
faster than Python (~250 steps/s).

---

## Citations

1. Booth V, Rinzel J, Kiehn O (1997). Compartmental model of
   vertebrate motoneurons for Ca²⁺-dependent spiking and plateau
   potentials under pharmacological treatment. *J Neurophysiol*
   78(6):3371–3385.
   DOI: [10.1152/jn.1997.78.6.3371](https://doi.org/10.1152/jn.1997.78.6.3371)

2. Heckman CJ, Enoka RM (2012). Motor unit. *Compr Physiol*
   2(4):2629–2682.
   DOI: [10.1002/cphy.c100087](https://doi.org/10.1002/cphy.c100087)

3. Hounsgaard J, Kiehn O (1989). Serotonin-induced bistability of
   turtle motoneurones caused by a nifedipine-sensitive calcium
   plateau potential. *J Physiol* 414:265–282.
   DOI: [10.1113/jphysiol.1989.sp017687](https://doi.org/10.1113/jphysiol.1989.sp017687)

4. Lee RH, Heckman CJ (1998). Bistability in spinal motoneurons
   in vivo: systematic variations in persistent inward currents.
   *J Neurophysiol* 80(2):583–593.
   DOI: [10.1152/jn.1998.80.2.583](https://doi.org/10.1152/jn.1998.80.2.583)

5. Rinzel J, Ermentrout GB (1998). Analysis of neural excitability
   and oscillations. In: Koch C, Segev I (eds). *Methods in Neuronal
   Modeling*. 2nd ed. MIT Press, pp. 251–291.

6. Powers RK, Binder MD (2001). Input-output functions of mammalian
   motoneurons. *Rev Physiol Biochem Pharmacol* 143:137–263.
   DOI: [10.1007/BFb0115594](https://doi.org/10.1007/BFb0115594)

---

**ALL 15 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**
**Rust parity: EXACT (verified commit b7134296, 12 defects fixed).**
**Criterion: 256 µs / 1K steps (256 ns/step, ~3.9 Mstep/s).**
    currents (PICs) relevant to motor control and spinal cord injury.

## Industrial hardening note

The Booth-Rinzel surface now uses the same fail-closed candidate-update contract across the Python reference implementation and the Go, Rust, and Julia companion kernels. Physical configuration is rejected before integration when conductances, capacitance, coupling fraction, calcium scales, timestep, or gate/calcium state leave their mathematical domains. Candidate substeps are validated before mutation, so invalid runtime inputs or non-finite intermediate states cannot partially corrupt the neuron state.

The historical performance figures in this page predate the candidate-validation hardening and should be treated as archival until the Booth-Rinzel benchmark is regenerated on the current code.
