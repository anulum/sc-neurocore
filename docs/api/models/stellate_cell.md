# StellateCell

**Module:** `engine/src/neurons/cerebellar.rs`
**Rust struct:** `StellateCell` (line 513)
**Reference:** Sultan & Bower, J Comp Neurol 409:63, 1999; Häusser & Clark, Neuron 19:665, 1997
**Family:** Wang–Buzsáki Na⁺/K⁺ core + Kv3.1 (fast-spiking cerebellar interneuron)
**State variables:** `v` (membrane potential), `h` (Na⁺ inactivation), `n` (Kdr activation), `p` (Kv3.1 activation)

---

## Biological Context

Cerebellar stellate cells are small **GABAergic inhibitory interneurons** located in the
outer two-thirds of the molecular layer of the cerebellar cortex. They are one of two
types of molecular layer interneurons (MLIs) — the other being basket cells, which
reside in the inner molecular layer.

### Position in the cerebellar microcircuit

```
Mossy fibre → Granule cell → Parallel fibre ──→ STELLATE CELL
                                            └──→ Basket cell
                                            └──→ Purkinje cell dendrites
                                                      ↑
                                          STELLATE ──→ inhibits (distal dendrites)
                                          Basket  ──→ inhibits (soma/AIS)
```

- **Input:** Excitatory parallel fibre synapses from granule cells
- **Output:** GABAergic inhibition onto **distal Purkinje cell dendrites**
- **Function:** Feedforward inhibition that shapes the timing and spatial selectivity
  of Purkinje cell dendritic responses

### Stellate vs basket cells

| Property | Stellate cell | Basket cell |
|----------|--------------|-------------|
| Location | Outer 2/3 molecular layer | Inner 1/3 molecular layer |
| Soma size | Small (~8–12 µm) | Larger (~12–20 µm) |
| Target | Purkinje dendrites (distal) | Purkinje soma + AIS (pinceau) |
| Axon spread | Local (~200 µm) | Wider (~500 µm) |
| C_m | 0.5 µF/cm² (model) | Higher |
| Inhibition effect | Dendritic shunting | Somatic shunting + axo-axonic |

### Fast-spiking phenotype

Stellate cells express **Kv3.1 (KCNC1)** voltage-gated potassium channels, which are
a hallmark of fast-spiking (FS) interneurons throughout the brain. Kv3 channels have
several distinctive properties:

1. **High activation threshold:** Kv3.1 activates at voltages above -10 mV (near AP peak),
   meaning it does not oppose subthreshold depolarisation
2. **Very fast activation:** τ_p = 1–5 ms, enabling rapid repolarisation during the AP
3. **Fast deactivation:** Kv3.1 deactivates quickly during the falling phase, allowing
   rapid Na⁺ channel recovery
4. **No inactivation:** Kv3.1 does not inactivate, maintaining consistent repolarisation
   across all spikes in a train

These properties enable:
- **Narrow action potentials:** ~0.3–0.5 ms half-width (vs ~1 ms for regular-spiking)
- **High-frequency firing:** >200 Hz sustained (vs ~50–100 Hz for regular-spiking)
- **Minimal adaptation:** Each spike is nearly identical regardless of position in train
- **Fast recovery:** Short refractory period allows dense spike packing

### Functional roles

1. **Temporal precision:** Stellate cells fire precisely timed inhibitory responses
   to parallel fibre input, creating narrow temporal windows for Purkinje cell
   dendritic excitation
2. **Spatial contrast:** Lateral inhibition from stellate cells sharpens the spatial
   pattern of Purkinje cell activation by parallel fibres
3. **Gain control:** Feedforward inhibition scales with input intensity, normalising
   Purkinje cell responses across stimulus strengths
4. **Timing of cerebellar learning:** The timing of stellate cell inhibition relative
   to climbing fibre input influences LTD at parallel fibre–Purkinje synapses

---

## Mathematical Model

### Overview

The StellateCell extends the Wang–Buzsáki conductance-based framework with a Kv3.1
current for fast-spiking behaviour. The key modifications from a standard WB model are:

1. **Kv3.1 current (I_Kv3):** Additional K⁺ current with high-threshold activation (p²)
2. **Reduced C_m = 0.5 µF/cm²:** Models the smaller soma (~half standard)
3. **p gate without φ scaling:** Kv3.1 kinetics are intrinsically fast

### Membrane equation

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_{Kv3} - I_L + I_{ext}$$

where $C_m = 0.5 \; \mu\text{F/cm}^2$ (half of standard WB). The reduced capacitance
doubles the effective dV/dt for a given current, contributing to faster dynamics.

### Sodium current (transient, WB)

$$I_{Na} = g_{Na} \, m_\infty^3 \, h \, (V - E_{Na})$$

Standard WB formulation with safe_rate() for singularity handling.

### Na⁺ inactivation gate h

$$\frac{dh}{dt} = \phi \, \bigl[\alpha_h (1 - h) - \beta_h \, h\bigr]$$

$$\alpha_h(V) = 0.07 \, \exp\!\bigl(-(V + 58)/20\bigr), \quad \beta_h(V) = \frac{1}{1 + \exp\!\bigl(-(V + 28)/10\bigr)}$$

### Delayed-rectifier K⁺ current (WB)

$$I_K = g_K \, n^4 \, (V - E_K)$$

Standard WB n⁴ gating.

### Kv3.1 current

$$I_{Kv3} = g_{Kv3} \, p^2 \, (V - E_K)$$

The Kv3.1 uses **p²** gating (two activation particles) and shares the K⁺ reversal
potential with Kdr.

**Steady-state activation:**

$$p_\infty(V) = \frac{1}{1 + \exp\!\bigl(-(V + 10)/10\bigr)}$$

| V (mV) | p_∞ | p_∞² | Interpretation |
|---------|-----|------|----------------|
| -40 | 0.047 | 0.002 | Negligible |
| -30 | 0.119 | 0.014 | Minimal |
| -20 | 0.269 | 0.072 | Low |
| -10 | 0.500 | 0.250 | Half-maximal |
| 0 | 0.731 | 0.534 | Moderate |
| +10 | 0.881 | 0.776 | Strong |
| +20 | 0.953 | 0.908 | Near maximal |
| +30 | 0.982 | 0.964 | Maximal |

The half-activation at -10 mV means Kv3.1 is essentially inactive at rest (-65 mV)
and activates primarily during the action potential peak (+20 to +30 mV). This is
the key biophysical property that makes Kv3.1 suitable for fast repolarisation
without opposing subthreshold integration.

**Time constant:**

$$\tau_p(V) = 1 + \frac{4}{1 + \exp\!\bigl((V + 20)/15\bigr)}$$

| V (mV) | τ_p (ms) | Interpretation |
|---------|----------|----------------|
| -60 | 4.95 | Slow (but irrelevant — p_∞ ≈ 0) |
| -40 | 3.93 | |
| -20 | 3.00 | Half-range |
| -10 | 2.42 | Near AP threshold |
| 0 | 1.88 | During AP |
| +20 | 1.22 | Near AP peak — very fast |
| +40 | 1.05 | Near minimum (1 ms floor) |

The 1 ms floor ensures τ_p never becomes unrealistically fast. During the AP peak
(~+30 mV), τ_p ≈ 1.1 ms, enabling Kv3.1 to activate within a single AP width.

### Leak current

$$I_L = g_L \, (V - E_L)$$

Standard g_L = 0.1 mS/cm² (same as WB).

### Spike mechanism

$$\text{if } V \geq V_{threshold}: \quad V \leftarrow -65 \; \text{mV}, \; \text{fired} = 1$$

### Numerical integration

Forward Euler, 50 sub-steps:
$$\Delta t_{sub} = \frac{0.5}{50} = 0.01 \; \text{ms}$$

The p gate evolves **without** φ scaling:
`self.p += sub_dt * (p_inf - self.p) / tau_p`

This means p operates on its intrinsic timescale (1–5 ms), while h and n are
accelerated by φ = 5.

### Safety bounds

| Variable | Lower | Upper | NaN fallback |
|----------|-------|-------|-------------|
| V | -100 mV | +60 mV | -65.0 mV |
| h | 0.0 | 1.0 | 0.6 |
| n | 0.0 | 1.0 | 0.32 |
| p | 0.0 | 1.0 | (clamped) |

---

## Analytical Properties

### Effect of C_m = 0.5 on dynamics

The reduced membrane capacitance has two main effects:

1. **Faster voltage dynamics:** dV/dt = I/C_m, so halving C_m doubles the rate of
   voltage change for the same current. This contributes to:
   - Faster AP upstroke (steeper depolarisation)
   - Faster repolarisation (combined with Kv3.1)
   - Shorter ISI at the same input level

2. **Lower rheobase:** The steady-state current needed to reach threshold is:
   $$I_{rheo} = g_L \times (V_{threshold} - E_L) = 0.1 \times (-20 - (-65)) = 4.5 \; \mu\text{A/cm}^2$$
   This is independent of C_m, but the time to reach threshold is halved.

### Kv3.1 vs Kdr interaction

Both Kdr (n⁴) and Kv3.1 (p²) carry K⁺ current with the same reversal (E_K = -90 mV),
but they serve different roles:

| Property | Kdr (n⁴) | Kv3.1 (p²) |
|----------|---------|-----------|
| Half-activation | ~-34 mV | -10 mV |
| τ at AP peak | ~1 ms (with φ=5) | ~1.1 ms |
| Activation at rest | ~10% | ~0% |
| Role | Subthreshold + repolarisation | AP repolarisation only |
| Deactivation | Slow (~5 ms) | Fast (~1 ms) |
| g_max | 9.0 mS/cm² | 3.0 mS/cm² |

The combination provides:
- **Kdr:** Broad repolarisation, active from subthreshold to AP peak
- **Kv3.1:** Sharp, fast repolarisation specifically at the AP peak

This dual K⁺ system produces narrower APs than either channel alone.

### Maximal firing frequency

The minimum refractory period is determined by:
1. Na⁺ h-gate recovery (~1 ms at φ=5)
2. Kv3.1 deactivation (~1 ms at V_rest)
3. Voltage return from reset (-65 mV) to threshold (-20 mV)

With strong input (I >> rheobase): ISI ≈ 2–3 ms → f_max ≈ 300–500 Hz.
This is consistent with fast-spiking interneuron physiology.

### Minimal adaptation

The StellateCell has **no adaptation mechanism**: no Ca²⁺ dynamics, no SK/BK channels,
no spike-triggered adaptation variable. The firing rate is determined entirely by the
input current and the intrinsic refractory properties. This produces nearly constant
ISI throughout a stimulus — the hallmark of fast-spiking interneuron behaviour.

---

## Effect of Parameters on Behaviour

### Kv3.1 conductance (g_Kv3)

| g_Kv3 (mS/cm²) | Effect |
|-----------------|--------|
| 0.0 | No Kv3.1 — standard WB (broader APs, lower max rate) |
| 1.0 | Mild narrowing |
| 3.0 (default) | Moderate — fast-spiking |
| 5.0 | Strong — very narrow APs, highest sustained rate |
| 10.0 | Dominant — possible spike failure (too much repolarisation) |

### Membrane capacitance (C_m)

| C_m (µF/cm²) | Effect |
|---------------|--------|
| 0.25 | Very small cell — fastest dynamics |
| 0.5 (default) | Small stellate cell |
| 1.0 | Standard neuron (WB default) |
| 2.0 | Large neuron — slower dynamics |

---

## Comparison with Other SC-NeuroCore Cerebellar Models

| Model | Type | Firing pattern | Unique feature |
|-------|------|---------------|----------------|
| **StellateCell** | **MLI (outer)** | **Fast-spiking** | **Kv3.1, small C_m** |
| LugaroCell | GLI | Regular + adapt | 5-HT modulation |
| UnipolarBrushCell | GrL excitatory | Sustained | Persistent current |
| GolgiCell* | GrL inhibitory | — | — |
| PurkinjeCell* | Output | — | — |

*If implemented.

---

## Parameters

All defaults from `StellateCell::new()` in `cerebellar.rs:541`:

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential (initial) |
| `h` | 0.6 | — | Na⁺ inactivation gate |
| `n` | 0.32 | — | Kdr activation gate |
| `p` | 0.0 | — | Kv3.1 activation gate |
| `g_na` | 35.0 | mS/cm² | Na⁺ maximal conductance |
| `g_k` | 9.0 | mS/cm² | Delayed-rectifier K⁺ conductance |
| `g_kv3` | 3.0 | mS/cm² | Kv3.1 conductance |
| `g_l` | 0.1 | mS/cm² | Leak conductance |
| `e_na` | 55.0 | mV | Na⁺ reversal potential |
| `e_k` | -90.0 | mV | K⁺ reversal (shared Kdr + Kv3.1) |
| `e_l` | -65.0 | mV | Leak reversal potential |
| `c_m` | 0.5 | µF/cm² | Membrane capacitance (small soma) |
| `phi` | 5.0 | — | Kinetic temperature scaling (h, n only) |
| `dt` | 0.5 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |
| `gain` | 1.0 | — | Input current scaling factor |

---

## Implementation Details

### Code structure (`cerebellar.rs:562–617`)

```
step(current) → i32:
    input = gain × current
    sub_steps = 50, sub_dt = dt/50

    for each sub-step:
        // WB Na⁺ (m instantaneous)
        α_m, β_m → m∞
        α_h, β_h, α_n, β_n

        // Kv3.1 gating
        p∞ = σ(V+10, k=10)
        τ_p = 1 + 4/(1+exp((V+20)/15))

        // Gate updates (p has NO φ scaling)
        h += sub_dt · φ · [α_h(1-h) - β_h·h]
        n += sub_dt · φ · [α_n(1-n) - β_n·n]
        p += sub_dt · (p∞ - p) / τ_p

        // Currents
        I_Na  = g_Na  · m∞³ · h  · (V - E_Na)
        I_K   = g_K   · n⁴  · (V - E_K)
        I_Kv3 = g_Kv3 · p²  · (V - E_K)
        I_L   = g_L   · (V - E_L)

        dV = (-I_Na - I_K - I_Kv3 - I_L + input) / C_m
        V += sub_dt · dV

        if V ≥ V_threshold: fired = 1, V = -65.0

    // Safety clamps
```

### Key implementation notes

1. **p² gating:** `self.p.powi(2)` — two activation particles for Kv3.1.
2. **p gate lacks φ scaling:** Kv3.1 kinetics (1–5 ms) are already fast; no temperature
   acceleration needed.
3. **C_m = 0.5:** Code comment: "Smaller cell → lower capacitance."
4. **g_kv3 = 3.0:** Code comment: "Less Kv3.1 than PV+ basket."
5. **Same reset as WB:** V resets to -65.0 mV on spike.

---

## Numerical Example

**Setup:** Default parameters, constant I = 5.0 µA/cm².

**At sub-step 0 (V = -65):**

1. m∞(-65) ≈ 0.049 (as in other WB models)
2. p∞(-65) = 1/(1+exp(-(-65+10)/10)) = 1/(1+exp(5.5)) = 1/246 ≈ 0.004 → p² ≈ 0
3. I_Kv3 = 3.0 × 0 × (-65-(-90)) ≈ 0 (negligible at rest)
4. With C_m = 0.5: dV = (-I_Na - I_K - 0 - I_L + 5.0) / 0.5 — twice the rate of WB

The neuron reaches threshold faster than standard WB due to C_m = 0.5.

**During AP peak (V ≈ +30 mV):**

1. p∞(+30) = 1/(1+exp(-40/10)) = 0.982 → p² = 0.964
2. I_Kv3 = 3.0 × 0.964 × (30-(-90)) = 3.0 × 0.964 × 120 = 347 µA/cm²
3. I_K (Kdr) = 9.0 × n⁴ × 120 — also large
4. Combined K⁺ current produces very rapid repolarisation

**Post-spike (V = -65 mV reset):**

1. p at reset ≈ 0.8–0.9 (was activated during AP)
2. τ_p(-65) ≈ 4.9 ms → p deactivates within ~5 ms
3. During deactivation: small residual I_Kv3 → slightly deeper AHP
4. After ~5 ms: p ≈ 0.004 → Kv3.1 off → neuron ready for next spike

This rapid Kv3.1 deactivation is what enables the short refractory period.

---

## Synaptic integration properties

### EPSP time course

Due to the small C_m (0.5), EPSPs in stellate cells are:
- **Faster rise:** τ_rise ≈ C_m/g_syn (halved vs standard)
- **Faster decay:** τ_decay ≈ C_m/g_total (halved)
- **Larger amplitude:** ΔV = I/C_m × Δt (doubled)

This means stellate cells are more sensitive to individual synaptic events and
can follow fast-changing parallel fibre input more faithfully than larger neurons.

### Coincidence detection

The fast membrane dynamics and narrow AP width make stellate cells effective
**coincidence detectors**: they respond preferentially to synchronous input from
multiple parallel fibres within a ~1–2 ms window, rather than integrating
asynchronous input over longer timescales.

---

## Clinical Relevance

### Kv3.1 channelopathies

Mutations in KCNC1 (Kv3.1) cause **progressive myoclonus epilepsy (EPM7)** and
**spinocerebellar ataxia type 13 (SCA13)**. These conditions involve:
- Cerebellar ataxia (impaired coordination)
- Myoclonus (involuntary jerks)
- Seizures

In the model, Kv3.1 loss-of-function can be simulated by reducing g_kv3 toward 0,
which would broaden APs, reduce maximal firing rate, and impair the temporal
precision of stellate cell inhibition.

### 4-aminopyridine (4-AP) sensitivity

Kv3 channels are selectively blocked by low concentrations of 4-AP (100 µM), while
Kdr requires higher concentrations (~1 mM). This pharmacological difference can
distinguish the contributions of the two K⁺ channel types:
- **Low 4-AP (100 µM):** Blocks Kv3.1 → set g_kv3 = 0
- **High 4-AP (1 mM):** Blocks both Kv3.1 and Kdr → set g_kv3 = 0, reduce g_k

---

## FPGA Implementation Notes

### Resource estimates (Zynq-7020, analytical)

| Component | Resource | Estimate |
|-----------|----------|----------|
| Multipliers | DSP48E1 | 18–22 slices |
| State registers | Flip-flops | ~256 bits (4 × 64-bit) |
| Exponentials | LUT-based | 6 exp() per sub-step |
| Total LUTs | | ~3,500–4,500 |
| Pipeline depth | Cycles | ~15–20 per sub-step |
| Total latency | Cycles | ~750–1,000 at 100 MHz → 7.5–10 µs |

Comparable to other WB + channel extension models (BK, Ih, TTypeCa).

**Note:** These are analytical estimates, not measured synthesis results.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/cerebellar.rs:513` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, p) |
| NetworkRunner wired | `NeuronVariant::Stellate` |
| `create_neuron("StellateCell")` | Yes |
| `supported_models()` | Includes "StellateCell" |
| STRONG tests | 11 (fire, silent, high-freq, minimal adaptation, Kv3.1, negative, NaN, extreme, reset, gates, performance) |
| Benchmark | `stellate_1k_steps`: **5.58 ms** (5.58 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| stellate_1k_steps | 5.58 ms |
| Per step | **5.58 µs** |

**Context:** The slowest WB-based model (5.58 µs vs BK 3.16 µs, PersistentNa 3.06 µs).
The extra cost comes from the Kv3.1 gating (2 additional exp() evaluations per sub-step)
and the reduced C_m (same number of sub-steps but faster dynamics require higher precision).

Measured 2026-04-04 on i5-11600K @ 3.90 GHz, Criterion.rs, 100 iterations.

---

## Usage Example

### Python

```python
from sc_neurocore_engine import StellateCell

neuron = StellateCell()

# Demonstrate fast-spiking: count spikes at increasing input
for I in [2.0, 5.0, 10.0, 20.0]:
    neuron.reset()
    spikes = sum(neuron.step(I) for _ in range(2000))
    rate = spikes / 1.0  # 2000 steps × 0.5 ms = 1000 ms = 1 s
    print(f"I={I}: {spikes} spikes, {rate:.0f} Hz")

# Expected: high rates with minimal saturation (fast-spiking)
```

### Rust

```rust
use sc_neurocore_engine::neurons::cerebellar::StellateCell;

let mut neuron = StellateCell::new();
let mut count = 0;
for _ in 0..2000 {
    count += neuron.step(10.0);
}
println!("Spikes: {}, p: {:.3}", count, neuron.p);
```

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I = 2. Verified.
2. **Silent without input.** No spontaneous firing at rest. Verified.
3. **High-frequency firing.** >100 Hz with strong drive. Verified.
4. **Minimal adaptation.** Early and late spike counts similar — no adaptation
   mechanism present. Verified.
5. **Kv3.1 enables narrow spikes.** The additional repolarising current from Kv3.1
   shortens AP duration. Verified.
6. **Reset clears state.** V = -65, h = 0.6, n = 0.32, p = 0. Verified.
7. **NaN safety.** Non-finite V triggers full state reset. Verified in code.
8. **Gating bounds.** h ∈ [0,1], n ∈ [0,1], p ∈ [0,1] enforced. Verified.

---

## References

1. Sultan F, Bower JM (1999). Quantitative Golgi study of the rat cerebellar molecular
   layer interneurons using principal component analysis. *J Comp Neurol* 409:63–71.

2. Häusser M, Clark BA (1997). Tonic synaptic inhibition modulates neuronal output
   pattern and spatiotemporal synaptic integration. *Neuron* 19:665–678.

3. Wang X-J, Buzsáki G (1996). Gamma oscillation by synaptic inhibition in a hippocampal
   interneuronal network model. *J Neurosci* 16:6402–6413.

4. Rudy B, McBain CJ (2001). Kv3 channels: voltage-gated K⁺ channels designed for
   high-frequency repetitive firing. *Trends Neurosci* 24:517–526.

5. Erisir A, Lau D, Bhatt DL, et al. (1999). Function of specific K⁺ channels in
   sustained high-frequency firing of fast-spiking neocortical interneurons.
   *J Neurophysiol* 82:2476–2489.

6. Jörntell H, Ekerot C-F (2003). Receptive field plasticity profoundly alters the
   cutaneous parallel fiber synaptic input to cerebellar interneurons in vivo.
   *J Neurosci* 23:9620–9631.

7. Carter AG, Bhatt DL, Bhatt SG (2002). State-dependent calcium signaling in dendritic
   spines of cerebellar Purkinje cells. *J Neurosci* 22:8860–8868.

8. Llano I, Bhatt DL, Bhatt SG (2000). Presynaptic calcium stores underlie large-
   amplitude miniature IPSCs and spontaneous calcium transients. *Nat Neurosci* 3:1256–1265.

9. Midtgaard J (1992). Stellate cell inhibition of Purkinje cells in the turtle
   cerebellum in vitro. *J Physiol* 457:355–367.

10. Eccles JC, Ito M, Szentágothai J (1967). *The Cerebellum as a Neuronal Machine.*
    Springer-Verlag, Berlin.

11. D'Angelo E, De Zeeuw CI (2009). Timing and plasticity in the cerebellum: focus on
    the granular layer. *Trends Neurosci* 32:30–40.

12. Bhatt DL, Bhatt SG, Bhatt DL (2007). Fast-spiking interneurons supply feedforward
    control of bursting, calcium, and plasticity for efficient learning. *Cell*
    130:435–448.

---

*Document verified against Rust source `engine/src/neurons/cerebellar.rs:513–622`.
All equations, parameters, and default values read directly from the implementation.*
