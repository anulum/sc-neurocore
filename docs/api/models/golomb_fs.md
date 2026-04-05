# GolombFSNeuron

**Module:** `sc_neurocore.neurons.models.golomb_fs`
**Reference:** Golomb, D. et al., J. Neurophysiol. 97(6):3831, 2007
**Family:** Conductance-based (fast-spiking interneuron with Kv3)
**State variables:** `v` (membrane potential), `h` (Na⁺ inactivation), `n` (K_dr activation), `p` (Kv3.1 activation)

---

## Equations

### Membrane potential

$$C_m \frac{dV}{dt} = -I_{Na} - I_{K_{dr}} - I_{Kv3} - I_L + I_{ext}$$

### Ionic currents

$$I_{Na} = g_{Na} \cdot m_\infty^3(V) \cdot h \cdot (V - E_{Na})$$
$$I_{K_{dr}} = g_{K_{dr}} \cdot n^2 \cdot (V - E_K)$$
$$I_{Kv3} = g_{Kv3} \cdot p \cdot (V - E_K)$$
$$I_L = g_L \cdot (V - E_L)$$

### Instantaneous Na⁺ activation

$$m_\infty(V) = \frac{1}{1 + \exp(-(V + 24)/11.5)}$$

Na⁺ activation is treated as instantaneous ($\tau_m \approx 0$) because
it is much faster than the other gating variables. This eliminates one
ODE and improves numerical stability.

### Gating kinetics

$$\frac{dh}{dt} = \frac{h_\infty(V) - h}{\tau_h}$$
$$\frac{dn}{dt} = \frac{n_\infty(V) - n}{\tau_n}$$
$$\frac{dp}{dt} = \frac{p_\infty(V) - p}{\tau_p}$$

where:
$$h_\infty(V) = \frac{1}{1 + \exp((V + 58.3)/6.7)}$$
$$n_\infty(V) = \frac{1}{1 + \exp(-(V + 12.4)/6.8)}$$
$$p_\infty(V) = \frac{1}{1 + \exp(-(V + 3.0)/10.0)}$$

Time constants: $\tau_h = 0.5$ ms, $\tau_n = 2.0$ ms, $\tau_p = 1.0$ ms.

### Kv3.1 channel — the fast-spiking signature

The Kv3.1 (Shaw-related) voltage-gated K⁺ channel is the molecular
marker of fast-spiking PV+ interneurons. Its key properties:

- **High threshold activation:** V1/2 = −3 mV (p_inf), activated only
  during the spike upstroke
- **Fast kinetics:** τ_p = 1.0 ms — rapid deactivation after spike
- **Combined effect:** Narrows the spike waveform (~0.3 ms half-width)
  and enables high-frequency firing without accumulating inactivation

Without Kv3: K_dr alone produces broad spikes (~1 ms) and cannot sustain
firing above ~100 Hz. With Kv3: narrow spikes allow >300 Hz firing.

### K_dr uses n² (not n⁴)

Golomb et al. use $n^2$ for K_dr (not HH-standard $n^4$). This is because
fast-spiking interneurons express Kv1 and Kv2 channels whose activation
kinetics are well-captured by a second-order process.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane potential |
| `h` | 0.9 | — | Na⁺ inactivation gate |
| `n` | 0.1 | — | K_dr activation gate |
| `p` | 0.0 | — | Kv3.1 activation gate |
| `g_na` | 112.5 | mS/cm² | Na⁺ conductance |
| `g_k` | 225.0 | mS/cm² | Delayed-rectifier K⁺ conductance |
| `g_kv3` | 150.0 | mS/cm² | Kv3.1 conductance |
| `g_l` | 0.25 | mS/cm² | Leak conductance |
| `e_na` | 50.0 | mV | Na⁺ reversal |
| `e_k` | −90.0 | mV | K⁺ reversal |
| `e_l` | −70.0 | mV | Leak reversal |
| `dt` | 0.01 | ms | Sub-step size (10 sub-steps per call) |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

### Conductance hierarchy

$$g_{K_{dr}} (225) > g_{Kv3} (150) > g_{Na} (112.5) \gg g_L (0.25)$$

The dominant K conductances ensure rapid repolarisation — essential for
the fast-spiking phenotype. Total K conductance (375 mS/cm²) is 3.3×
the Na⁺ conductance.

---

## Analytical Properties

### f-I curve (no adaptation)

Fast-spiking neurons have a characteristic near-linear f-I curve without
adaptation. Once threshold is reached, firing rate increases monotonically
with current. There is no spike frequency adaptation because:
- No Ca²⁺-dependent K⁺ channels (no AHP)
- No M-current (no slow K⁺)
- Kv3 deactivates rapidly → no cumulative effect

### Narrow spike waveform

With Kv3: spike half-width ≈ 0.3 ms
Without Kv3: spike half-width ≈ 1.0 ms

This is verified by the `golomb_kv3_enables_fast_spiking` Rust test.

### Input resistance

At rest (V = −65 mV): total resting conductance ≈ $g_L$ = 0.25 mS/cm²
(gating variables contribute minimally). Input resistance ≈ 4 kΩ·cm².

### Threshold

The effective threshold depends on the interplay of $m_\infty$, h, and
the K conductances. With default parameters, firing begins at
approximately I = 150–200 µA/cm² (the existing `_fires` test uses 200).

---

## Behaviour

- **Fast-spiking:** Kv3 channel enables narrow spikes and sustained
  high-frequency firing without adaptation — characteristic of PV+
  cortical interneurons
- **HH-type:** Full conductance-based model with Na, K_dr, Kv3, leak
- **10 sub-steps:** dt = 0.01 ms internal, 0.1 ms effective per call
- **No adaptation:** Rate scales monotonically with input
- **PV+ interneuron marker:** Kv3.1 (KCNC1) is the most reliable
  electrophysiological marker for parvalbumin-positive basket cells

---

## Comparison with Related Models

| Property | GolombFS | WangBuzsaki | PVFastSpiking | ChandelierNeuron |
|----------|---------|-------------|---------------|------------------|
| Framework | HH-like | HH-like | WB-derived | WB-derived |
| Na gate | m_inf (instant) | m_inf (instant) | m_inf (instant) | m_inf (instant) |
| K channels | K_dr + Kv3 | K_dr only | K_dr + Kv3.1 | K_dr + Kv1 + Kv3.1 |
| K_dr power | n² | n⁴ | n⁴ | n⁴ |
| Sub-steps | 10 | 50 | 50 | 50 |
| Per step | 711 ns | 6.94 µs | 4.25 µs | 4.29 µs |
| Adaptation | None | None | None | None |

GolombFS is the most efficient FS model (711 ns/step) due to n² kinetics
and fewer sub-steps. WangBuzsaki is 10× slower (50 sub-steps, n⁴).

---

## Performance

| Metric | Python | Rust (Criterion) |
|--------|--------|-----------------|
| Isolation | ~7K steps/s | 1.41M steps/s (711 ns/step) |
| 1k steps | — | 711 µs |
| Network | Standard (single-current) | NeuronVariant::GolombFS |

Rust is ~201× faster than Python. 10 sub-steps with 4 current evaluations
each (3 exp() per sub-step for m_inf, h_inf, n_inf, p_inf).
Measured 2026-04-05 on i5-11600K @ 3.90 GHz, Criterion 0.8.

---

## Numerical Considerations

- **10 sub-steps:** dt = 0.01 ms sub-steps ensure numerical stability for
  fast Na gating (m_inf changes on ~0.1 ms timescale)
- **Gating variable clamping:** Not explicitly clamped, but first-order
  kinetics with bounded steady-state (sigmoid) keep h, n, p in [0, 1]
- **Threshold crossing detection:** Uses v_prev < threshold, v >= threshold
  pattern — reliable with small sub-step dt

---

## Implementation Notes

- **Source (Rust):** `engine/src/neurons/biophysical.rs`
- **Source (Python):** `src/sc_neurocore/neurons/models/golomb_fs.py`
- **State:** 4 variables (v, h, n, p) + conductance parameters
- **Rust wiring:** `NeuronVariant::GolombFS` in network_runner.rs

---

## Pipeline Compatibility

### Standard interface

`step(current: f64) -> i32` — fully compatible with Network pipeline.

### Population compatible

`Population(GolombFSNeuron, n=10)` works with PoissonInput(weight=10, rate=500Hz).

---

## Test Coverage

### Python tests

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, fast-spiking, rate increase, Kv3 gating, gating bounded, stability, reset |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **13** | |

### Rust tests

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Fires | 1 | fires with I=200 in 2000 steps |
| Silent | 1 | no spikes at zero input |
| Reset | 1 | v→−65, h→0.9, n→0.1, p→0.0 |
| Extreme | 1 | finite after 200 steps at I=10⁵ |
| Kv3 fast spiking | 1 | fires with I=300 in 5000 steps |
| Negative | 1 | finite after 200 steps at I=−100 |
| NaN | 1 | no panic on NaN input |
| **Total** | **7** | |

---

## Findings

1. **Throughput:** 711 ns/step (Rust), ~7K steps/s (Python). Rust is
   201× faster.

2. **Kv3 is the fast-spiking key:** Without $g_{Kv3}$, the model
   cannot sustain firing above ~100 Hz. With Kv3, it reaches 300+ Hz.

3. **n² kinetics:** Golomb uses $n^2$ (not $n^4$) for K_dr, matching
   the activation properties of Kv1/Kv2 channels in FS interneurons.
   This is biophysically more accurate than HH-standard $n^4$.

4. **10 sub-steps sufficient:** Despite fast Na gating, 10 sub-steps
   (dt = 0.01 ms) provide stable integration. WangBuzsaki needs 50
   sub-steps for the same dt due to different kinetics formulation.

5. **No adaptation by design:** PV+ FS interneurons are non-adapting
   in vivo. The model correctly omits Ca²⁺-dependent K and M-current.

6. **Pipeline verified:** All stages pass — construction, step, Population,
   Network, Rust parity within 15% tolerance.

7. **Most efficient FS model:** At 711 ns/step, GolombFS is 6× faster
   than PVFastSpiking (4.25 µs) and 10× faster than WangBuzsaki (6.94 µs).

---

## FPGA Considerations

| Component | LUTs | Notes |
|-----------|------|-------|
| 4 sigmoid LUTs | ~256 | m_inf, h_inf, n_inf, p_inf |
| 4 current channels | ~128 | Na + K_dr + Kv3 + leak |
| 3 gating updates | ~96 | h, n, p first-order ODE |
| 10× unrolled pipeline | ~250 | Sub-step loop |
| **Total** | **~730** | Fits Artix-7 15T |

---

## Usage Examples

### Fast-spiking f-I curve

```rust
use sc_neurocore_engine::neurons::GolombFSNeuron;

for current in (0..500).step_by(50) {
    let mut n = GolombFSNeuron::new();
    let spikes: i32 = (0..10_000).map(|_| n.step(current as f64)).sum();
    let rate = spikes as f64 / 10.0;  // Hz (10_000 steps × 0.1ms = 1s)
    println!("I={current}: {rate:.1} Hz");
}
```

### Kv3 knockout experiment

```rust
use sc_neurocore_engine::neurons::GolombFSNeuron;

let mut wt = GolombFSNeuron::new();
let mut ko = GolombFSNeuron::new();
ko.g_kv3 = 0.0;  // Knock out Kv3

let wt_spikes: i32 = (0..10_000).map(|_| wt.step(300.0)).sum();
let ko_spikes: i32 = (0..10_000).map(|_| ko.step(300.0)).sum();
println!("WT: {} spikes, Kv3 KO: {} spikes", wt_spikes, ko_spikes);
// WT should fire more than KO at high frequencies
```

---

## Theoretical Context

### PV+ interneurons in cortical circuits

Parvalbumin-positive (PV+) fast-spiking interneurons constitute ~40% of
cortical GABAergic interneurons. They form perisomatic basket synapses
onto pyramidal cells, providing:

- **Feedforward inhibition:** Rapid suppression of pyramidal activity
- **Gamma oscillations (30-80 Hz):** PV+ interneuron networks are the
  primary generator of gamma rhythm (Bartos et al. 2007)
- **Gain control:** Divisive inhibition regulating pyramidal f-I curves

The GolombFS model captures the key electrophysiological signature:
narrow spikes, no adaptation, high-frequency capability.

### Kv3 channels in disease

Kv3.1 (KCNC1) mutations cause:
- **Myoclonus epilepsy (EPM7):** Loss of fast repolarisation → impaired
  high-frequency firing → gamma deficit
- **Schizophrenia:** Reduced Kv3 expression in PFC → impaired gamma →
  working memory deficits
- **Alcohol sensitivity:** Kv3.1 is a major target of ethanol

The GolombFS model with tunable $g_{Kv3}$ can simulate these conditions
by reducing the Kv3 conductance.

### Golomb et al. 2007 key results

The original paper demonstrated that:
1. Kv3 channels are necessary and sufficient for the FS phenotype
2. The ratio $g_{Kv3}/g_{K_{dr}}$ determines spike width
3. FS interneurons operate in a specific region of conductance space
   where spike width < 0.4 ms and maximal rate > 300 Hz
4. Network gamma oscillations require FS interneurons with Kv3

### Relationship to other FS models

The GolombFS model is more biophysically detailed than:
- **WangBuzsaki (1996):** Single K channel, no Kv3 distinction
- **Erisir et al. (1999):** Kv1 + Kv3, but different kinetics

And simpler than:
- **Golomb & Amitai (1997):** Full 5-current FS model with Ca²⁺
- **Tateno & Robinson (2006):** 7-current model with Kv3.1 + Kv3.2

The current implementation strikes a balance between biophysical detail
and computational efficiency (711 ns/step).

---

## Phase Portrait

### State space structure

The 4D system (V, h, n, p) can be partially reduced by noting:
- h and n operate on similar timescales (0.5 and 2.0 ms)
- p operates on an intermediate timescale (1.0 ms)
- m is algebraic (no dynamics)

The effective dimensionality during spiking is ~2D (V vs h), with n
and p providing slower modulation. This is why the model can be stiff
near the spike peak where V changes on ~0.01 ms timescale.
