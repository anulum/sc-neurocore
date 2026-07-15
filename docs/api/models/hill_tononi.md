# HillTononiNeuron

**Module:** `sc_neurocore.neurons.models.hill_tononi`
**Rust:** `sc_neurocore_engine::neurons::biophysical::HillTononiNeuron`
**Reference:** Hill, S. & Tononi, G. (2005)
**Publication:** *Modeling sleep and wakefulness in the thalamocortical system.* Journal of Neurophysiology, 93(3), 1671–1698.
**Family:** Conductance-based (thalamocortical, Hodgkin-Huxley type)
**State variables:** `v` (voltage), `h_na` (Na inactivation), `n_k` (K activation),
`m_h` (Ih activation), `h_t` (T-type Ca inactivation), `na_i` (intracellular Na concentration)

---

## 1. Mathematical Formalism

The Hill-Tononi model is a single-compartment conductance-based neuron
with six ionic currents designed to reproduce thalamocortical relay cell
dynamics during sleep-wake transitions. The full system from Hill &
Tononi (2005) Equations 1–8:

### 1.1 Membrane Equation

$$
C_m \frac{dV}{dt} = -I_{Na} - I_K - I_h - I_T - I_{KNa} - I_L + I_{\text{ext}}
$$

where $C_m = 1$ µF/cm² (normalised out of the implementation).

### 1.2 Ionic Currents

**Transient sodium (I_Na):**

$$
I_{Na} = g_{Na} \cdot m_{Na,\infty}^3 \cdot h_{Na} \cdot (V - E_{Na})
$$

where $m_{Na}$ is instantaneous (no differential equation).

**Delayed-rectifier potassium (I_K):**

$$
I_K = g_K \cdot n_K^4 \cdot (V - E_K)
$$

**Hyperpolarisation-activated cation current (I_h):**

$$
I_h = g_h \cdot m_h \cdot (V - E_h)
$$

**Low-threshold T-type calcium (I_T):**

$$
I_T = g_T \cdot m_{T,\infty}^2 \cdot h_T \cdot (V - E_{Ca})
$$

where $m_T$ is instantaneous.

**Na-dependent potassium (I_KNa):**

$$
I_{KNa} = g_{KNa} \cdot w_{KNa}([Na]_i) \cdot (V - E_K)
$$

$$
w_{KNa} = \frac{0.37}{1 + \left(\frac{38.7}{[Na]_i}\right)^{3.5}}
$$

**Leak (I_L):**

$$
I_L = g_L \cdot (V - E_L)
$$

### 1.3 Steady-State Activation/Inactivation

All gating variables use Boltzmann sigmoids:

| Variable | Formula | V₁/₂ (mV) | k (mV) |
|----------|---------|-----------|--------|
| $m_{Na,\infty}$ | $\frac{1}{1 + \exp(-(V + 38) / 10)}$ | -38 | 10 |
| $h_{Na,\infty}$ | $\frac{1}{1 + \exp((V + 43) / 6)}$ | -43 | 6 |
| $n_{K,\infty}$ | $\frac{1}{1 + \exp(-(V + 27) / 11.5)}$ | -27 | 11.5 |
| $m_{h,\infty}$ | $\frac{1}{1 + \exp((V + 75) / 5.5)}$ | -75 | 5.5 |
| $m_{T,\infty}$ | $\frac{1}{1 + \exp(-(V + 59) / 6.2)}$ | -59 | 6.2 |
| $h_{T,\infty}$ | $\frac{1}{1 + \exp((V + 83) / 4)}$ | -83 | 4 |

### 1.4 Time Constants

$$
\tau_{h_{Na}} = \max\!\left(1 + \frac{10}{1 + \exp\!\left(\frac{V + 40}{10}\right)},\, 0.1\right)
$$

$$
\tau_{n_K} = \max\!\left(5 + 47 \cdot \exp\!\left(-\left(\frac{V + 50}{25}\right)^2\right),\, 0.1\right)
$$

$$
\tau_{m_h} = \max\!\left(20 + \frac{1000}{\exp\!\left(\frac{V + 71.5}{14.2}\right) + \exp\!\left(-\frac{V + 89}{11.6}\right)},\, 1.0\right)
$$

$$
\tau_{h_T} = \begin{cases}
\max\!\left(30.8 + \frac{211.4 \cdot \exp\!\left(\frac{V + 115.2}{5}\right)}{1 + \exp\!\left(\frac{V + 86}{3.2}\right)},\, 0.1\right) & \text{if } V < -81 \\
10 & \text{otherwise}
\end{cases}
$$

### 1.5 Intracellular Sodium Dynamics

$$
\frac{d[Na]_i}{dt} = -0.001 \cdot I_{Na} - J_{\text{pump}}\!\left([Na]_i\right)
$$

$$
J_{\text{pump}} = J_{\max} \cdot \frac{[Na]_i}{[Na]_i + [Na]_{\text{eq}}}
$$

where $J_{\max} = 20$ mM/s and $[Na]_{\text{eq}} = 9.5$ mM.
The factor $-0.001$ converts current to concentration flux.
$[Na]_i$ is clamped to $\geq 0$.

### 1.6 Gating Variable Update (Forward Euler)

$$
x \leftarrow x + \frac{x_\infty - x}{\tau_x} \cdot \Delta t
$$

for $x \in \{h_{Na}, n_K, m_h, h_T\}$.

### 1.7 Spike Detection

Threshold crossing: spike when $V(t) \geq V_{\text{thresh}}$ and
$V(t - \Delta t) < V_{\text{thresh}}$, with $V_{\text{thresh}} = -20$ mV.

---

## 2. Theoretical Context

### 2.1 Background

Hill & Tononi (2005) constructed this model to study the mechanisms
underlying the transition between sleep and wakefulness in
thalamocortical circuits. The model reproduces two key physiological
modes of thalamic relay neurons:

1. **Tonic mode (wake):** At depolarised resting potentials, the
   T-type Ca channel is inactivated ($h_T \approx 0$), and the neuron
   fires regular tonic spikes driven by I_Na/I_K.

2. **Burst mode (sleep):** At hyperpolarised potentials (from cortical
   disfacilitation or neuromodulator withdrawal), I_T de-inactivates
   ($h_T \to 1$). A small depolarisation then triggers a low-threshold
   Ca spike crowned with a burst of Na spikes — the signature of
   thalamic sleep oscillations.

### 2.2 Role of I_KNa

The Na-dependent potassium current I_KNa is the key homeostatic
mechanism in the model. During sustained firing:

1. I_Na drives sodium influx, increasing $[Na]_i$
2. Rising $[Na]_i$ activates $w_{KNa}$, increasing I_KNa
3. I_KNa hyperpolarises the cell, reducing firing
4. The Na/K pump gradually restores $[Na]_i$ toward equilibrium

This creates a negative feedback loop operating on a time scale of
seconds, modelling the slow component of sleep pressure. Hill & Tononi
(2005) showed that this mechanism alone can produce alternating epochs
of firing and silence resembling UP/DOWN states.

### 2.3 Role of I_h (HCN Current)

The hyperpolarisation-activated current I_h contributes to:

- **Rebound excitation:** After hyperpolarisation, I_h slowly activates
  (τ_m_h up to ~1 s), depolarising the cell back toward threshold
- **Pacemaker activity:** The interplay between I_h and I_T creates
  intrinsic oscillations even without external input (delta rhythm,
  ~1–4 Hz during NREM sleep)
- **Resting potential regulation:** I_h keeps the resting potential
  closer to -60 mV than pure leak would

### 2.4 Relation to Other Models

- **McCormick & Huguenard (1992):** Earlier thalamic model with I_T
  and I_h but without I_KNa or Na dynamics. Hill-Tononi adds the
  homeostatic sleep mechanism.
- **Bazhenov et al. (2002):** Large-scale thalamocortical network
  model using similar currents but different parameterisation.
- **Destexhe et al. (1994):** Simplified thalamic relay model. Hill-
  Tononi includes more complete Na dynamics.
- **Izhikevich (2003):** Phenomenological alternative — captures burst/
  tonic mode switching with 2 variables, but without biophysical
  currents.

### 2.5 Non-Monotonic f-I Relationship

Unlike standard HH-type models with monotonic f-I curves, the Hill-
Tononi model can show non-monotonic behaviour at high input currents.
Strong depolarisation pushes voltage past the T-current window
(h_T inactivates, m_T saturates), reducing the contribution of I_T to
burst generation. Combined with strong I_KNa activation from elevated
Na, this can actually decrease firing rate at very high drive — a
biologically realistic feature of thalamic relay cells.

---

## 3. Pipeline Position

```
sc_neurocore Pipeline
├── Python layer
│   └── sc_neurocore.neurons.models.hill_tononi.HillTononiNeuron
│       ├── step(current) → int {0, 1}
│       ├── reset() → None
│       ├── Population(HillTononiNeuron, n=N)
│       ├── Network(pop, drive, monitor)
│       ├── PoissonInput(weight=5, rate=200Hz)
│       └── Analysis: spike_count(), firing_rate(), isi()
│
├── Rust engine
│   └── sc_neurocore_engine::neurons::biophysical::HillTononiNeuron
│       ├── new() → Self
│       ├── step(&mut self, current: f64) → i32
│       └── reset(&mut self)
│
├── PyO3 binding
│   └── sc_neurocore_engine.HillTononiNeuron (Python class)
│       ├── __init__()
│       ├── step(current) → int
│       ├── reset()
│       └── get_state() → dict {v, h_na, n_k, m_h, h_t, na_i}
│
├── Network runner
│   └── NeuronVariant::HillTononi(HillTononiNeuron)
│       ├── Wired in network_runner.rs:198
│       ├── Voltage access: network_runner.rs:474
│       └── Factory: "HillTononi" | "HillTononiNeuron" → new()
│
└── Verilog target (planned)
    └── 6 channels + Na pump, ~350 LUTs estimated
```

### 3.1 Data Flow

1. External current $I_{\text{ext}}$ enters via `step(current)`
2. Steady-state activation/inactivation computed for all gates
3. Voltage-dependent time constants computed
4. Gating variables updated (forward Euler)
5. Six ionic currents computed
6. Membrane voltage updated
7. Intracellular Na concentration updated with pump dynamics
8. Spike detection via threshold crossing
9. Returns binary spike indicator (0 or 1)

---

## 4. Features

### 4.1 Core Features

- **Six ionic currents:** I_Na, I_K, I_h, I_T, I_KNa, I_L
- **Intracellular sodium dynamics:** [Na]_i tracked with Na/K pump
- **Burst/tonic mode switching:** I_T de-inactivation at hyperpolarised
  potentials enables thalamic burst firing
- **Sleep homeostasis:** I_KNa provides slow negative feedback via
  Na accumulation, modelling sleep pressure
- **Intrinsic oscillation:** I_h/I_T rebound creates rhythmic firing
  even at zero external input (delta rhythm)
- **Six state variables:** Complete biophysical model with full state

### 4.2 Supported Operations

| Operation | Python | Rust | PyO3 |
|-----------|--------|------|------|
| step(current) → spike | ✅ | ✅ | ✅ |
| reset() | ✅ | ✅ | ✅ |
| get_state() → dict | — | — | ✅ (6 vars) |
| Population wrapping | ✅ | via NeuronVariant | — |
| Network integration | ✅ | ✅ | — |
| PoissonInput drive | ✅ | — | — |
| Spike analysis | ✅ | — | — |

### 4.3 Parameter Sensitivity

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `g_t` ↑ | More prominent burst mode | 0.5–5.0 mS/cm² |
| `g_h` ↑ | Stronger rebound, faster oscillation | 0.1–2.0 mS/cm² |
| `g_kna` ↑ | Stronger sleep pressure, shorter UP states | 0.5–3.0 mS/cm² |
| `na_pump_max` ↑ | Faster Na recovery, shorter DOWN states | 5–40 mM/s |
| `na_eq` ↑ | Higher Na steady-state, more I_KNa at rest | 5–15 mM |
| `g_na` ↑ | Larger Na spikes, faster Na accumulation | 20–100 mS/cm² |

---

## 5. Usage Examples

### 5.1 Basic Tonic Firing (Python)

```python
from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron

neuron = HillTononiNeuron()
spikes = []
for t in range(10000):
    spike = neuron.step(current=5.0)
    if spike:
        spikes.append(t)

print(f"Spike count: {len(spikes)}")
print(f"Mean firing rate: {len(spikes) / (10000 * 0.05 / 1000):.1f} Hz")
```

### 5.2 Observing Burst/Tonic Transition

```python
from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron

neuron = HillTononiNeuron()
v_trace = []
na_trace = []

# Phase 1: Tonic firing with drive
for _ in range(5000):
    neuron.step(current=5.0)
    v_trace.append(neuron.v)
    na_trace.append(neuron.na_i)

# Phase 2: Remove drive — observe rebound bursting
for _ in range(5000):
    neuron.step(current=0.0)
    v_trace.append(neuron.v)
    na_trace.append(neuron.na_i)

# na_i rises during tonic firing (Phase 1)
# After drive removal, I_h rebound + I_T de-inactivation → bursts
```

### 5.3 Na Accumulation Dynamics

```python
from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron

neuron = HillTononiNeuron()
na_initial = neuron.na_i  # 5.0 mM
for _ in range(2000):
    neuron.step(current=10.0)
na_after = neuron.na_i
print(f"Na: {na_initial:.1f} → {na_after:.2f} mM (delta={na_after-na_initial:+.2f})")
```

### 5.4 Population Simulation

```python
from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput

pop = Population(HillTononiNeuron, n=50, label="thalamic_relay")
drive = PoissonInput(n=50, rate_hz=200.0, weight=5.0, dt=0.001, seed=42)
monitor = SpikeMonitor(pop)
net = Network(pop, drive, monitor)
net.run(duration=1.0, dt=0.001, backend="python")
print(f"Total spikes: {monitor.count}")
```

### 5.5 Rust Backend (via PyO3)

```python
from sc_neurocore_engine import HillTononiNeuron as RustHT

neuron = RustHT()
spikes = sum(neuron.step(5.0) for _ in range(10000))
state = neuron.get_state()
print(f"Spikes: {spikes}")
print(f"V={state['v']:.2f}, Na_i={state['na_i']:.3f}")
print(f"h_na={state['h_na']:.3f}, n_k={state['n_k']:.3f}")
print(f"m_h={state['m_h']:.3f}, h_t={state['h_t']:.3f}")
```

---

## 6. Technical Reference

### 6.1 Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane voltage (initial) |
| `h_na` | 0.6 | — | Na inactivation gate (initial) |
| `n_k` | 0.3 | — | K activation gate (initial) |
| `m_h` | 0.0 | — | I_h activation gate (initial) |
| `h_t` | 0.9 | — | T-type Ca inactivation gate (initial) |
| `na_i` | 5.0 | mM | Intracellular Na concentration (initial) |
| `g_na` | 50.0 | mS/cm² | Na maximal conductance |
| `g_k` | 5.0 | mS/cm² | K (delayed rectifier) conductance |
| `g_h` | 1.0 | mS/cm² | I_h (HCN) maximal conductance |
| `g_t` | 3.0 | mS/cm² | T-type Ca maximal conductance |
| `g_kna` | 1.33 | mS/cm² | Na-dependent K conductance |
| `g_l` | 0.02 | mS/cm² | Leak conductance |
| `e_na` | 50.0 | mV | Na reversal potential |
| `e_k` | -90.0 | mV | K reversal potential |
| `e_h` | -43.0 | mV | I_h reversal potential |
| `e_ca` | 120.0 | mV | Ca reversal potential |
| `e_l` | -70.0 | mV | Leak reversal potential |
| `na_pump_max` | 20.0 | mM/s | Na/K pump maximum rate |
| `na_eq` | 9.5 | mM | Na pump half-activation |
| `dt` | 0.05 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |

### 6.2 State Variables

| Variable | Type | Description |
|----------|------|-------------|
| `v` | f64 / float | Membrane voltage |
| `h_na` | f64 / float | Na inactivation gating variable |
| `n_k` | f64 / float | K activation gating variable |
| `m_h` | f64 / float | I_h activation gating variable |
| `h_t` | f64 / float | T-type Ca inactivation gating variable |
| `na_i` | f64 / float | Intracellular Na concentration |

### 6.3 Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `(current: f64) → i32` | 0 or 1 | Advance one timestep |
| `reset` | `() → ()` | — | Reset all 6 state variables |
| `new` | `() → Self` | — | Rust constructor with defaults |
| `get_state` | `() → dict` | 6 vars | PyO3 only: state inspection |

### 6.4 Python/Rust Implementation Comparison

| Aspect | Python | Rust |
|--------|--------|------|
| Source | `hill_tononi.py` (85 lines) | `engine/src/neurons/biophysical/hill_tononi.rs` |
| Gating sigmoids | Identical (6 Boltzmann functions) | Identical |
| Time constants | Identical (4 voltage-dependent) | Identical (after b515e5c fix) |
| Currents | 6 currents, same formulae | 6 currents, same formulae |
| Na dynamics | -0.001·I_Na - pump, clamp ≥ 0 | -0.001·I_Na - pump, clamp ≥ 0 |
| tau_n_k | `5 + 47·exp(-((V+50)/25)²)` | `5 + 47·exp(-((V+50)/25)²)` (fixed) |
| **Parity** | **EXACT** (after tau_n_k fix) | |

### 6.5 NeuronVariant Wiring

```rust
// network_runner.rs:198
HillTononi(HillTononiNeuron),

// network_runner.rs:474 — voltage access
NeuronVariant::HillTononi(n) => n.v,

// network_runner.rs:918 — factory
"HillTononi" | "HillTononiNeuron" => Ok(NeuronVariant::HillTononi(HillTononiNeuron::new())),
```

---

## 7. Performance Benchmarks

### 7.1 Rust (Criterion 0.8)

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-05.

| Benchmark | Iterations | Median | Per-step | Notes |
|-----------|-----------|--------|----------|-------|
| `hill_tononi_1k_steps` | 1,000 | 248 µs | **248.4 ns** | 6 currents + Na pump per step |

### 7.2 Python

Measured on same hardware, single-threaded, 2026-04-04.

| Metric | Value |
|--------|-------|
| Isolation throughput | ~28K steps/s (~35.7 µs/step) |
| Spikes (10K steps, I=5.0) | 35 |

### 7.3 Speedup

| Metric | Python | Rust | Speedup |
|--------|--------|------|---------|
| Per-step latency | ~35,700 ns | 248.4 ns | **~144×** |

The lower speedup compared to simpler models (e.g., GIF at 220×)
reflects the computational cost of 6 ionic currents, 4 gating ODEs,
Na dynamics, and the Na/K pump — all of which involve transcendental
functions (exp, powf) that limit vectorisation gains.

### 7.4 Numerical Stability

| Test | Duration | Result |
|------|----------|--------|
| 20,000 steps at I=5.0 | 1 s sim time | All 6 state variables finite |
| 200 steps at I=100 | 10 ms sim time | Voltage finite |
| 200 steps at I=-30 | 10 ms sim time | Voltage finite |
| Na non-negativity | 10K steps, varying I | na_i ≥ 0 always (clamped) |

---

## 8. Test Coverage

### 8.1 Python Tests (30 total)

**File:** `tests/test_model_hill_tononi.py` (28 tests)

| Class | Tests | What is verified |
|-------|-------|-----------------|
| TestIsolation | 4 | Construction, step returns {0,1}, 6 state variables, reset |
| TestDynamics | 6 | Fires at I=5, intrinsic oscillation at I=0, Na accumulation, Na non-negative, T-gate evolution, I_h gate evolution |
| TestCurrents | 5 | I_KNa activation with Na, I_h hyperpolarisation, I_T burst, Na pump, w_kna sigmoidal |
| TestStability | 4 | 20K steps finite, negative input, extreme input, NaN detection |
| TestPerformance | 2 | Isolation throughput, network throughput |
| TestPipeline | 4 | Population, projection wiring, network spikes, spike analysis |
| TestParametric | 3 | g_t sweep, g_h sweep, g_kna sweep |

**File:** `tests/test_model_hill_tononi.py` (dedicated model tests)

| Test | What is verified |
|------|-----------------|
| `test_fires` | Fires under I=5 in 300 steps |
| `test_h_current_evolves` | I_h gate changes under drive |

### 8.2 Rust Tests (7 total)

**File:** `engine/src/neurons/biophysical/hill_tononi.rs`

| Test | What is verified |
|------|-----------------|
| `hill_tononi_fires` | Fires at I=5 with seed=42 |
| `hill_tononi_silent_without_input` | Intrinsic activity at I=0 (thalamic oscillator) |
| `hill_tononi_reset_clears_state` | All 6 state variables restored |
| `hill_tononi_extreme_bounded` | v finite at I=100 |
| `hill_tononi_na_accumulation` | na_i increases during sustained firing |
| `hill_tononi_negative_no_crash` | v finite at I=-30 |
| `hill_tononi_nan_no_panic` | NaN input does not crash |

### 8.3 Coverage Summary

| Category | Python | Rust | Total |
|----------|--------|------|-------|
| Construction/reset | 3 | 1 | 4 |
| Dynamics/spiking | 8 | 2 | 10 |
| Individual currents | 5 | 0 | 5 |
| Na dynamics | 3 | 1 | 4 |
| Numerical stability | 4 | 3 | 7 |
| Performance | 2 | 0 | 2 |
| Pipeline integration | 4 | 0 | 4 |
| Parametric sweeps | 3 | 0 | 3 |
| **Total** | **30** | **7** | **37** |

---

## 9. Citations

1. **Hill, S. & Tononi, G.** (2005).
   Modeling sleep and wakefulness in the thalamocortical system.
   *Journal of Neurophysiology*, 93(3), 1671–1698.
   DOI: [10.1152/jn.00915.2004](https://doi.org/10.1152/jn.00915.2004)

2. **Bazhenov, M., Timofeev, I., Steriade, M., & Sejnowski, T. J.** (2002).
   Model of thalamocortical slow-wave sleep oscillations and transitions to activated states.
   *Journal of Neuroscience*, 22(19), 8691–8704.
   DOI: [10.1523/JNEUROSCI.22-19-08691.2002](https://doi.org/10.1523/JNEUROSCI.22-19-08691.2002)

3. **McCormick, D. A. & Huguenard, J. R.** (1992).
   A model of the electrophysiological properties of thalamocortical relay neurons.
   *Journal of Neurophysiology*, 68(4), 1384–1400.
   DOI: [10.1152/jn.1992.68.4.1384](https://doi.org/10.1152/jn.1992.68.4.1384)

4. **Destexhe, A., Contreras, D., Sejnowski, T. J., & Steriade, M.** (1994).
   A model of spindle rhythmicity in the isolated thalamic reticular nucleus.
   *Journal of Neurophysiology*, 72(2), 803–818.
   DOI: [10.1152/jn.1994.72.2.803](https://doi.org/10.1152/jn.1994.72.2.803)

5. **Compte, A., Sanchez-Vives, M. V., McCormick, D. A., & Wang, X.-J.** (2003).
   Cellular and network mechanisms of slow oscillatory activity (<1 Hz) and wave propagations
   in a cortical network model.
   *Journal of Neurophysiology*, 89(5), 2707–2725.
   DOI: [10.1152/jn.00845.2002](https://doi.org/10.1152/jn.00845.2002)

6. **Tononi, G. & Cirelli, C.** (2006).
   Sleep function and synaptic homeostasis.
   *Sleep Medicine Reviews*, 10(1), 49–62.
   DOI: [10.1016/j.smrv.2005.05.002](https://doi.org/10.1016/j.smrv.2005.05.002)

7. **Huguenard, J. R. & McCormick, D. A.** (1992).
   Simulation of the currents involved in rhythmic oscillations in thalamic relay neurons.
   *Journal of Neurophysiology*, 68(4), 1373–1383.
   DOI: [10.1152/jn.1992.68.4.1373](https://doi.org/10.1152/jn.1992.68.4.1373)

---

*SC-NeuroCore v3.14.0 — ANULUM / Fortis Studio*
*© 2020–2026 Miroslav Šotek. All rights reserved.*
