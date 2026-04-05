# GIFPopulationNeuron

**Module:** `sc_neurocore.neurons.models.gif_population`
**Rust:** `sc_neurocore_engine::neurons::biophysical::GIFPopulationNeuron`
**Reference:** Mensi, Naud, Pozzorini, Avermann, Petersen & Gerstner (2012)
**Publication:** *Parameter extraction and classification of three cortical neuron types reveals two distinct adaptation mechanisms.* Journal of Neurophysiology, 107(6), 1756–1775.
**Family:** Integrate-and-fire (generalised, stochastic, escape-rate)
**State variables:** `v` (membrane voltage, mV), `eta` (adaptation current, mV)

---

## 1. Mathematical Formalism

The GIF (Generalised Integrate-and-Fire) model combines subthreshold
leaky integration with a stochastic escape-rate threshold mechanism
and spike-triggered adaptation. The complete system from Mensi et al.
(2012) Equations 1–4:

### 1.1 Subthreshold Dynamics

$$
\tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) - \eta + I_{\text{ext}}
$$

where $V$ is the membrane potential, $V_{\text{rest}}$ is the resting
potential, $\eta$ is the spike-triggered adaptation current, and
$I_{\text{ext}}$ is the external drive.

### 1.2 Adaptation Current Decay

Between spikes, the adaptation current $\eta$ decays exponentially:

$$
\eta(t + \Delta t) = \eta(t) \cdot \exp\!\left(-\frac{\Delta t}{\tau_\eta}\right)
$$

where $\tau_\eta$ is the adaptation time constant.

### 1.3 Escape-Rate Hazard Function

The instantaneous firing rate (hazard) follows an exponential
escape-rate model:

$$
\lambda(V) = \lambda_0 \exp\!\left(\frac{V - \theta}{\Delta_V}\right)
$$

where $\lambda_0$ is the baseline hazard rate, $\theta$ is the soft
threshold voltage, and $\Delta_V$ controls the sharpness of the
stochastic threshold. Smaller $\Delta_V$ approaches a hard threshold;
larger $\Delta_V$ produces broader firing probability distributions.

The exponent is clamped at 20.0 in both Python and Rust to prevent
numerical overflow: $\min\!\left(\frac{V - \theta}{\Delta_V},\, 20\right)$.

### 1.4 Spike Probability

The probability of firing in the interval $[t, t + \Delta t]$ is:

$$
P(\text{spike}) = 1 - \exp(-\lambda \cdot \Delta t)
$$

This is the exact Poisson process probability, not the linear
approximation $\lambda \cdot \Delta t$.

### 1.5 Spike-Triggered Update

On spike ($V \to V_{\text{reset}}$):

$$
V \leftarrow V_{\text{reset}}, \qquad \eta \leftarrow \eta + \eta_{\text{inc}}
$$

The adaptation current $\eta$ accumulates after each spike with
increment $\eta_{\text{inc}}$, producing spike-frequency adaptation:
sustained firing gradually increases $\eta$, which opposes further
depolarisation (Eq. 1), reducing subsequent firing probability.

### 1.6 Euler Integration (Implementation)

Both Python and Rust use forward Euler with timestep `dt`:

```
V += (-(V - V_rest) - eta + I_ext) / tau_m * dt
eta *= exp(-dt / tau_eta)
exponent = min((V - theta) / delta_v, 20.0)
hazard = lambda_0 * exp(exponent)
p_spike = 1 - exp(-hazard * dt)
if uniform_random() < p_spike:
    V = V_reset
    eta += eta_increment
    spike = 1
else:
    spike = 0
```

---

## 2. Theoretical Context

### 2.1 Background

The GIF model was introduced by Mensi et al. (2012) as a
computationally efficient yet biologically grounded framework for
classifying cortical neuron types. The model captures two key features
of real neurons that simpler IF models miss:

1. **Stochastic threshold:** Real neurons do not fire deterministically
   at a fixed voltage. The escape-rate mechanism models threshold
   variability via an exponential hazard function, producing trial-to-
   trial variability consistent with in vivo recordings.

2. **Spike-frequency adaptation:** Most cortical neurons reduce their
   firing rate during sustained stimulation. The additive adaptation
   current $\eta$ accumulates with each spike and decays between
   spikes, reproducing both fast and slow adaptation time scales.

### 2.2 Relation to Other Models

The GIF model is positioned between:

- **LIF** (Leaky Integrate-and-Fire): Hard threshold, no adaptation,
  deterministic. GIF adds stochastic threshold + adaptation.
- **EIF** (Exponential IF, Fourcaud-Trocmé et al. 2003): Adds
  exponential voltage-dependent current for spike initiation, but
  remains deterministic. GIF replaces voltage instability with
  stochastic firing.
- **GLM** (Generalised Linear Model, Pillow et al. 2008): Statistical
  model with similar escape-rate mechanism, but GLM typically uses
  spike history filters rather than explicit adaptation variables.
- **AdEx** (Brette & Gerstner 2005): Deterministic exponential IF with
  adaptation current. GIF replaces the hard reset with stochastic
  firing.

### 2.3 Classification Result

Mensi et al. (2012) fitted the GIF model to recordings from layer 5
pyramidal neurons, fast-spiking interneurons, and intrinsically
bursting neurons in mouse somatosensory cortex. The model correctly
classified all three types based on their adaptation parameters
($\tau_\eta$, $\eta_{\text{inc}}$).

### 2.4 Population-Level Usage

The "Population" suffix indicates this model is designed for mean-field
and population simulations. Due to the stochastic threshold, a
population of identical neurons driven with the same input produces
asynchronous firing — a requirement for realistic population dynamics
without explicit noise injection.

### 2.5 Adaptation Mechanism Classification

Mensi et al. (2012) identified two distinct adaptation mechanisms in
cortical neurons using the GIF model:

- **Type A (fast adaptation):** Short τ_η (~30–80 ms), moderate
  η_increment. Found predominantly in fast-spiking interneurons.
  Produces phasic onset response followed by sustained tonic firing.

- **Type B (slow adaptation):** Long τ_η (~100–500 ms), large
  η_increment. Found in regular-spiking pyramidal neurons and
  intrinsically bursting cells. Produces pronounced spike-frequency
  adaptation over hundreds of milliseconds.

The default parameters (τ_η = 100 ms, η_inc = 5 mV) correspond to
a moderate Type B adaptation profile.

### 2.6 Escape-Rate vs Threshold Noise

Two approaches exist for modelling stochastic thresholds:

1. **Threshold noise:** Add Gaussian noise to a fixed threshold
   (V_thresh + σξ). Simple but biologically implausible — the noise
   source is external to the membrane dynamics.

2. **Escape-rate:** The firing probability depends exponentially on
   how far voltage exceeds the soft threshold. This is the approach
   used here. It naturally emerges from channel noise theory (Schwalger
   et al. 2015) and produces statistics consistent with in vivo
   cortical recordings.

The escape-rate parameter Δ_V controls the transition: Δ_V → 0
recovers a deterministic threshold; Δ_V → ∞ produces rate-independent
random firing.

---

## 3. Pipeline Position

```
sc_neurocore Pipeline
├── Python layer
│   └── sc_neurocore.neurons.models.gif_population.GIFPopulationNeuron
│       ├── step(current) → int {0, 1}
│       ├── reset() → None
│       ├── Population(GIFPopulationNeuron, n=N)
│       ├── Network(pop, drive, monitor)
│       ├── PoissonInput(weight, rate_hz)
│       └── Analysis: spike_count(), firing_rate(), isi()
│
├── Rust engine
│   └── sc_neurocore_engine::neurons::biophysical::GIFPopulationNeuron
│       ├── new(seed: u64) → Self
│       ├── step(&mut self, current: f64) → i32
│       └── reset(&mut self)
│
├── PyO3 binding
│   └── sc_neurocore_engine.GIFPopulationNeuron (Python class)
│       ├── __init__(seed=42)
│       ├── step(current) → int
│       ├── reset()
│       └── get_state() → dict {v, theta, eta}
│
├── Network runner
│   └── NeuronVariant::GIFPopulation(GIFPopulationNeuron)
│       ├── Wired in network_runner.rs:195
│       ├── Voltage access: network_runner.rs:471
│       └── Factory: "GIFPopulation" | "GIFPopulationNeuron" → new(42)
│
└── Verilog target (planned)
    └── exp LUT + LFSR + adaptation register, ~70 LUTs estimated
```

### 3.1 Data Flow

1. External current $I_{\text{ext}}$ enters via `step(current)`
2. Membrane equation updates `v` (forward Euler)
3. Adaptation `eta` decays exponentially
4. Hazard function computes instantaneous firing rate
5. RNG draw determines spike/no-spike
6. On spike: voltage resets, adaptation increments
7. Returns binary spike indicator (0 or 1)

### 3.2 RNG Implementation

- **Python:** `numpy.random.default_rng()` (PCG64)
- **Rust:** `Xoshiro256PlusPlus` seeded from `u64`
- **Consequence:** Same seed produces same spike train within each
  backend, but cross-backend exact parity is not achievable due to
  different RNG algorithms. Statistical properties (mean rate,
  adaptation time course) are equivalent.

---

## 4. Features

### 4.1 Core Features

- **Stochastic escape-rate threshold:** Spike probability increases
  smoothly with membrane voltage, not a hard threshold
- **Spike-frequency adaptation:** Cumulative adaptation current reduces
  firing rate over time during sustained drive
- **Two time scales:** Membrane time constant τ_m (fast) and adaptation
  time constant τ_η (slow) enable rich temporal dynamics
- **Seed-based reproducibility:** Deterministic RNG per seed for
  reproducible stochastic simulations
- **Numerical safety:** Exponent clamped at 20.0 prevents overflow

### 4.2 Supported Operations

| Operation | Python | Rust | PyO3 |
|-----------|--------|------|------|
| step(current) → spike | ✅ | ✅ | ✅ |
| reset() | ✅ | ✅ | ✅ |
| get_state() → dict | — | — | ✅ |
| Population wrapping | ✅ | via NeuronVariant | — |
| Network integration | ✅ | ✅ | — |
| PoissonInput drive | ✅ | — | — |
| Spike analysis | ✅ | — | — |

### 4.3 Parameter Sensitivity

| Parameter | Effect | Range |
|-----------|--------|-------|
| `delta_v` ↓ | Sharper threshold → more deterministic | 0.5–5.0 mV |
| `delta_v` ↑ | Softer threshold → more stochastic | 5.0–20.0 mV |
| `lambda_0` ↑ | Higher baseline rate → more spontaneous firing | 0.0001–0.01 ms⁻¹ |
| `eta_increment` ↑ | Stronger adaptation → faster rate decrease | 1.0–20.0 mV |
| `tau_eta` ↑ | Slower adaptation recovery → prolonged rate decrease | 50–500 ms |
| `tau_m` ↓ | Faster membrane → quicker voltage response | 5–50 ms |

---

## 5. Usage Examples

### 5.1 Basic Simulation (Python)

```python
from sc_neurocore.neurons.models.gif_population import GIFPopulationNeuron

neuron = GIFPopulationNeuron()
spikes = []
for t in range(10000):
    spike = neuron.step(current=50.0)
    if spike:
        spikes.append(t)

print(f"Spike count: {len(spikes)}")
print(f"Mean ISI: {sum(b-a for a,b in zip(spikes, spikes[1:])) / max(len(spikes)-1, 1):.1f} steps")
```

### 5.2 Population Simulation

```python
from sc_neurocore.neurons.models.gif_population import GIFPopulationNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput

pop = Population(GIFPopulationNeuron, n=100, label="gif_pop")
drive = PoissonInput(n=100, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
monitor = SpikeMonitor(pop)
net = Network(pop, drive, monitor)
net.run(duration=1.0, dt=0.001, backend="python")

print(f"Total spikes: {monitor.count}")
```

### 5.3 Adaptation Dynamics Visualisation

```python
import numpy as np
from sc_neurocore.neurons.models.gif_population import GIFPopulationNeuron

neuron = GIFPopulationNeuron()
eta_trace = []
v_trace = []
for _ in range(5000):
    neuron.step(current=50.0)
    v_trace.append(neuron.v)
    eta_trace.append(neuron.eta)

# eta increases on spikes (visible jumps) and decays between spikes
# v fluctuates with stochastic resets on spikes
```

### 5.4 Rust Backend (via PyO3)

```python
from sc_neurocore_engine import GIFPopulationNeuron as RustGIF

neuron = RustGIF(seed=42)
spikes = sum(neuron.step(50.0) for _ in range(10000))
print(f"Rust spikes: {spikes}")

state = neuron.get_state()
print(f"V={state['v']:.2f} mV, eta={state['eta']:.2f} mV")
```

### 5.5 Projection Wiring

```python
from sc_neurocore.neurons.models.gif_population import GIFPopulationNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput

src = Population(GIFPopulationNeuron, n=50, label="src")
tgt = Population(GIFPopulationNeuron, n=50, label="tgt")
drive = PoissonInput(n=50, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
mon = SpikeMonitor(src)
net = Network(src, tgt, drive, proj, mon)
net.run(duration=2.0, dt=0.001, backend="python")
print(f"Source spikes: {mon.count}")
```

---

## 6. Technical Reference

### 6.1 Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane voltage (initial) |
| `theta` | -50.0 | mV | Baseline threshold for escape-rate |
| `eta` | 0.0 | mV | Adaptation current (initial) |
| `tau_m` | 20.0 | ms | Membrane time constant |
| `tau_eta` | 100.0 | ms | Adaptation decay time constant |
| `delta_v` | 2.0 | mV | Escape-rate sharpness |
| `lambda_0` | 0.001 | ms⁻¹ | Baseline hazard rate |
| `eta_increment` | 5.0 | mV | Spike-triggered adaptation increment |
| `v_rest` | -65.0 | mV | Resting potential |
| `v_reset` | -65.0 | mV | Post-spike reset voltage |
| `dt` | 0.5 | ms | Integration timestep |

### 6.2 State Variables

| Variable | Type | Description |
|----------|------|-------------|
| `v` | f64 / float | Membrane voltage |
| `eta` | f64 / float | Adaptation current |
| `rng` | Xoshiro256++ / np.random.Generator | Stochastic RNG |

### 6.3 Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `(current: f64) → i32` | 0 or 1 | Advance one timestep, return spike indicator |
| `reset` | `() → ()` | — | Reset v to -65.0, eta to 0.0 |
| `new` | `(seed: u64) → Self` | — | Rust constructor with RNG seed |
| `get_state` | `() → dict` | v, theta, eta | PyO3 only: state inspection |

### 6.4 Python/Rust Implementation Comparison

| Aspect | Python | Rust |
|--------|--------|------|
| Source | `gif_population.py` (52 lines) | `biophysical.rs:1157-1205` |
| RNG | numpy PCG64 | Xoshiro256PlusPlus |
| Voltage eq. | `-(V-Vr) - η + I` | `-(V-Vr) - η + I` |
| Hazard | `λ₀ exp(min((V-θ)/ΔV, 20))` | `λ₀ exp(min((V-θ)/ΔV, 20))` |
| Spike prob. | `1 - exp(-λ·dt)` | `1 - exp(-λ·dt)` |
| Exponent clamp | 20.0 | 20.0 |
| **Parity** | **EXACT** (formulae identical, commit b515e5c) | |

### 6.5 NeuronVariant Wiring

```rust
// network_runner.rs:195
GIFPopulation(GIFPopulationNeuron),

// network_runner.rs:471 — voltage access
NeuronVariant::GIFPopulation(n) => n.v,

// network_runner.rs:909 — factory
"GIFPopulation" | "GIFPopulationNeuron" => {
    Ok(NeuronVariant::GIFPopulation(GIFPopulationNeuron::new(42)))
}
```

---

## 7. Performance Benchmarks

### 7.1 Rust (Criterion 0.8)

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-05.

| Benchmark | Iterations | Median | Per-step | Notes |
|-----------|-----------|--------|----------|-------|
| `gif_pop_10k_steps` | 10,000 | 368 µs | **36.8 ns** | Includes RNG draw per step |

### 7.2 Python

Measured on same hardware, single-threaded, 2026-04-04.

| Metric | Value |
|--------|-------|
| Isolation throughput | ~124K steps/s (~8.1 µs/step) |
| Network throughput (50 neurons, 500 steps) | >5K neuron-steps/s |

### 7.3 Speedup

| Metric | Python | Rust | Speedup |
|--------|--------|------|---------|
| Per-step latency | ~8,100 ns | 36.8 ns | **~220×** |

The 220× speedup includes RNG cost in both backends. The Rust
implementation uses Xoshiro256++ which is faster than numpy's PCG64
for scalar draws.

### 7.4 Numerical Stability

| Test | Duration | Result |
|------|----------|--------|
| 20,000 steps at I=50 | 10 s sim time | All state variables finite |
| 200 steps at I=10⁴ | 100 ms sim time | Voltage finite (clamped hazard) |
| 200 steps at I=-30 | 100 ms sim time | Voltage finite |

---

## 8. Test Coverage

### 8.1 Python Tests (13 total)

**File:** `tests/test_model_gif_population.py` (12 tests)

| Class | Test | What is verified |
|-------|------|-----------------|
| TestIsolation | `test_step_returns_binary` | step() output ∈ {0, 1} |
| TestIsolation | `test_state_finite` | v finite after 5000 steps at I=50 |
| TestIsolation | `test_reset` | reset() restores initial state |
| TestDynamics | `test_fires_at_test_current` | ≥10 spikes in 5000 steps at I=50 |
| TestDynamics | `test_rate_increases_with_current` | More spikes at I=100 than I=30 |
| TestDynamics | `test_two_runs_differ` | Stochastic: different RNG → different trains |
| TestPerformance | `test_isolation_throughput` | >20K steps/s |
| TestPerformance | `test_network_throughput` | >5K neuron-steps/s |
| TestPipeline | `test_population` | Population(n=10) creates correctly |
| TestPipeline | `test_network_spikes` | Network produces spikes |
| TestPipeline | `test_projection_wiring` | Projection src→tgt fires |
| TestPipeline | `test_analysis` | spike_count ≥ 5, firing_rate > 0 |

**File:** `tests/test_final_neuron_batch.py` (1 test)

| Test | What is verified |
|------|-----------------|
| `test_stochastic_firing` | Fires under I=30 in 500 steps |

### 8.2 Rust Tests (6 total)

**File:** `engine/src/neurons/biophysical.rs`

| Test | What is verified |
|------|-----------------|
| `gif_pop_fires` | Fires at I=30 with seed=42 |
| `gif_pop_silent_without_input` | <5 spikes at I=0 in 200 steps |
| `gif_pop_reset_clears_state` | v=-65.0, eta=0.0 after reset |
| `gif_pop_extreme_bounded` | v finite at I=10⁴ |
| `gif_pop_stochastic_variability` | seed=1 ≠ seed=999 spike counts |
| `gif_pop_negative_no_crash` | v finite at I=-30 |

### 8.3 Coverage Summary

| Category | Python | Rust | Total |
|----------|--------|------|-------|
| Construction/reset | 2 | 1 | 3 |
| Dynamics/spiking | 4 | 2 | 6 |
| Stochastic properties | 1 | 1 | 2 |
| Numerical stability | 1 | 2 | 3 |
| Performance | 2 | 0 | 2 |
| Pipeline integration | 3 | 0 | 3 |
| **Total** | **13** | **6** | **19** |

---

## 9. Citations

1. **Mensi, S., Naud, R., Pozzorini, C., Avermann, M., Petersen, C. C. H., & Gerstner, W.** (2012).
   Parameter extraction and classification of three cortical neuron types reveals two distinct adaptation mechanisms.
   *Journal of Neurophysiology*, 107(6), 1756–1775.
   DOI: [10.1152/jn.00408.2011](https://doi.org/10.1152/jn.00408.2011)

2. **Pozzorini, C., Mensi, S., Hagens, O., Naud, R., Koch, C., & Gerstner, W.** (2015).
   Automated high-throughput characterization of single neurons by means of simplified spiking models.
   *PLoS Computational Biology*, 11(6), e1004275.
   DOI: [10.1371/journal.pcbi.1004275](https://doi.org/10.1371/journal.pcbi.1004275)

3. **Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L.** (2014).
   *Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.*
   Cambridge University Press. Chapter 9: Escape rate models.

4. **Jolivet, R., Rauch, A., Lüscher, H.-R., & Gerstner, W.** (2006).
   Predicting spike timing of neocortical pyramidal neurons by simple threshold models.
   *Journal of Computational Neuroscience*, 21(1), 35–49.
   DOI: [10.1007/s10827-006-7074-5](https://doi.org/10.1007/s10827-006-7074-5)

5. **Pillow, J. W., Shlens, J., Paninski, L., Sher, A., Litke, A. M., Chichilnisky, E. J., & Simoncelli, E. P.** (2008).
   Spatio-temporal correlations and visual signalling in a complete neuronal population.
   *Nature*, 454(7207), 995–999.

6. **Schwalger, T., Deger, M., & Gerstner, W.** (2017).
   Towards a theory of cortical columns: From spiking neurons to interacting neural populations of finite size.
   *PLoS Computational Biology*, 13(4), e1005507.
   DOI: [10.1371/journal.pcbi.1005507](https://doi.org/10.1371/journal.pcbi.1005507)

7. **Brette, R. & Gerstner, W.** (2005).
   Adaptive exponential integrate-and-fire model as an effective description of neuronal activity.
   *Journal of Neurophysiology*, 94(5), 3637–3642.
   DOI: [10.1152/jn.00686.2005](https://doi.org/10.1152/jn.00686.2005)

---

*SC-NeuroCore v3.14.0 — ANULUM / Fortis Studio*
*© 2020–2026 Miroslav Šotek. All rights reserved.*
