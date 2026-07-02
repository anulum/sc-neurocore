# Model Zoo

**Module:** `sc_neurocore.model_zoo`
**Sub-modules:** `configs` (10 architecture builders), `pretrained` (weight loader)
**Family:** Pre-configured network-level SNN architectures
**Purpose:** One-call construction of complete, runnable spiking neural networks

---

## Overview

The Model Zoo provides factory functions that return fully wired
`Network` objects — populations, projections, stimuli, and monitors —
ready for `net.run(duration)`. Each architecture is parameterised from
a published neuroscience reference and uses biologically plausible
parameter values from the cited papers.

No trained weight files are shipped with the architecture builders.
Weights use random initialisation (Xavier/Glorot or uniform). The
separate `load_pretrained()` function loads pre-initialised `.npz`
weight files for three classifier architectures.

```python
from sc_neurocore.model_zoo import brunel_balanced_network

net = brunel_balanced_network(n_exc=800, n_inh=200)
net.run(1.0)  # 1 second simulation
print(net.spike_monitors[0].count)
```

---

## Architectures

### 1. mnist_classifier

**Reference:** Zenke & Ganguli, Neural Computation 30(6), 2018 (SuperSpike)
**Topology:** 784 → n_hidden → 10 feedforward
**Neuron model:** StochasticLIFNeuron (all layers)

| Component | Count | Notes |
|-----------|-------|-------|
| Populations | 3 | input (784), hidden (configurable), output (10) |
| Projections | 2 | input→hidden, hidden→output |
| Stimuli | 1 | PoissonInput(784, rate=500Hz, weight=2.0) |
| Monitors | 2 | output_spikes, hidden_spikes |

**Parameters:**
- `n_hidden` (int, default=128): Hidden layer size
- LIF params: τ_mem=10ms, v_threshold=1.0, v_rest=0.0, v_reset=0.0, noise_std=0.0
- Weight scaling: Xavier-uniform — w_ih = √(2/784) × 20, w_ho = √(2/n_hidden) × 20
- Connection probability: 0.3 (input→hidden), 0.5 (hidden→output)

**Intended use:** Feedforward classification of rate-coded MNIST digits.
The Poisson input layer encodes pixel intensities as spike rates. The
hidden layer performs nonlinear feature extraction. The output layer
provides a winner-take-all readout (highest firing neuron = predicted digit).

---

### 2. dvs_gesture_classifier

**Reference:** Amir et al., CVPR 2017 (IBM DVS128 gesture recognition)
**Topology:** 256 → 256 → n_classes feedforward
**Neuron model:** StochasticLIFNeuron (all layers)

| Component | Count | Notes |
|-----------|-------|-------|
| Populations | 3 | input (256), hidden (256), output (n_classes) |
| Projections | 2 | input→hidden, hidden→output |
| Stimuli | 1 | PoissonInput(256, rate=800Hz, weight=2.0) |
| Monitors | 1 | gesture_spikes (output only) |

**Parameters:**
- `n_classes` (int, default=11): Number of gesture classes (DVS128 has 11)
- Weight: w_ih=0.5, w_ho=0.8 (uniform, not Xavier)
- Connection probability: 0.2 (input→hidden), 0.4 (hidden→output)

**Intended use:** Event-camera gesture classification. The 256-neuron input
layer represents 16×16 spatial pixels of a DVS128 event stream. Poisson
input at 800 Hz simulates realistic event rates from dynamic gestures.

---

### 3. shd_speech_classifier

**Reference:** Cramer et al., IEEE TNNLS 33(7), 2022 (Spiking Heidelberg Digits)
**Topology:** 700 → 256 (recurrent) → 20
**Neuron model:** StochasticLIFNeuron (all layers)

| Component | Count | Notes |
|-----------|-------|-------|
| Populations | 3 | input (700), recurrent (256), output (20) |
| Projections | 3 | input→rec, rec→rec (recurrent), rec→output |
| Stimuli | 1 | PoissonInput(700, rate=500Hz, weight=2.0) |
| Monitors | 2 | shd_output_spikes, shd_recurrent_spikes |

**Key design features:**
- **Recurrent connectivity:** The rec→rec projection (weight=0.15, p=0.1)
  creates temporal memory — the recurrent layer integrates information
  across time, critical for speech recognition.
- **Longer time constant:** Recurrent layer uses τ_mem=20ms (vs input τ_mem=10ms),
  allowing slower integration dynamics that match speech timescales.
- **20-class output:** Corresponds to the 20 spoken digits in the SHD dataset
  (0–9 in German and English).

---

### 4. brunel_balanced_network

**Reference:** Brunel, J. Comput. Neurosci. 8(3), 2000, Sec. 2
**Topology:** Sparse random E/I with 4:1 ratio
**Neuron model:** StochasticLIFNeuron (excitatory and inhibitory)

| Component | Count | Notes |
|-----------|-------|-------|
| Populations | 2 | excitatory (n_exc), inhibitory (n_inh) |
| Projections | 4 | E→E, E→I, I→E, I→I (full 2×2 matrix) |
| Stimuli | 2 | PoissonInput for E and I populations (1000Hz, weight=2.0) |
| Monitors | 2 | exc_spikes, inh_spikes |

**Parameters:**
- `n_exc` (int, default=800): Excitatory neurons
- `n_inh` (int, default=200): Inhibitory neurons
- `g` (float, default=5.0): Inhibitory strength ratio (|J_I| = g × J_E)
- `eta` (float, default=2.0): External rate / threshold rate
- Connection probability: ε=0.1 (sparse, from Table 1)
- Synaptic delay: 1.5 ms (all projections)
- J_E = 0.1, J_I = −g × J_E = −0.5 (default)

**Dynamical regimes** (Brunel 2000, Fig. 8):
- g=3, η=2: Asynchronous irregular (AI)
- g=5, η=2: Synchronous irregular (SI) — **default**
- g=6, η=4: Synchronous regular (SR)
- g=8, η=1: Slow oscillation (SO)

**Key property:** The balance between excitation and inhibition (g > 1)
creates irregularity — individual neurons fire irregularly (CV ≈ 1)
despite the network being in a stationary state. This is the hallmark
of balanced network dynamics observed in cortex.

---

### 5. cortical_column

**Reference:** Potjans & Diesmann, Cerebral Cortex 24(3), 2014
**Topology:** 4-layer column (L2/3, L4, L5, L6) with E+I per layer
**Neuron models:** PospischilNeuron (excitatory RS), GolombFSNeuron (inhibitory FS)

| Component | Count | Notes |
|-----------|-------|-------|
| Populations | 8 | 4 layers × 2 (E+I) |
| Projections | 15 | 12 intra-layer (3 per layer) + 3 inter-layer feedforward |
| Stimuli | 1 | PoissonInput(L4_E, rate=800Hz, weight=8.0) thalamic drive |
| Monitors | 8 | one per population |

**Layer sizes (scaled to ~5% of original):**

| Layer | E neurons | I neurons |
|-------|-----------|-----------|
| L2/3 | 50 | 15 |
| L4 | 50 | 15 |
| L5 | 50 | 10 |
| L6 | 50 | 10 |

**Intra-layer connectivity (per layer):**
- E→I: weight=2.0, probability=0.2
- I→E: weight=−3.0, probability=0.2
- E→E: weight=1.0, probability=0.1

**Feedforward (inter-layer):**
- L4_E → L23_E (thalamocortical relay)
- L23_E → L5_E (corticofugal output)
- L5_E → L6_E (deep layer feedback)
- All: weight=1.5, probability=0.1

**Key design:** Thalamic drive enters L4 (the granular layer), propagates
upward to L2/3 (superficial), downward to L5 (output) and L6 (feedback).
This matches the canonical cortical microcircuit.

**Parameters:**
- `n_layers` (int, default=6, effective max=4): Number of layers to include
- PospischilNeuron params: g_m=0.07 (slow M-type current for regular spiking)

---

### 6. central_pattern_generator

**Reference:** Ijspeert, Neural Networks 21(4), 2008, Sec. 3
**Topology:** Half-centre oscillator pairs with inter-pair phase coupling
**Neuron model:** HindmarshRoseNeuron (bursting)

| Component | Count | Notes |
|-----------|-------|-------|
| Populations | 2 × n_oscillators | flexor + extensor per oscillator, 5 neurons each |
| Projections | 3 × n_oscillators | 2 mutual inh + 1 inter-oscillator coupling |
| Stimuli | 2 × n_oscillators | PoissonInput per half-centre (800Hz, weight=5.0) |
| Monitors | 2 × n_oscillators | one per half-centre |

**Parameters:**
- `n_oscillators` (int, default=4): Number of oscillator pairs (4 = quadruped)
- HindmarshRose params: b=3.0, r=0.005, s=4.0 (bursting regime)
- Mutual inhibition: weight=−2.0, probability=0.8 (within pair)
- Inter-oscillator coupling: weight=1.0, probability=0.5 (excitatory, circular)

**How it works:**
1. Each oscillator pair consists of a flexor and extensor half-centre
2. Mutual inhibition (−2.0) ensures alternation: when flexor bursts,
   extensor is suppressed, and vice versa
3. Inter-oscillator coupling (flex[i] → flex[i+1]) creates phase lag
   between adjacent pairs, producing walk/trot/gallop gaits
4. Circular topology: last oscillator couples back to first

**Locomotion gaits** (set by coupling weight and phase lag):
- Walk: π/2 phase lag between adjacent limbs
- Trot: π phase lag between diagonal limbs
- Gallop: near-synchronous fore/hind

---

### 7. decision_making_circuit

**Reference:** Wang, Neuron 36(5), 2002, Fig. 1
**Topology:** Two competing excitatory pools + shared inhibition (attractor)
**Neuron models:** HodgkinHuxleyNeuron (excitatory), WangBuzsakiNeuron (inhibitory)

| Component | Count | Notes |
|-----------|-------|-------|
| Populations | 4 | pool_A, pool_B, nonselective, inhibitory |
| Projections | 9 | A→A, B→B, A→I, B→I, I→A, I→B, NS→A, NS→B, NS→I |
| Stimuli | 3 | PoissonInput for A, B, nonselective |
| Monitors | 2 | pool_A_spikes, pool_B_spikes |

**Parameters:**
- `n_per_pool` (int, default=240): Neurons per selective pool
- Pool A/B drive: 800Hz, weight=15.0
- Nonselective drive: 600Hz, weight=10.0
- n_nonsel = max(10, n_per_pool // 6)
- n_inh = max(15, n_per_pool // 4)

**Connectivity weights:**
- Within-pool recurrent: 3.0 (potentiated NMDA-like)
- Pool→Inhibitory: 2.0
- Inhibitory→Pool: −4.0 (cross-inhibition via shared I pool)
- Nonselective→All: 1.0

**Attractor dynamics:** Potentiated recurrent excitation within each pool
creates bistable attractors. When one pool "wins" (receives slightly more
input), its firing rate increases, which drives the shared inhibitory pool,
which suppresses the competing pool — positive feedback leading to a
categorical decision. This is the neural basis of perceptual decision-making
as modelled by Wang 2002 and Wong & Wang 2006.

---

### 8. working_memory_circuit

**Reference:** Compte et al., J. Neurophysiol. 84(3), 2000
**Topology:** Ring attractor with NMDA-based persistent activity
**Neuron models:** CompteWMNeuron (excitatory), WangBuzsakiNeuron (inhibitory)

| Component | Count | Notes |
|-----------|-------|-------|
| Populations | 2 | excitatory (80%), inhibitory (20%) |
| Projections | 4 | E→E (ring), E→I, I→E, I→I |
| Stimuli | 1 | PoissonInput(n_exc, rate=800Hz, weight=5.0) |
| Monitors | 2 | wm_exc_spikes, wm_inh_spikes |

**Parameters:**
- `n_neurons` (int, default=500): Total neurons (80% E, 20% I)
- CompteWMNeuron params: g_nmda=0.165, g_ampa=0.005, τ_nmda=100ms, Mg²⁺=1.0
- E→E: ring_topology with k nearest neighbours, weight=0.5
- E→I: weight=2.0, probability=0.3
- I→E: weight=−3.0, probability=0.3
- I→I: weight=−2.0, probability=0.2

**Ring topology:** The E→E projection uses distance-dependent connectivity
via `ring_topology(n_exc, k, weight)`. Each neuron connects to its k
nearest neighbours in both directions on the ring. This creates a bump
attractor: a localised pattern of activity that persists after a transient
cue, encoding a remembered spatial location.

**NMDA kinetics:** The slow τ_nmda=100ms time constant is critical for
persistent activity — it provides temporal integration long enough to
sustain the bump without external drive. The Mg²⁺ block voltage-dependence
creates the nonlinearity needed for bistability.

---

### 9. auditory_processing

**Reference:** Goodman & Brette, Front. Neurosci. 4, 2010
**Topology:** Cochlear → onset detection → integration
**Neuron models:** HodgkinHuxleyNeuron (cochlear, integration), WangBuzsakiNeuron (onset)

| Component | Count | Notes |
|-----------|-------|-------|
| Populations | 3 | cochlear (n_channels), onset (n_channels), integration (n_channels//2) |
| Projections | 3 | cochlear→onset, onset→onset (lateral inh), onset→integration |
| Stimuli | 1 | PoissonInput(n_channels, rate=800Hz, weight=15.0) targets cochlear |
| Monitors | 2 | cochlear_spikes, integration_spikes |

**Parameters:**
- `n_channels` (int, default=32): Tonotopic frequency channels
- cochlear→onset: weight=3.0, probability=0.4
- onset→onset: weight=−2.0, probability=0.2 (lateral inhibition)
- onset→integration: weight=3.0, probability=0.3

**Processing pipeline:**
1. **Cochlear layer:** HH neurons model auditory nerve fibre dynamics.
   Each neuron represents one frequency channel of a gammatone filterbank.
2. **Onset layer:** WangBuzsaki fast-spiking neurons with lateral inhibition
   (−2.0) detect sound onsets — only transient changes pass through.
3. **Integration layer:** HH neurons pool across onset channels, creating
   a spectro-temporal representation for downstream decoding.

**Lateral inhibition:** The onset→onset inhibitory connectivity sharpens
frequency selectivity and implements a temporal derivative: constant tones
are suppressed, onsets/offsets are amplified. This matches the function of
bushy cells in the cochlear nucleus.

---

### 10. visual_cortex_v1

**Reference:** Hubel & Wiesel, J. Physiol. 160, 1962; Carandini & Heeger, Nat. Rev. Neurosci. 13, 2012
**Topology:** Orientation-tuned simple cells → complex cells + cross-orientation inhibition
**Neuron models:** HodgkinHuxleyNeuron (simple), WangBuzsakiNeuron (complex)

| Component | Count | Notes |
|-----------|-------|-------|
| Populations | 2 × n_orientation | n_orient simple + n_orient complex |
| Projections | n_orient + n_orient×(n_orient−1) | feedforward + cross-orientation |
| Stimuli | n_orientation | PoissonInput per simple cell population |
| Monitors | 2 × n_orientation | one per population |

**Parameters:**
- `n_orientation` (int, default=8): Number of orientation channels
- `n_per_orientation` (int, default=50): Neurons per simple cell population
- n_complex = max(1, n_per_orientation // 2)
- Simple→complex: weight=3.0, probability=0.5 (feedforward)
- Cross-orientation: weight = −1/(1+dist), probability=0.1

**Cross-orientation inhibition:**
Weight decreases with orientation distance:
- Adjacent (dist=1): w = −1/2 = −0.500
- Two apart (dist=2): w = −1/3 = −0.333
- Three apart (dist=3): w = −1/4 = −0.250
- Furthest (dist=4, for 8 orientations): w = −1/5 = −0.200

This implements the normalization model of Carandini & Heeger (2012):
each orientation suppresses non-preferred orientations proportional to
their similarity, sharpening tuning curves and creating contrast invariance.

**Simple→complex cell transformation:** Simple cells are orientation-tuned
and phase-sensitive. Complex cells pool over phase (via the feedforward
projection) to achieve phase invariance — they respond to their preferred
orientation regardless of the bar's position within their receptive field.

---

## Pretrained Weight Loading

**Module:** `sc_neurocore.model_zoo.pretrained`
**Function:** `load_pretrained(name: str) → Network`

Loads pre-initialised weights from `.npz` files in
`sc_neurocore/model_zoo/weights/`. The architecture is built via the
corresponding builder function, then the Projection CSR data is overwritten
with dense-to-sparse converted weight matrices.

### Available pretrained models

| Name | Builder | Weight file | Projections loaded |
|------|---------|-------------|-------------------|
| `"mnist"` | `mnist_classifier()` | `mnist_784_128_10.npz` | W0 (784→128), W1 (128→10) |
| `"shd"` | `shd_speech_classifier()` | `shd_700_256_20.npz` | W0 (700→256), W_rec (256→256), W1 (256→20) |
| `"dvs_gesture"` | `dvs_gesture_classifier()` | `dvs_256_256_11.npz` | W0 (256→256), W1 (256→11) |

**Weight format:** Dense numpy arrays stored in `.npz`, shape `(fan_in, fan_out)`.
Converted to CSR via `scipy.sparse.csr_matrix` for efficient sparse-matrix
vector multiplication during projection propagation.

Pretrained archives are loaded with numpy pickle support disabled. The
loader rejects missing or unexpected archive members, non-2-D arrays,
shape mismatches, complex or object arrays, and non-finite values before
overwriting projection weights.

The Python loader also validates model-zoo wiring before archive data is
applied: the built network must expose exactly the expected number of
projections, and each projection's source/target population sizes must match
the registered archive matrix shape. These checks fail closed before any CSR
arrays are overwritten, so a stale builder or registry entry cannot partially
mutate a network.

The generated Julia and Mojo mirror files under `src/sc_neurocore/accel/` are
not dispatched pretrained-weight loaders. The supported runtime boundary for
pretrained archives is this Python API, backed by the committed `.npz` files
and the contract tests documented here.

**Spiking correction:** Weights use Xavier/Glorot with spiking correction
factor 0.5 (Zenke & Ganguli 2018). This accounts for the fact that SNN
gradients are noisier than ANN gradients due to the non-differentiable
spike function.

```python
from sc_neurocore.model_zoo.pretrained import load_pretrained

net = load_pretrained("mnist")
net.run(0.1)
print(net.spike_monitors[0].count)
```

**Error handling:**
- Unknown name → `ValueError` with list of available models
- Missing weight file → `FileNotFoundError`
- Registry/schema or projection-layout mismatch → `ValueError` before weight mutation

---

## Infrastructure Pipeline

All 10 architectures produce a `Network` object with the standard pipeline:

```
model_zoo.configs.<builder>()
├── Population(NeuronModel, n, label, params)
│   └── step_all(current) → binary spike vector
├── Projection(source, target, weight, probability, ...)
│   └── propagate(src_spikes) → target_currents (CSR matrix-vector)
├── PoissonInput(n, rate_hz, weight)
│   └── get_current(t, dt) → stochastic current array
├── SpikeMonitor(population, label)
│   ├── .count → total spikes
│   ├── .spike_times → timestep array
│   └── .spike_trains → per-neuron timestep dict
└── Network.run(duration, dt, backend)
    ├── backend="python": pure-Python event loop
    ├── backend="rust": Rust engine (if all models supported)
    └── backend="auto": Rust when available, else Python
```

**Simulation loop (per timestep):**
1. Zero all current buffers
2. Apply stimuli (PoissonInput → target population current)
3. Propagate spikes through projections (CSR matrix-vector)
4. Step all populations (current → spikes)
5. Record spikes in monitors
6. Update plasticity (if any)
7. Apply FIM feedback (if fim_lambda > 0)

**Seed determinism:** All builders use `seed=42`. Two identical calls
produce identical spike counts and spike trains.

---

## Performance

Measured on Intel Xeon E5-2620v4 (ML350 Gen8), Python 3.12, backend="python".

| Architecture | Neurons | Throughput (neuron-steps/s) | Duration tested |
|-------------|---------|---------------------------|-----------------|
| mnist (n_hidden=16) | 810 | ~75K | 50ms |
| dvs (n_classes=4) | 516 | ~70K | 50ms |
| shd | 976 | ~55K | 50ms |
| brunel (50/12) | 62 | ~120K | 50ms |
| cortical (2 layers) | 130 | ~15K | 50ms |
| cpg (2 oscillators) | 20 | ~18K | 50ms |
| decision (n_per_pool=10) | 37 | ~20K | 50ms |
| wm (n_neurons=50) | 50 | ~12K | 50ms |
| auditory (n_ch=8) | 20 | ~25K | 50ms |
| v1 (n_orient=2, n_per=5) | 15 | ~30K | 50ms |

**Bottleneck analysis:**
- Feedforward LIF networks (mnist, dvs, shd) are fastest — simple
  neuron model, no recurrence overhead
- Biophysical models (cortical, cpg, decision, wm) are 3–5× slower due to
  multi-compartment HH/HR dynamics with sub-stepping
- Network size scales linearly: projections dominate at large N

**Rust backend:** Currently not available for model_zoo configs because all
configs use PoissonInput stimuli, and the Rust engine (`_can_use_rust()`)
returns False when stimuli are present. This is a design limitation —
PoissonInput generation happens in Python and cannot be dispatched to Rust.

---

## Neuron Model Summary

| Architecture | Excitatory | Inhibitory |
|-------------|------------|------------|
| mnist | StochasticLIFNeuron | — |
| dvs | StochasticLIFNeuron | — |
| shd | StochasticLIFNeuron | — |
| brunel | StochasticLIFNeuron | StochasticLIFNeuron |
| cortical | PospischilNeuron (RS) | GolombFSNeuron (FS) |
| cpg | HindmarshRoseNeuron | HindmarshRoseNeuron |
| decision | HodgkinHuxleyNeuron | WangBuzsakiNeuron |
| wm | CompteWMNeuron (NMDA) | WangBuzsakiNeuron |
| auditory | HodgkinHuxleyNeuron | WangBuzsakiNeuron |
| v1 | HodgkinHuxleyNeuron | WangBuzsakiNeuron |

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Construction | 10 | Each builder returns Network |
| Topology | 30 | Population counts, projection wiring, monitor counts, stimulus attachment |
| Neuron types | 12 | Correct model class per population (PospischilNeuron, GolombFSNeuron, etc.) |
| Analytical | 15 | Xavier weight scaling, inhibition ratio |J_I|=g·J_E, NMDA params, bursting params, cross-orientation weights |
| Dynamics | 10 | Every config produces spikes under Poisson drive |
| Scaling | 20 | Parameter sweeps (n_hidden, n_classes, n_layers, g, n_oscillators, n_per_pool, n_neurons, n_channels, n_orientation) |
| Performance | 10 | Network throughput > threshold for all configs |
| Pretrained | 8 | Load mnist/shd/dvs, weights differ from default, all produce spikes, unknown name raises ValueError |
| Cross-cutting | 70 | All 10 configs × 7 properties: populations≥2, projections≥1, monitors≥1, stimuli≥1, seed determinism, spike_count analysis, per-neuron ISI |
| **Total** | **201** | |

See `tests/test_model_zoo.py`.

---

## Findings

1. **All 10 configs produce spikes** under default Poisson drive within
   100ms. No silent networks. This confirms that all parameter values
   from the published references are within the spiking regime for the
   implemented neuron models.

2. **Seed determinism holds for all configs.** Two identical calls with
   seed=42 produce identical spike counts. This is critical for
   reproducible experiments and regression testing.

3. **Xavier scaling verified analytically.** The mnist_classifier uses
   w = √(2/fan_in) × 20 for both projections. This matches Zenke &
   Ganguli 2018 Table 1 (with the ×20 scaling factor for spiking networks).

4. **Brunel inhibition ratio exact.** |J_I| = g × J_E to machine
   precision for all tested g values (3.0, 5.0, 8.0).

5. **Cross-orientation inhibition distance-dependent.** Measured weights
   confirm w = −1/(1+d) formula: closer orientations receive stronger
   suppression, matching the normalization model.

6. **Recurrent tau differentiation in SHD.** The recurrent layer uses
   τ_mem=20ms vs input τ_mem=10ms, verified by comparing neuron
   parameters. This longer time constant provides the temporal integration
   necessary for speech processing.

7. **Working memory ring is self-recurrent.** The E→E projection source
   and target are the same population, confirming ring attractor topology.

8. **CPG mutual inhibition verified.** Flexor→extensor and extensor→flexor
   both carry weight=−2.0, ensuring alternating burst dynamics.

9. **Pretrained weights differ from default.** The loaded .npz weights
   are not identical to the default Xavier initialisation, confirming
   that the weight loading pipeline successfully overwrites the defaults.

10. **Rust backend unavailable for all configs.** Every config uses
    PoissonInput, which disables the Rust backend. This is a known
    limitation — the Rust engine does not support stimulus injection.
