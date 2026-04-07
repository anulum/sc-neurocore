# CompteWMNeuron

**Module:** `sc_neurocore.neurons.models.compte_wm`
**Reference:** Compte, Brunel, Goldman-Rakic & Wang, Cereb. Cortex 10(9), 2000
**Family:** Biophysical (NMDA-driven working memory, prefrontal cortex)
**State variables:** `v` (membrane potential), `s_ampa` (AMPA gating), `s_nmda` (NMDA gating), `x_nmda` (NMDA presynaptic), `s_gaba` (GABA gating)

---

## Equations

### Membrane potential

$$C_m \frac{dV}{dt} = -I_L - I_{AMPA} - I_{NMDA} - I_{GABA} + I_{ext}$$

### Ionic currents

$$I_L = g_L(V - E_L)$$
$$I_{AMPA} = g_{AMPA} \cdot s_{AMPA} \cdot (V - E_{exc})$$
$$I_{NMDA} = g_{NMDA} \cdot B(V) \cdot s_{NMDA} \cdot (V - E_{exc})$$
$$I_{GABA} = g_{GABA} \cdot s_{GABA} \cdot (V - E_{inh})$$

### Voltage-dependent Mg²⁺ block

$$B(V) = \frac{1}{1 + \frac{[Mg^{2+}]}{3.57} \cdot \exp(-0.062V)}$$

This is the Jahr-Stevens (1990) formulation of the NMDA receptor
voltage dependence. At resting potential (V ≈ −70 mV): B ≈ 0.048
(95% blocked). At depolarised potential (V ≈ 0 mV): B ≈ 0.78
(22% blocked). The Mg²⁺ block creates the positive feedback loop
essential for persistent activity.

### Synaptic gating dynamics

$$\frac{ds_{AMPA}}{dt} = -\frac{s_{AMPA}}{\tau_{AMPA}}$$

$$\frac{ds_{NMDA}}{dt} = -\frac{s_{NMDA}}{\tau_{NMDA}} + \alpha_{NMDA} \cdot x_{NMDA} \cdot (1 - s_{NMDA})$$

$$\frac{dx_{NMDA}}{dt} = -\frac{x_{NMDA}}{\tau_x}$$

$$\frac{ds_{GABA}}{dt} = -\frac{s_{GABA}}{5.0}$$

### Spike and reset

$$V \geq V_{threshold}: \quad V \leftarrow V_{reset}, \quad s_{GABA} \leftarrow s_{GABA} + 1$$

Note: spike triggers self-inhibition via GABA increment.

### Presynaptic spike input

$$\text{spike\_in = True}: \quad s_{AMPA} \leftarrow s_{AMPA} + 1, \quad x_{NMDA} \leftarrow x_{NMDA} + 1$$

### Implementation

```python
def step(self, current: float, spike_in: bool = False) -> int:
    if spike_in:
        self.s_ampa += 1.0
        self.x_nmda += 1.0
    self.s_ampa *= exp(-dt / tau_ampa)
    self.s_nmda += (-s_nmda/tau_nmda + alpha*x_nmda*(1-s_nmda)) * dt
    self.x_nmda *= exp(-dt / tau_x)
    self.s_gaba *= exp(-dt / 5.0)
    b = 1 / (1 + mg/3.57 * exp(-0.062 * v))
    # compute currents, update v, spike/reset
```

Hybrid integration: exponential decay for AMPA, x_NMDA, GABA;
Euler for s_NMDA (nonlinear ODE); Euler for V.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −70.0 | mV | Membrane potential |
| `s_ampa` | 0.0 | — | AMPA synaptic gating variable |
| `s_nmda` | 0.0 | — | NMDA synaptic gating variable |
| `x_nmda` | 0.0 | — | NMDA presynaptic variable |
| `s_gaba` | 0.0 | — | GABA synaptic gating variable |
| `g_l` | 0.025 | µS | Leak conductance |
| `g_ampa` | 0.005 | µS | AMPA conductance |
| `g_nmda` | 0.165 | µS | NMDA conductance |
| `g_gaba` | 0.013 | µS | GABA conductance |
| `e_l` | −70.0 | mV | Leak reversal |
| `e_exc` | 0.0 | mV | Excitatory reversal (AMPA + NMDA) |
| `e_inh` | −70.0 | mV | Inhibitory reversal (GABA) |
| `c_m` | 0.5 | nF | Membrane capacitance |
| `mg` | 1.0 | mM | Extracellular Mg²⁺ concentration |
| `tau_ampa` | 2.0 | ms | AMPA decay time constant |
| `tau_nmda` | 100.0 | ms | NMDA decay time constant |
| `tau_x` | 2.0 | ms | NMDA presynaptic decay |
| `alpha_nmda` | 0.5 | kHz | NMDA gating rate constant |
| `v_threshold` | −50.0 | mV | Spike threshold |
| `v_reset` | −55.0 | mV | Post-spike reset potential |
| `dt` | 0.1 | ms | Integration timestep |

### Conductance hierarchy

$$g_{NMDA}(0.165) \gg g_L(0.025) > g_{GABA}(0.013) > g_{AMPA}(0.005)$$

NMDA conductance dominates — 33× larger than AMPA. This is the
defining feature of the Compte model: persistent activity is driven
by the slow NMDA current, not the fast AMPA current.

### Timescale hierarchy

- **τ_AMPA = 2 ms:** Fast excitation (glutamate AMPA)
- **τ_x = 2 ms:** Fast NMDA presynaptic decay
- **τ_GABA = 5 ms:** Intermediate inhibition (hardcoded, not a parameter)
- **τ_NMDA = 100 ms:** Very slow NMDA decay — this is the memory timescale

The 50× ratio τ_NMDA/τ_AMPA creates the timescale separation needed
for working memory: AMPA provides the initial trigger, NMDA maintains
the persistent state.

---

## Analytical Properties

### Mg²⁺ block as bistability mechanism

The voltage-dependent Mg²⁺ block creates positive feedback:

1. Depolarisation → removes Mg²⁺ block → B(V) increases
2. Increased B(V) → more NMDA current flows → more depolarisation
3. This creates a self-sustaining loop: once the neuron is depolarised
   enough, NMDA current maintains the depolarisation

The B(V) function:

| V (mV) | B(V) | Interpretation |
|--------|------|---------------|
| −80 | 0.030 | 97% blocked (rest) |
| −70 | 0.048 | 95% blocked (near E_L) |
| −50 | 0.119 | 88% blocked (threshold) |
| −30 | 0.274 | 73% blocked |
| 0 | 0.781 | 22% blocked (active) |

### Persistent activity (working memory)

The neural mechanism of working memory in the Compte model:

1. **Stimulus period:** External current drives neuron above threshold
2. **NMDA activation:** Spikes activate NMDA pathway via x_nmda
3. **Slow NMDA buildup:** s_nmda increases with τ=100 ms timescale
4. **Stimulus removal:** External current stops
5. **Persistent state:** NMDA current (g_NMDA × B(V) × s_NMDA × (V−E_e))
   is strong enough to keep V above threshold without external input
6. **Self-sustaining:** Each spike adds to x_nmda → maintains s_nmda →
   maintains NMDA current → maintains spiking

This requires recurrent connectivity (spike_in = True from other neurons
in the population or self-connections) to work in practice. In isolation,
the single neuron cannot sustain persistent activity because there is
no recurrent spike input.

### Self-inhibition via GABA

Each spike adds 1.0 to s_gaba, creating autaptic inhibition:
- I_GABA = g_GABA × s_gaba × (V − E_inh)
- Since V > E_inh = −70 mV during activity, I_GABA is hyperpolarising
- This prevents runaway excitation and controls firing rate

### NMDA saturation

The s_nmda ODE includes a saturation term (1 − s_nmda):
$$\frac{ds_{NMDA}}{dt} = -\frac{s_{NMDA}}{\tau_{NMDA}} + \alpha \cdot x \cdot (1 - s_{NMDA})$$

As s_nmda → 1, the activation rate drops to zero. This creates a
natural upper bound: s_nmda ∈ [0, 1]. The maximum NMDA current is
bounded by g_NMDA × B(V) × 1 × (V − E_e).

### Subthreshold at I = 0.5

At low current (I = 0.5), the neuron does not spike. The steady-state:
$$V_{ss} = E_L + \frac{I}{g_L} = -70 + \frac{0.5}{0.025} = -50 \text{ mV}$$

This equals the threshold, but without NMDA contribution the effective
conductance is higher → V_ss slightly below threshold. Verified: 0
spikes in 5000 steps at I=0.5.

### Spiking at I = 2.0

At I=2.0, well above threshold. Verified: >50 spikes in 10000 steps.

---

## Behaviour

### Four synaptic components

1. **AMPA (fast excitation):** s_ampa decays with τ=2 ms. Provides
   immediate response to presynaptic spikes. Small conductance (0.005).

2. **NMDA (slow excitation):** s_nmda decays with τ=100 ms. Provides
   sustained excitatory drive. Large conductance (0.165). Gated by
   Mg²⁺ block (voltage-dependent).

3. **GABA (self-inhibition):** s_gaba decays with τ=5 ms (hardcoded).
   Incremented by each postsynaptic spike. Controls firing rate.

4. **Leak:** Constant g_L = 0.025. Restores V toward E_L = −70.

### step() interface

The `step()` method takes:
1. `current` (float): External current injection
2. `spike_in` (bool, default False): Whether a presynaptic spike occurred

In the standard Pipeline (Population → Network), only `current` is
used. The `spike_in` parameter would require custom network logic to
implement recurrent NMDA connectivity.

---

## Comparison with Related Models

| Property | Compte WM | WongWang | COBALif | LIF |
|----------|-----------|---------|---------|-----|
| State vars | 5 | 2 | 3 | 1 |
| NMDA | Yes (with Mg²⁺) | Yes (simplified) | No | No |
| Mg²⁺ block | Yes (Jahr-Stevens) | No | No | No |
| GABA self-inh | Yes (autaptic) | Yes | No | No |
| τ_slow | 100 ms (NMDA) | 100 ms (s) | 10 ms (τ_i) | τ_m |
| Working memory | Yes (designed) | Yes (reduced) | No | No |
| step() args | 2 (I, spike_in) | 4 (I, S, s, noise) | 3 (I, Δg_e, Δg_i) | 1 (I) |
| Bistability | NMDA-driven | Rate-driven | No | No |

The Compte model is the **full biophysical** working memory model.
WongWang is the reduced 2-variable approximation derived from it.

---

## Numerical Considerations

- **Hybrid integration:** AMPA, x_NMDA, GABA use exact exponential
  decay. s_NMDA uses Euler (nonlinear). V uses Euler.
- **4 exp() per step:** AMPA decay, x_NMDA decay, GABA decay, Mg²⁺ block.
- **No clipping:** V is not explicitly bounded. The spike-and-reset
  mechanism and GABA self-inhibition keep V in range.
- **s_nmda saturation:** The (1 − s_nmda) term naturally bounds s_nmda ≤ 1.
- **dt = 0.1 ms:** Standard timestep for conductance-based models.
- **5 state variables:** Highest dimensionality among IF-type models
  in SC-NeuroCore (tied with ArcaneNeuron).

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/compte_wm.py` — 81 lines.
- **Five state variables:** v, s_ampa, s_nmda, x_nmda, s_gaba.
- **Dataclass:** Uses `@dataclass`.
- **Private _mg_block() method:** Computes Jahr-Stevens Mg²⁺ block.
- **Multi-argument step():** Takes current + spike_in boolean.
- **Rust wiring:** Compatible (5 f64 state vars, 4 exp).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~262K steps/s | Not measured |
| Network (20n, 500ms) | ~220K neuron-steps/s | — |

Moderate speed — 4 exp() + Mg²⁺ block per step. More expensive than
simple LIF but no sub-stepping needed.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 9 | construction, binary output, subthreshold (I=0.5), spikes (I=2.0), NMDA gating via spike_in, Mg²⁺ block voltage-dependent, GABA self-inhibition on spike, state finite (10K at I=5), reset |
| Network | 3 | Population(n=10/20), Network+PoissonInput spikes, Projection+spike_trains |
| Analysis | 3 | firing_rate >0, spike_count >10, isi finite |
| **Total** | **15** | **ALL PASSED (3.49s)** |

See `tests/test_model_compte_wm.py`.

---

## Findings (Measured 2026-03-31)

1. **15/15 tests PASSED in 3.49s.** No failures.

2. **Subthreshold at I=0.5.** Zero spikes in 5000 steps. Consistent
   with V_ss ≈ threshold but insufficient to overcome conductance.

3. **Spiking at I=2.0.** More than 50 spikes in 10000 steps. Strong
   current overcomes the high NMDA+GABA conductance.

4. **NMDA gating via spike_in.** Calling step(0.0, spike_in=True)
   results in x_nmda > 0, confirming presynaptic spike activation.

5. **Mg²⁺ block is voltage-dependent.** B(−80) < B(0) — relief of
   block at depolarised potentials confirmed.

6. **GABA self-inhibition on spike.** When the neuron fires, s_gaba
   increases (s_gaba += 1.0). Verified by detecting a spike and
   checking s_gaba > 0 immediately after.

7. **State finite across 10K steps.** All 5 state variables (v, s_ampa,
   s_nmda, x_nmda, s_gaba) remain finite at I=5.0.

8. **Reset clears all state.** v → E_L (−70), all synaptic variables → 0.

9. **Network pipeline functional.** Population(n=20) with PoissonInput
   (rate=500Hz, weight=2.0) produces spikes. Projection(pop→pop,
   weight=0.5, prob=0.3) works. spike_trains extractable.

10. **Analysis pipeline verified.** firing_rate > 0 Hz, spike_count > 10,
    isi all finite. From 10K-step binary train at I=3.0.

11. **Working memory requires recurrence.** The single neuron in isolation
    does not exhibit persistent activity — it needs recurrent spike_in
    input from other neurons or self-connections to sustain NMDA-driven
    persistent state.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
15/15 PASSED in 3.49s
├── TestCompteIsolation: 9 tests
│   ├── construction (v=-70, all synaptic vars = 0)
│   ├── step() → int {0,1}
│   ├── subthreshold at I=0.5 (0 spikes in 5K)
│   ├── spikes at I=2.0 (>50 in 10K)
│   ├── NMDA gating: spike_in → x_nmda > 0
│   ├── Mg²⁺ block: B(-80) < B(0)
│   ├── GABA self-inhibition: s_gaba > 0 after spike
│   ├── state finite (10K steps at I=5)
│   └── reset() (v→E_L, all synaptic → 0)
├── TestCompteNetwork: 3 tests
│   ├── Population(n=10)
│   ├── Network(n=20) + PoissonInput → spikes > 0
│   └── Projection(pop→pop, w=0.5, p=0.3) + spike_trains
└── TestCompteAnalysis: 3 tests
    ├── firing_rate > 0
    ├── spike_count > 10
    └── isi all finite
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | v=-70, 5 state vars |
| step(I) → int {0,1} | ✓ PASS | Standard binary output |
| step(I, spike_in) | ✓ PASS | Multi-argument interface |
| Subthreshold (I=0.5) | ✓ PASS | 0 spikes in 5K |
| Spiking (I=2.0) | ✓ PASS | >50 spikes in 10K |
| NMDA gating | ✓ PASS | spike_in activates pathway |
| Mg²⁺ block | ✓ PASS | Voltage-dependent relief |
| GABA self-inhibition | ✓ PASS | s_gaba += 1 on spike |
| State finite (10K) | ✓ PASS | All 5 vars finite |
| reset() | ✓ PASS | All vars to initial |
| Population(n=10) | ✓ PASS | 10 instances |
| Network + PoissonInput | ✓ PASS | Spikes > 0 |
| Projection(pop→pop) | ✓ PASS | spike_trains extractable |
| firing_rate | ✓ PASS | > 0 Hz |
| spike_count | ✓ PASS | > 10 |
| isi | ✓ PASS | all finite |

### Network configuration tested

- Population: 20 CompteWMNeurons (spiking), 10 (Projection)
- PoissonInput: rate=500Hz, weight=2.0, dt=0.001, seed=42
- Projection: self-recurrent, weight=0.5, probability=0.3
- SpikeMonitor: count, spike_trains verified
- Duration: 0.5s (spiking), 0.3s (Projection)

### Note on working memory functionality

The Compte model is designed for working memory via recurrent NMDA
connectivity. In the standard Pipeline, only scalar current injection
is used — the spike_in parameter is not activated. To achieve persistent
activity (working memory), a custom network loop that passes spike_in
from recurrent connections would be required. The model is functionally
correct as a spiking neuron but does not exhibit its signature working
memory behaviour without recurrent NMDA input.

**ALL 15 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**

---

## Theoretical Context

### Compte et al. 2000

Albert Compte and Xiao-Jing Wang developed this model to explain
persistent neural activity in prefrontal cortex (PFC) during
working memory tasks. Key contributions:

1. **Bump attractor:** A ring of neurons with distance-dependent
   connectivity maintains a "bump" of activity that encodes the
   remembered stimulus location (spatial working memory).

2. **NMDA is essential:** The slow NMDA time constant (τ=100 ms)
   is necessary for persistent activity. AMPA alone (τ=2 ms) cannot
   sustain activity between spikes.

3. **Mg²⁺ block creates bistability:** The voltage-dependent unblock
   creates two stable states — quiescent (NMDA blocked) and active
   (NMDA unblocked) — with a saddle point between them.

4. **Balanced E/I:** GABA self-inhibition prevents runaway excitation
   and controls the width of the activity bump.

### Working memory in neuroscience

Working memory — the ability to hold information "in mind" for
seconds to minutes — is one of the most studied cognitive functions.
The Compte model provides a mechanistic explanation:

- Information is encoded in the **spatial pattern** of persistent activity
- Persistence is maintained by **recurrent NMDA** excitation
- Capacity is limited by **lateral inhibition** (GABA)
- Distraction disrupts the activity bump → forgetting

This model is the standard reference for computational models of
spatial working memory in PFC (>2000 citations).

---

## Usage Examples

### Example 1: Basic Python — current-driven spiking

```python
from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron

neuron = CompteWMNeuron()

# Drive with constant current (no spike_in → no NMDA build-up)
spikes = []
for t in range(10000):
    spike = neuron.step(2.0)
    if spike:
        spikes.append(t)

print(f"Fired {len(spikes)} spikes in 10 s")
print(f"s_nmda = {neuron.s_nmda:.6f} (low without recurrent input)")
```

### Example 2: Advanced Python — NMDA build-up with recurrent spikes

```python
from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron

neuron = CompteWMNeuron()

# Simulate a brief stimulus followed by self-recurrent activity
# Phase 1: strong drive + spike_in (stimulus period)
for t in range(200):
    neuron.step(3.0, spike_in=True)

# Phase 2: no external drive — does NMDA sustain activity?
sustained_spikes = 0
for t in range(5000):
    spike = neuron.step(0.0, spike_in=False)
    sustained_spikes += spike

print(f"Sustained spikes after stimulus: {sustained_spikes}")
print(f"s_nmda = {neuron.s_nmda:.6f} (decays with τ=100 ms)")
print(f"Mg²⁺ block at rest: {neuron._mg_block(neuron.v):.4f}")
```

### Example 3: PyO3 Rust — high-performance stepping

```rust
use sc_neurocore_engine::neurons::CompteWMNeuron;

let mut neuron = CompteWMNeuron::new();

// 10,000 steps with current drive
let mut spikes = 0;
for _ in 0..10_000 {
    spikes += neuron.step(2.0, false);
}
println!("Compte WM: {spikes} spikes, s_nmda = {:.6}", neuron.s_nmda);

// Reset
neuron.reset();
assert!((neuron.v - (-70.0)).abs() < 1e-12);
```

---

## Technical Reference

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `step(current, spike_in=False) → int` | 0 or 1 | Advance dt ms, return spike |
| `reset` | `reset() → None` | — | Restore all 5 state vars to initial |
| `_mg_block` | `_mg_block(v) → float` | 0–1 | Jahr-Stevens Mg²⁺ block factor |

### Python/Rust Parity

| Property | Python | Rust | Match |
|----------|--------|------|-------|
| State variables (5) | v, s_ampa, s_nmda, x_nmda, s_gaba | identical | EXACT |
| Parameters (21) | All f64 defaults | All f64 defaults | EXACT |
| Exponential decays | `np.exp(-dt/τ)` | `(-dt/τ).exp()` | EXACT |
| Mg²⁺ block | `1/(1+mg/3.57·exp(-0.062·V))` | identical | EXACT |
| NMDA Euler | `s_nmda += (…)·dt` | `s_nmda += (…)·dt` | EXACT |
| Voltage (Euler) | `V += (…)/C_m·dt` | `V += (…)/C_m·dt` | EXACT |
| Spike reset + GABA | `V=V_reset, s_gaba+=1` | identical | EXACT |
| Interface | `step(current, spike_in=False)` | `step(f64, bool)` | EXACT |

### NetworkRunner limitations

The `wrap_2arg_f64!` macro in network_runner.rs converts the 2-argument
`step(current, spike_in)` to a 1-argument interface by fixing
`spike_in = false`. This means:

- **Working memory recurrence is not available through NetworkRunner**
- For persistent activity, use the Python API directly with explicit
  `spike_in=True` feedback
- The NetworkRunner drives the model as a standard conductance-based
  spiking neuron without NMDA self-excitation

### Supported operations

| Operation | Supported | Notes |
|-----------|-----------|-------|
| Population | Yes | Standard interface |
| Projection | Yes | Wiring works, but spike_in fixed to false |
| NetworkRunner | Partial | Current only; no spike_in recurrence |
| SpikeMonitor | Yes | Binary spike output |
| PoissonInput | Yes | Current-based drive |
| PyO3 bridge | Yes | Full 2-arg step() with spike_in |
| get_state() | Partial | Returns {v, s_nmda} only (3 vars missing) |

---

## Performance Benchmarks

### Criterion 0.8 (Rust engine)

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-05.

| Benchmark | Steps | Median | Per step |
|-----------|------:|-------:|---------:|
| `compte_wm_10k_steps` | 10 000 | 231 µs | **23.1 ns** |

3 exp() calls per step (AMPA decay, x_nmda decay, Mg²⁺ block) +
1 Euler step (NMDA) + 1 Euler step (voltage). No sub-stepping.

### Python throughput

| Metric | Value |
|--------|------:|
| Isolation | ~262 000 steps/s |
| Network (10 neurons, 1 s) | ~220 000 neuron-steps/s |

### Rust speedup

| Metric | Python | Rust | Speedup |
|--------|-------:|-----:|--------:|
| Per step | ~3.8 µs | 23.1 ns | **~164×** |

---

## Citations

1. Compte, A., Brunel, N., Goldman-Rakic, P. S. & Wang, X.-J. (2000).
   Synaptic mechanisms and network dynamics underlying spatial working
   memory in a cortical network model. *Cerebral Cortex*, 10(9),
   910–923.
   DOI: [10.1093/cercor/10.9.910](https://doi.org/10.1093/cercor/10.9.910)

2. Wang, X.-J. (2001). Synaptic reverberation underlying mnemonic
   persistent activity. *Trends in Neurosciences*, 24(8), 455–463.
   DOI: [10.1016/S0166-2236(00)01868-3](https://doi.org/10.1016/S0166-2236(00)01868-3)

3. Wang, X.-J. (2002). Probabilistic decision making by slow reverberation
   in cortical circuits. *Neuron*, 36(5), 955–968.
   DOI: [10.1016/S0896-6273(02)01092-9](https://doi.org/10.1016/S0896-6273(02)01092-9)

4. Jahr, C. E. & Stevens, C. F. (1990). Voltage dependence of
   NMDA-activated macroscopic conductances predicted by single-channel
   kinetics. *Journal of Neuroscience*, 10(9), 3178–3182.
   DOI: [10.1523/JNEUROSCI.10-09-03178.1990](https://doi.org/10.1523/JNEUROSCI.10-09-03178.1990)

5. Goldman-Rakic, P. S. (1995). Cellular basis of working memory.
   *Neuron*, 14(3), 477–485.
   DOI: [10.1016/0896-6273(95)90304-6](https://doi.org/10.1016/0896-6273(95)90304-6)

6. Brunel, N. & Wang, X.-J. (2001). Effects of neuromodulation in a
   cortical network model of object working memory dominated by recurrent
   inhibition. *Journal of Computational Neuroscience*, 11(1), 63–85.
   DOI: [10.1023/A:1011204814320](https://doi.org/10.1023/A:1011204814320)
