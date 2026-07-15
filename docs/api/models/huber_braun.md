# HuberBraunNeuron

**Module:** `sc_neurocore.neurons.models.huber_braun`
**Reference:** Braun, H.A., Huber, M.T. et al., Int. J. Bifurcation Chaos 8(4):881, 1998
**Family:** Conductance-based (cold receptor, temperature-dependent)
**State variables:** `v` (membrane potential), `a_sd` (slow depolarising activation), `a_sr` (slow repolarising activation)

---

## Equations

### Membrane potential

$$C_m \frac{dV}{dt} = -I_{sd} - I_{sr} - I_L + I_{ext}$$

where the three ionic currents are:

$$I_{sd} = g_{sd} \cdot a_{sd} \cdot (V - E_{sd})$$
$$I_{sr} = g_{sr} \cdot a_{sr} \cdot (V - E_{sr})$$
$$I_L = g_L \cdot (V - E_L)$$

### Gating variables

$$\tau_{sd} \frac{da_{sd}}{dt} = \sigma_{sd}(V) - a_{sd}$$
$$\tau_{sr} \frac{da_{sr}}{dt} = \sigma_{sr}(V) - a_{sr}$$

### Activation functions (sigmoidal)

$$\sigma_{sd}(V) = \frac{1}{1 + \exp(-(V + 25)/5)}$$
$$\sigma_{sr}(V) = \frac{1}{1 + \exp((V + 40)/5)}$$

Note: $\sigma_{sd}$ activates with depolarisation (V1/2 = −25 mV),
$\sigma_{sr}$ activates with hyperpolarisation (V1/2 = −40 mV). This
creates the slow oscillatory cycle:

1. $a_{sd}$ depolarises the membrane → V rises
2. At depolarised V, $a_{sr}$ activates → repolarisation
3. At hyperpolarised V, $a_{sr}$ deactivates, $a_{sd}$ deactivates
4. Leak drives V back toward $E_L$ → cycle restarts

### Spike detection

Threshold crossing: spike when $V \geq V_{threshold}$ and $V_{prev} < V_{threshold}$.

### Implementation

Rust: single Euler step (no sub-stepping). C_m = 1 (implicit).

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −50.0 | mV | Membrane potential |
| `a_sd` | 0.0 | — | Slow depolarising activation |
| `a_sr` | 0.0 | — | Slow repolarising activation |
| `g_sd` | 1.5 | mS/cm² | Slow depolarising conductance |
| `g_sr` | 0.4 | mS/cm² | Slow repolarising conductance |
| `g_l` | 0.1 | mS/cm² | Leak conductance |
| `e_sd` | 50.0 | mV | SD reversal (Na⁺-like, excitatory) |
| `e_sr` | −90.0 | mV | SR reversal (K⁺-like, inhibitory) |
| `e_l` | −60.0 | mV | Leak reversal |
| `tau_sd` | 10.0 | ms | SD time constant |
| `tau_sr` | 20.0 | ms | SR time constant |
| `eta` | 0.012 | mV/√ms | Noise amplitude (not used in Rust) |
| `dt` | 0.1 | ms | Integration timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

### Conductance hierarchy

$$g_{sd} (1.5) \gg g_{sr} (0.4) > g_L (0.1)$$

The SD conductance dominates, making the model prone to depolarisation
block. Temperature tuning in the original paper adjusts this ratio.

### Reversal potential separation

$$E_{sd} - E_{sr} = 140 \text{ mV}$$

This large reversal spread creates the oscillatory driving force.

---

## Analytical Properties

### Time constant ratio

$$\frac{\tau_{sr}}{\tau_{sd}} = 2.0$$

The slow repolarising current has twice the time constant of the
depolarising one. This asymmetry determines:
- Burst duration (dominated by $\tau_{sd}$ = 10 ms)
- Inter-burst interval (dominated by $\tau_{sr}$ = 20 ms)

### Steady-state activation at rest (V = −50 mV)

$$\sigma_{sd}(-50) = \frac{1}{1 + e^{5}} = 0.0067$$
$$\sigma_{sr}(-50) = \frac{1}{1 + e^{-2}} = 0.881$$

At rest, SR is strongly active (repolarising) while SD is nearly off.
External current must overcome this SR bias to trigger depolarisation.

### Bifurcation structure

The model exhibits a rich bifurcation structure (Braun et al. 1998):
- **Tonic spiking** at high $g_{sd}/g_{sr}$ ratio
- **Bursting** at intermediate ratios
- **Subthreshold oscillations** at low ratios
- **Stochastic resonance** when $\eta > 0$ and parameters near bifurcation

Temperature maps to conductance scaling in the original paper: lower
temperature → higher $g_{sd}$ → more excitable.

---

## Behaviour

- **Cold receptor:** Models temperature-sensitive neurons in dorsal root
  ganglia. Oscillation regime depends on temperature (mapped to
  conductance ratios).
- **Default parameters:** Produce an initial transient depolarisation then
  settle to a depolarised equilibrium (~+46 mV) due to strong $g_{sd}$.
  Sustained oscillation requires parameter tuning.
- **Stochastic resonance:** Noise ($\eta > 0$) near bifurcation boundaries
  can enhance signal detection — a hallmark of cold receptors.
- **Simplified model:** Lacks fast Na⁺ inactivation present in full HH
  models. The "spikes" are slow depolarisation events, not fast action
  potentials.

---

## Comparison with Related Models

| Property | HuberBraun | Prescott | FitzHughNagumo | HodgkinHuxley |
|----------|-----------|---------|----------------|---------------|
| Variables | 3 (V, a_sd, a_sr) | 2 (V, w) | 2 (v, w) | 4 (V, m, h, n) |
| Temperature | Yes (conductance scaling) | No | No | Implicit |
| Noise | Built-in (η) | No | No | No |
| Spike type | Slow depolarisation | Fast AP | Relaxation | Fast AP |
| Currents | SD + SR + leak | Fast + slow + leak | Cubic + linear | Na + K + leak |
| Bifurcation | Rich (Braun 1998) | Type I/II/III | Hopf | Hopf |

---

## Performance

| Metric | Python | Rust (Criterion) |
|--------|--------|-----------------|
| Isolation | ~73K steps/s | 14.0M steps/s (71.4 ns/step) |
| 1k steps | — | 71.4 µs |
| Network | Standard (single-current) | NeuronVariant::HuberBraun |

Rust is ~192× faster than Python. Single Euler step, 2 sigmoid evaluations
(exp()), 3 conductance-current products per step.
Measured 2026-04-05 on i5-11600K @ 3.90 GHz, Criterion 0.8.

---

## Numerical Considerations

- **Single Euler step:** dt = 0.1 ms. Adequate for tau_sd = 10 ms
  (dt/tau = 0.01) and tau_sr = 20 ms (dt/tau = 0.005).
- **No sub-stepping:** The slow dynamics (no fast Na gating) make this
  model stable with large timesteps.
- **Sigmoid overflow:** At extreme V, exp() argument stays bounded
  because V is clamped by conductance-based negative feedback.
- **Noise:** The η parameter is defined but not applied in the Rust
  implementation (Python applies Gaussian noise). Rust is deterministic.

---

## Implementation Notes

- **Source (Rust):** `engine/src/neurons/biophysical/huber_braun.rs`
- **Source (Python):** `src/sc_neurocore/neurons/models/huber_braun.py`
- **State:** 3 variables (v, a_sd, a_sr) + parameters
- **Spike detection:** Threshold crossing at V = −20 mV
- **Rust wiring:** `NeuronVariant::HuberBraun` in network_runner.rs

---

## Pipeline Compatibility

### Standard interface

`step(current: f64) -> i32` — fully compatible with Network pipeline.

### Population compatible

`Population(HuberBraunNeuron, n=10)` works for construction.

---

## Test Coverage

### Python tests (28 total)

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 16 | defaults, step binary, finite long run, reset, noise present, sd_inf/sr_inf complementary, sd_inf midpoint, sd activates depolarised, sr activates hyperpolarised, three currents, reversal ordering, noise amplitude, sd slower than sr, gating bounded, fires under drive, rate increases |
| Parametric | 4 | f-I sweep, g_sd sweep, eta noise sweep, dt stability |
| Throughput | 2 | isolation throughput, network throughput |
| Pipeline | 3 | Population, Projection wiring, Network spikes |
| Analysis | 3 | spike_count, ISI, firing_rate |
| **Total** | **28** | |

### Rust tests

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Fires | 1 | fires with I=10.0 in 5000 steps |
| Zero input | 1 | stable at zero input |
| Reset | 1 | v→−50, a_sd→0, a_sr→0 |
| Extreme | 1 | finite after 200 steps at I=10⁴ |
| Negative | 1 | finite after 500 steps at I=−10 |
| NaN | 1 | no panic on NaN input |
| **Total** | **6** | |

---

## Findings

1. **Throughput:** 71.4 ns/step (Rust), ~73K steps/s (Python). Rust is
   192× faster.

2. **Default regime:** Initial transient spike then depolarisation block.
   The strong $g_{sd}$ = 1.5 drives V toward $E_{sd}$ = 50 mV. Sustained
   spiking requires reducing $g_{sd}$ or increasing $g_{sr}$.

3. **Temperature sensitivity:** Not directly parameterised as temperature.
   The original paper maps temperature to conductance scaling. Users must
   manually adjust $g_{sd}$, $g_{sr}$ for different temperature regimes.

4. **Noise absent in Rust:** The $\eta$ parameter exists but is not applied
   in the Rust step() function. The Python implementation includes Gaussian
   noise. This is a known limitation for Rust simulations requiring
   stochastic resonance.

5. **No fast spiking mechanism:** Unlike HH or WangBuzsaki, this model
   has no fast Na⁺ activation/inactivation. "Spikes" are slow
   depolarisation threshold crossings, not fast action potentials with
   ~1 ms duration.

6. **Sigmoid half-activation:** SD at −25 mV (depolarising), SR at −40 mV
   (hyperpolarising). Both use k = 5 mV slope.

7. **Pipeline verified:** All stages pass — construction, step, Population,
   Network, Rust parity.

---

## Theoretical Context

### Cold receptor physiology

Cold receptors in mammalian skin are free nerve endings of Aδ and C
fibres that respond to temperatures below ~35°C. Their firing pattern
encodes absolute temperature (static response) and temperature change
(dynamic response). Key features:

- **Paradoxical cold:** Some C-fibre cold receptors fire at temperatures
  above 45°C ("paradoxical cold sensation")
- **Burst patterns:** Cold receptors typically fire bursts of spikes
  whose inter-burst interval encodes temperature
- **Menthol sensitisation:** TRP channels (TRPM8) set the activation
  threshold, but the oscillatory mechanism is ionic

The Huber-Braun model captures the burst/tonic transition with just
two slow conductances, making it analytically tractable.

### Stochastic resonance

Braun et al. (1998) showed that adding noise ($\eta$) near a subcritical
Hopf bifurcation enhances the coherence of oscillations — a phenomenon
called **stochastic resonance**. This is physiologically relevant: thermal
noise in ion channels near threshold helps cold receptors detect small
temperature changes.

The optimal noise amplitude is:
$$\eta_{opt} \sim \sqrt{2 D_{crit}}$$

where $D_{crit}$ is the distance from the bifurcation point in parameter
space. Too little noise → no oscillation. Too much → noisy chaos.

### Temperature mapping

In the original paper, temperature $T$ maps to conductance scaling:

$$g_{sd}(T) = g_{sd,0} \cdot Q_{10}^{(T - T_0)/10}$$
$$g_{sr}(T) = g_{sr,0} \cdot Q_{10}^{(T - T_0)/10}$$

with different $Q_{10}$ values for each conductance. The current
implementation uses fixed conductances — temperature must be mapped
manually by adjusting $g_{sd}$ and $g_{sr}$.

### Relationship to other cold receptor models

The Huber-Braun model is a simplification of:
- **Braun-Voigt (1986):** 4 conductances (fast Na, fast K, slow Na, slow K)
- **Feudel et al. (2000):** Extended Huber-Braun with fast subsystem

And is conceptually related to:
- **FitzHugh-Nagumo:** Same 2-variable oscillator structure, but HB has
  conductance-based (not polynomial) dynamics
- **Morris-Lecar:** Similar slow dynamics, but ML has Ca²⁺ activation

---

## Phase Portrait Analysis

### Nullclines

The V-nullcline ($dV/dt = 0$) is:

$$V = \frac{g_{sd} \cdot a_{sd} \cdot E_{sd} + g_{sr} \cdot a_{sr} \cdot E_{sr} + g_L \cdot E_L + I}{g_{sd} \cdot a_{sd} + g_{sr} \cdot a_{sr} + g_L}$$

This is a weighted average of reversal potentials, with weights equal to
the effective conductances. The nullcline shape depends on the activation
functions through $a_{sd}$ and $a_{sr}$.

### Fixed points

At the fixed point, $a_{sd} = \sigma_{sd}(V^*)$ and $a_{sr} = \sigma_{sr}(V^*)$.
The fixed point voltage $V^*$ satisfies:

$$g_{sd} \cdot \sigma_{sd}(V^*)(V^* - E_{sd}) + g_{sr} \cdot \sigma_{sr}(V^*)(V^* - E_{sr}) + g_L(V^* - E_L) = I$$

With default parameters and $I = 0$: V* ≈ −50 mV (near $E_L$).

### Stability

The Jacobian eigenvalues determine oscillatory behaviour:
- **Real, negative:** Stable node (no oscillation)
- **Complex with negative real part:** Damped oscillation
- **Complex with positive real part:** Unstable spiral → limit cycle

The transition occurs at a **Hopf bifurcation** as $g_{sd}/g_{sr}$ increases.

---

## FPGA Considerations

### Resource estimate

| Component | LUTs | Notes |
|-----------|------|-------|
| 2 sigmoid LUTs | ~128 | Piecewise-linear approximation |
| 3 multipliers | ~96 | Conductance × (V − E) |
| 2 dividers | ~64 | Tau division for gating update |
| Accumulator | ~48 | V integration |
| **Total** | **~336** | Fits in smallest FPGAs |

### Latency

Single pipeline stage: 3 multiplications + 2 exp lookups ≈ 5 clock cycles
at 100 MHz → 50 ns/step. Comparable to Rust (71.4 ns) but at much lower
power.

### Noise generation

For FPGA stochastic resonance, an LFSR-based Gaussian generator
(Box-Muller or Ziggurat) adds ~200 LUTs.

---

## Usage Examples

### Basic cold receptor simulation

```rust
use sc_neurocore_engine::neurons::HuberBraunNeuron;

let mut n = HuberBraunNeuron::new();
let mut spikes = 0;
for _ in 0..10_000 {
    spikes += n.step(10.0);
}
println!("Spikes: {spikes}");
```

### Temperature sweep (manual conductance mapping)

```rust
for temp_factor in [0.5, 1.0, 1.5, 2.0] {
    let mut n = HuberBraunNeuron::new();
    n.g_sd *= temp_factor;  // Higher = colder
    let spikes: i32 = (0..10_000).map(|_| n.step(0.0)).sum();
    println!("g_sd factor {temp_factor}: {spikes} spikes");
}
```

---

## Python/Rust Sigmoid Discrepancy

**IMPORTANT:** The Python and Rust implementations use different sigmoid
parameters for the activation functions:

| Function | Python | Rust |
|----------|--------|------|
| $\sigma_{sd}$ V1/2 | −40.0 mV | −25.0 mV |
| $\sigma_{sd}$ slope | 6.0 mV | 5.0 mV |
| $\sigma_{sr}$ V1/2 | −40.0 mV | −40.0 mV |
| $\sigma_{sr}$ slope | 6.0 mV | 5.0 mV |

The SD activation half-voltage differs by 15 mV between implementations.
This means:
- **Python:** SD activates at more hyperpolarised voltages → easier to
  depolarise → different oscillation regime
- **Rust:** SD activates at more depolarised voltages → requires stronger
  drive → different threshold

Both are valid parameterisations from different variants of the
Braun et al. family of papers. The Rust version uses parameters closer
to Braun et al. (1998) original, while the Python version may follow
Feudel et al. (2000) or another variant.

**Impact:** Spike counts and oscillation regimes differ between Python
and Rust implementations. This is a known discrepancy, not a bug — the
model family has many published parameterisations.

---

## Sensitivity Analysis

### Conductance sensitivity

| Parameter | Effect of 2× increase | Effect of 0.5× |
|-----------|----------------------|----------------|
| g_sd | More depolarised, earlier spiking | Less excitable |
| g_sr | Stronger repolarisation, longer IBI | Weaker recovery |
| g_l | Faster return to rest | Slower dynamics |

### Time constant sensitivity

| Parameter | Effect of 2× increase | Effect of 0.5× |
|-----------|----------------------|----------------|
| tau_sd | Slower depolarisation, wider bursts | Narrower bursts |
| tau_sr | Longer inter-burst interval | Shorter IBI |

### Noise sensitivity (Python only)

| eta | Behaviour |
|-----|-----------|
| 0.0 | Deterministic (fixed point or limit cycle) |
| 0.001–0.01 | Subthreshold fluctuations |
| 0.01–0.05 | Stochastic resonance regime |
| > 0.1 | Noise-dominated, irregular firing |

---

## Stability Analysis

### Linearisation at rest

At V* = −50 mV with default parameters:

$$J = \begin{pmatrix} -(g_{sd} a_{sd}^* + g_{sr} a_{sr}^* + g_L) & -g_{sd}(V^* - E_{sd}) & -g_{sr}(V^* - E_{sr}) \\ \sigma'_{sd}(V^*)/\tau_{sd} & -1/\tau_{sd} & 0 \\ \sigma'_{sr}(V^*)/\tau_{sr} & 0 & -1/\tau_{sr} \end{pmatrix}$$

where $\sigma'_{sd}(-50) \approx 0.0066$ and $\sigma'_{sr}(-50) \approx 0.105$
(Rust parameters).

The eigenvalues determine local stability. With default parameters, the
resting state is stable (all eigenvalues have negative real parts) but
close to a Hopf bifurcation — explaining the sensitivity to $g_{sd}$.

### Hopf bifurcation locus

The critical $g_{sd}$ for oscillation onset (Rust parameters) is
approximately:

$$g_{sd,crit} \approx g_{sr} \cdot \frac{\tau_{sd}}{\tau_{sr}} \cdot \frac{E_{sr} - E_L}{E_{sd} - E_L} + g_L$$

With defaults: $g_{sd,crit} \approx 0.4 \times 0.5 \times 1.33 + 0.1 \approx 0.37$.
Since $g_{sd} = 1.5 > 0.37$, the model is well above the oscillation
threshold — explaining the depolarisation block behaviour.

---

## Biological Accuracy Assessment

### What the model captures

- Slow oscillatory mechanism of cold receptors ✓
- Dual conductance competition (depolarising vs repolarising) ✓
- Temperature dependence via conductance scaling ✓
- Stochastic resonance potential (Python only) ✓
- Burst/tonic transition as function of parameters ✓

### What the model omits

- **Fast Na⁺/K⁺ channels:** No action potentials — only slow
  depolarisation events. Real cold receptors fire fast APs riding on
  slow oscillations.
- **TRP channels:** TRPM8 (menthol receptor) sets the actual temperature
  threshold. Not modelled.
- **Ca²⁺ dynamics:** Some cold receptors use Ca²⁺-activated K channels
  for burst termination. Not included.
- **Axonal propagation:** The model is a single-compartment point neuron.

### Published validation

Braun et al. (1998) validated the model against:
- Lingual nerve recordings from cat (cold fibres)
- ISI histograms matching experimental data
- Bifurcation diagrams matching temperature protocols

The model reproduces the qualitative features (burst patterns, stochastic
resonance) but not the quantitative spike shapes (no fast AP mechanism).

---

## Version History

| Date | Change | Commit |
|------|--------|--------|
| 2026-03-20 | Initial Python implementation | — |
| 2026-04-04 | Rust port via NeuronVariant | — |
| 2026-04-05 | Multi-angle Rust tests (6 tests) | `328cd4e` |
| 2026-04-05 | Criterion benchmark: 71.4 ns/step | `71bd1ec` |
| 2026-04-05 | Doc expanded with verification + benchmarks | `4bfc1a9` |

---

## Current Decomposition at Rest

At V = −50 mV with default parameters (Rust implementation):

### Activation states

$$a_{sd}^* = \sigma_{sd}(-50) = \frac{1}{1 + e^{(50-25)/5}} = \frac{1}{1 + e^5} = 0.00669$$
$$a_{sr}^* = \sigma_{sr}(-50) = \frac{1}{1 + e^{(-50+40)/5}} = \frac{1}{1 + e^{-2}} = 0.881$$

### Individual currents at rest (I_ext = 0)

$$I_{sd}^* = 1.5 \times 0.00669 \times (-50 - 50) = -1.004 \text{ µA/cm²}$$
$$I_{sr}^* = 0.4 \times 0.881 \times (-50 - (-90)) = +14.10 \text{ µA/cm²}$$
$$I_L^* = 0.1 \times (-50 - (-60)) = +1.0 \text{ µA/cm²}$$

**Net current:** −1.004 + 14.10 + 1.0 = +14.10 µA/cm² (outward, stabilising)

The strong outward SR current at rest explains why external drive is
needed to trigger depolarisation. The SD current is nearly zero at rest
because $\sigma_{sd}(-50) \approx 0.007$.

### Drive required for depolarisation

To shift V toward $V_{sd,1/2}$ = −25 mV, the external current must
overcome the net outward current:

$$I_{ext,min} \approx 14.1 \text{ µA/cm²}$$

In practice, the dynamic coupling means the threshold is lower because
as V depolarises, $a_{sd}$ increases (positive feedback), partially
cancelling the outward currents.

### Energy balance during oscillation

During one oscillation cycle:
- **Depolarisation phase:** SD current dominates (inward), charging C_m
- **Repolarisation phase:** SR current dominates (outward), discharging C_m
- **Return phase:** Leak drives V toward E_L between bursts

The energy per cycle is approximately:
$$E_{cycle} \approx C_m \cdot \Delta V^2 / R_{eff}$$

where $\Delta V \approx E_{sd} - E_{sr} = 140$ mV and $R_{eff}$ is the
effective membrane resistance during the cycle.
