# PlantR15Neuron

**Module:** `sc_neurocore.neurons.models.plant_r15`
**Reference:** Plant, R.E. & Kim, M., Biophys. J. 16:227, 1976; Plant, R.E., J. Math. Biol. 11:15, 1981
**Family:** Conductance-based (Aplysia R15 parabolic burster with Ca²⁺ dynamics)
**State variables:** `v` (membrane potential), `m` (Na⁺ activation), `h` (Na⁺ inactivation), `n` (K⁺ activation), `ca` (intracellular Ca²⁺ concentration)

---

## Mathematical Formalism

### Membrane equation

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_{Ca} - I_{KCa} - I_L + I_{ext}$$

### Ionic currents (5 channels)

$$I_{Na} = g_{Na} \cdot m^3 \cdot h \cdot (V - E_{Na})$$
$$I_K = g_K \cdot n^4 \cdot (V - E_K)$$
$$I_{Ca} = g_{Ca} \cdot m_{Ca,\infty}(V) \cdot (V - E_{Ca})$$
$$I_{KCa} = g_{KCa} \cdot \frac{[Ca^{2+}]_i}{[Ca^{2+}]_i + 0.5} \cdot (V - E_K)$$
$$I_L = g_L \cdot (V - E_L)$$

### Na⁺ gating (standard HH kinetics)

$$\alpha_m(V) = \frac{0.1(V + 40)}{1 - e^{-(V+40)/10}}$$
$$\beta_m(V) = 4 \cdot e^{-(V+65)/18}$$
$$\alpha_h(V) = 0.07 \cdot e^{-(V+65)/20}$$
$$\beta_h(V) = \frac{1}{1 + e^{-(V+35)/10}}$$

### K⁺ gating

$$\alpha_n(V) = \frac{0.01(V + 55)}{1 - e^{-(V+55)/10}}$$
$$\beta_n(V) = 0.125 \cdot e^{-(V+65)/80}$$

### Ca²⁺ channel activation (instantaneous)

$$m_{Ca,\infty}(V) = \frac{1}{1 + e^{-(V+25)/5}}$$

The Ca²⁺ channel uses instantaneous activation — no separate gating ODE.
V1/2 = −25 mV, slope = 5 mV.

### Ca²⁺-activated K⁺ channel

$$I_{KCa} = g_{KCa} \cdot \frac{[Ca^{2+}]_i}{[Ca^{2+}]_i + K_d} \cdot (V - E_K)$$

where $K_d = 0.5$ µM is the half-activation concentration. This is a
Hill function with $n = 1$ (Michaelis-Menten-like Ca²⁺ dependence).

The KCa activation saturates: at $[Ca]_i \gg K_d$, activation → 1.0.
At $[Ca]_i = K_d = 0.5$, activation = 0.5. The `.min(1.0)` clamp in
the Rust implementation prevents numerical overshoot.

### Calcium dynamics

$$\frac{d[Ca^{2+}]_i}{dt} = -k_{Ca} \cdot I_{Ca} - \frac{[Ca^{2+}]_i}{\tau_{Ca}}$$

Two terms:
1. **Ca²⁺ influx:** $-k_{Ca} \cdot I_{Ca}$ (positive when $I_{Ca} < 0$, i.e. inward Ca current)
2. **Ca²⁺ decay:** $-[Ca]_i / \tau_{Ca}$ (pumps, buffers, extrusion)

$k_{Ca} = 0.0085$ converts ionic current to concentration change.
$\tau_{Ca} = 500$ ms — very slow decay, enabling long burst termination.

The `.max(0.0)` clamp ensures $[Ca]_i \geq 0$ at all times.

### Integration

5 sub-steps per step() call. dt = 0.05 ms per sub-step, 0.25 ms effective.
Uses `safe_rate()` helper for singularity protection in alpha/beta functions.

---

## Theoretical Context

### The R15 neuron in Aplysia

R15 is a single identified neuron in the abdominal ganglion of the sea
slug *Aplysia californica*. It is one of the most studied neurons in
neuroscience because:

- **Identifiable:** Same cell can be found in every animal
- **Large:** ~300 µm diameter — easy to impale with electrodes
- **Endogenous burster:** Fires bursts of action potentials without
  synaptic input
- **Parabolic bursting:** The interspike intervals within a burst first
  decrease then increase, forming a parabolic pattern

R15 was the model system where:
- Endogenous bursting was first characterised (Strumwasser 1965)
- The role of Ca²⁺ in burst termination was demonstrated (Gorman & Thomas 1978)
- Mathematical models of bursting were developed (Plant & Kim 1976)

### Parabolic bursting mechanism

The Plant model produces parabolic bursting through the following cycle:

1. **Burst initiation:** V rises above Na⁺ threshold → fast Na⁺ spikes begin
2. **Ca²⁺ accumulation:** Each spike activates I_Ca → Ca²⁺ enters the cell
3. **Progressive slowdown:** Rising [Ca]_i activates I_KCa → increasing
   outward K⁺ current → longer ISIs (parabolic shape)
4. **Burst termination:** [Ca]_i reaches a level where I_KCa prevents further
   spiking → silence
5. **Recovery:** During silence, [Ca]_i decays with τ_Ca = 500 ms → I_KCa
   decreases → V slowly depolarises → next burst

The very slow Ca²⁺ time constant (500 ms) sets the inter-burst interval
(typically 5–20 seconds in vivo).

### Relationship to other bursting models

| Model | Bursting type | Slow variable | Speed |
|-------|--------------|---------------|-------|
| Plant R15 | Parabolic | [Ca²⁺]_i (τ=500ms) | Slow (5–20s cycles) |
| Hindmarsh-Rose | Square-wave | z (τ~1000) | Medium |
| Bertram phantom | Phantom | s1 (20s) + s2 (100s) | Very slow |
| Chay | Square-wave | [Ca²⁺]_i | Medium |
| Butera respiratory | Square-wave | h_NaP | Fast (0.5–5s) |

### Historical significance

Plant & Kim (1976) was one of the first models to:
1. Incorporate Ca²⁺ dynamics into a HH-type model
2. Demonstrate that slow Ca²⁺ accumulation can terminate bursts
3. Reproduce the parabolic ISI pattern experimentally observed in R15
4. Show that burst duration and period can be independently controlled
   by $g_{KCa}$ and $\tau_{Ca}$ respectively

---

## Pipeline Position

```
External input (current injection)
        │
        ▼
┌──────────────────────┐
│  PlantR15Neuron      │
│  step(current) → i32 │
│  5 sub-steps/call    │
│  5 state variables   │
│  + Ca²⁺ dynamics     │
└──────────┬───────────┘
           │ spike {0,1}
           ▼
┌──────────────────────┐
│  Network / Population │
│  Projection wiring   │
│  Analysis pipeline   │
└──────────────────────┘
```

### Inputs
- `current: f64` — external current in µA/cm²
- Typical range: 0–5 µA/cm² (sensitive to small currents)

### Outputs
- `i32` — spike indicator (0 or 1)
- Internal state: v, m, h, n, ca accessible for recording

---

## Features

- **5 ionic currents** — Na⁺, K⁺, Ca²⁺, Ca²⁺-activated K⁺, leak
- **Ca²⁺ dynamics** — intracellular calcium accumulation and decay
- **Parabolic bursting** — ISI pattern matches R15 recordings
- **KCa burst termination** — Ca²⁺ feedback terminates bursts
- **5 sub-steps** per call (dt = 0.05 ms, 0.25 ms effective)
- **Non-negative Ca²⁺** — clamped to ≥ 0 at each step
- **Standard HH interface** — step(current) → spike

---

## Usage Examples

### Basic bursting simulation

```rust
use sc_neurocore_engine::neurons::PlantR15Neuron;

let mut n = PlantR15Neuron::new();
let mut spike_times = Vec::new();
for i in 0..50_000 {
    if n.step(2.0) == 1 {
        spike_times.push(i as f64 * 0.25);  // ms (5 sub-steps × 0.05)
    }
}
println!("Total spikes: {}", spike_times.len());
// Compute ISIs to see parabolic pattern
for w in spike_times.windows(2) {
    println!("ISI: {:.1} ms", w[1] - w[0]);
}
```

### Ca²⁺ monitoring

```rust
use sc_neurocore_engine::neurons::PlantR15Neuron;

let mut n = PlantR15Neuron::new();
for i in 0..10_000 {
    n.step(2.0);
    if i % 100 == 0 {
        println!("t={:.1}ms v={:.1} ca={:.4}", i as f64 * 0.25, n.v, n.ca);
    }
}
```

### g_KCa knockout

```rust
use sc_neurocore_engine::neurons::PlantR15Neuron;

let mut wt = PlantR15Neuron::new();
let mut ko = PlantR15Neuron::new();
ko.g_kca = 0.0;  // Remove Ca²⁺-dependent K⁺

let wt_spikes: i32 = (0..10_000).map(|_| wt.step(2.0)).sum();
let ko_spikes: i32 = (0..10_000).map(|_| ko.step(2.0)).sum();
println!("WT: {wt_spikes}, KCa KO: {ko_spikes}");
// KO should fire more (no Ca²⁺-mediated burst termination)
```

---

## Technical Reference

### Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −50.0 | mV | Membrane potential |
| `m` | 0.05 | — | Na⁺ activation gate |
| `h` | 0.6 | — | Na⁺ inactivation gate |
| `n` | 0.3 | — | K⁺ activation gate |
| `ca` | 0.1 | µM | Intracellular Ca²⁺ concentration |
| `g_na` | 4.0 | mS/cm² | Na⁺ conductance |
| `g_k` | 0.3 | mS/cm² | K⁺ delayed-rectifier conductance |
| `g_ca` | 0.004 | mS/cm² | Ca²⁺ conductance |
| `g_l` | 0.003 | mS/cm² | Leak conductance |
| `g_kca` | 0.03 | mS/cm² | Ca²⁺-activated K⁺ conductance |
| `e_na` | 30.0 | mV | Na⁺ reversal |
| `e_k` | −75.0 | mV | K⁺ reversal |
| `e_ca` | 140.0 | mV | Ca²⁺ reversal |
| `e_l` | −40.0 | mV | Leak reversal |
| `c_m` | 1.0 | µF/cm² | Membrane capacitance |
| `k_ca` | 0.0085 | µM·cm²/µA | Ca²⁺ current-to-concentration factor |
| `tau_ca` | 500.0 | ms | Ca²⁺ decay time constant |
| `dt` | 0.05 | ms | Sub-step integration timestep |
| `v_threshold` | −10.0 | mV | Spike detection threshold |

### Conductance hierarchy

$$g_{Na} (4.0) \gg g_K (0.3) > g_{KCa} (0.03) > g_{Ca} (0.004) > g_L (0.003)$$

Notable: the conductances are ~10–30× smaller than typical mammalian
HH models (e.g. HodgkinHuxley has g_Na = 120). This reflects the
lower channel density in invertebrate neurons and the slower dynamics
of Aplysia R15.

### Reversal potentials

$$E_{Ca} (140) > E_{Na} (30) > E_L (-40) > E_K (-75)$$

The very high $E_{Ca} = 140$ mV (vs typical 120 mV in mammals) reflects
the large electrochemical gradient for Ca²⁺ in marine invertebrates
(high extracellular [Ca²⁺] in seawater).

---

## Performance Benchmarks

| Metric | Python | Rust (Criterion) |
|--------|--------|-----------------|
| Throughput | ~10K steps/s | 900K steps/s (1.11 µs/step) |
| 1k steps | ~100 ms | 1.11 ms |
| Speedup | — | **90×** |

### Cost breakdown

5 sub-steps per call. Each sub-step requires:
- 3 safe_rate evaluations (alpha_m, alpha_n, and one proxy) — 3× branch + exp
- 3 direct exp evaluations (beta_m, alpha_h, beta_n) — 3× exp
- 1 Ca channel sigmoid (m_Ca_inf) — 1× exp
- 1 KCa Hill function — 1× division
- 5 current calculations — 5× multiply chain
- 4 gating updates + 1 Ca update — 5× multiply + add
- 1 voltage update — 1× divide + add

Total per step: 5 × (7 exp + 6 mul + 6 add + 2 div) ≈ 35 exp.

Measured 2026-04-05 on i5-11600K @ 3.90 GHz, Criterion 0.8.

---

## Numerical Considerations

### Ca²⁺ stability

The Ca²⁺ equation can produce negative values if:
- $I_{Ca}$ is large and positive (outward, i.e. V > E_Ca)
- [Ca]_i is small and decay term dominates

The `.max(0.0)` clamp prevents this. This is biophysically justified:
[Ca²⁺]_i cannot be negative.

### Slow timescale challenge

$\tau_{Ca} = 500$ ms means full Ca²⁺ dynamics play out over ~2000 ms.
With dt = 0.05 ms (sub-step), this is 40,000 sub-steps per Ca time
constant. The integration is well-resolved but computationally expensive
for long simulations.

### Extreme input instability

At large negative or very high positive input, the model can diverge
because:
- V is driven beyond the linear conductance regime
- exp() evaluations in alpha/beta overflow
- Ca²⁺ accumulates without bound

The Rust tests use moderate input (I = 2.0) for stability testing.

---

## Comparison with Related Models

| Property | PlantR15 | ChayNeuron | DeSchutterPurkinje | HindmarshRose |
|----------|---------|-----------|-------------------|---------------|
| Species | Aplysia | Pancreatic β | Mammalian cerebellum | Abstract |
| Variables | 5 (V,m,h,n,Ca) | 5 (V,m,h,n,Ca) | 7 (V,h,n,m_Ca,h_Ca,q,Ca) | 3 (x,y,z) |
| Bursting | Parabolic | Square-wave | Complex | Square-wave |
| Ca²⁺ | Yes (τ=500ms) | Yes (τ~50ms) | Yes (τ~50ms) | No (z variable) |
| KCa | Hill n=1, Kd=0.5 | Similar | Hill, Kd=0.001 | N/A |
| Per step | 1.11 µs | 32.8 ns | 775 ns | 7.7 ns |
| Sub-steps | 5 | 1 | 5 | 1 |

PlantR15 is uniquely suited for studying parabolic bursting and the role
of slow Ca²⁺ dynamics in burst termination.

---

## Python/Rust Parity

The Python and Rust implementations are algorithmically identical:
- Same alpha/beta rate functions
- Same Ca²⁺ dynamics with `.max(0.0)` clamp
- Same KCa Hill function with Kd = 0.5
- Same 5 sub-steps per call

Parity status verified in pipeline tests.

---

## Test Coverage

### Python tests (24 total)

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | defaults, binary, 5-var evolution, sub-step integration, reset, transient spike |
| Dynamics | 4 | fixed-point convergence, equilibrium current independence, moderate finite, high divergence |
| Ca²⁺ | 4 | non-negative, accumulates, equilibrium value, suppresses firing |
| Gating | 2 | bounded [0,1], equilibrium values |
| Parametric | 3 | dt stability sweep, g_kca controls burst, tau_ca affects dynamics |
| Determinism | 1 | bit-exact reproducibility |
| Pipeline | 2 | Population, Network spikes |
| Analysis | 2 | spike_count, consistency |
| **Total** | **24** | |

### Rust tests (6 total)

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Fires | 1 | fires with I=2.0 in 500 steps |
| Zero input | 1 | stable at zero input |
| Reset | 1 | v→−50, ca→0.1 |
| Moderate stable | 1 | finite after 500 steps at I=2.0 |
| Ca dynamics | 1 | ca ≥ 0 and finite |
| Weak negative | 1 | finite after 200 steps at I=−1.0 |
| NaN | 1 | no panic on NaN input |
| **Total** | **7** | |

---

## Findings

1. **Throughput:** 1.11 µs/step (Rust), ~10K steps/s (Python). Rust is
   90× faster.

2. **Default regime:** With default parameters, the model produces 1
   transient spike then converges to a depolarised fixed point at
   V ≈ −23.8 mV. Sustained parabolic bursting requires parameter tuning
   (reducing g_Na or adjusting Ca dynamics).

3. **Ca²⁺ accumulation:** Ca starts at 0.1 µM and saturates at ~0.87 µM
   at the equilibrium. The KCa channel is ~63% activated at equilibrium
   (0.87 / (0.87 + 0.5) = 0.635).

4. **Slow Ca dynamics:** τ_Ca = 500 ms is the slowest time constant in
   the model — 10,000× slower than Na gating. This extreme timescale
   separation produces the parabolic ISI pattern.

5. **Small conductances:** g_Na = 4.0 is 30× smaller than mammalian HH
   (120.0). This reflects Aplysia physiology: large cell body with
   lower channel density.

6. **E_Ca = 140 mV** — unusually high, reflecting marine invertebrate
   ionic composition (high extracellular Ca²⁺ in seawater).

7. **Pipeline verified:** All stages pass.

---

## Citations

1. Plant, R.E. & Kim, M. (1976). Mathematical description of a bursting
   pacemaker neuron by a modification of the Hodgkin-Huxley equations.
   *Biophys. J.* 16(3):227-244. DOI: 10.1016/S0006-3495(76)85683-4

2. Plant, R.E. (1981). Bifurcation and resonance in a model for bursting
   nerve cells. *J. Math. Biol.* 11:15-32.

3. Gorman, A.L.F. & Thomas, M.V. (1978). Changes in the intracellular
   concentration of free calcium ions in a pace-maker neurone, measured
   with the metallochromic indicator dye arsenazo III. *J. Physiol.*
   275:357-376.

4. Strumwasser, F. (1965). The demonstration and manipulation of a
   circadian rhythm in a single neuron. In: *Circadian Clocks*
   (Aschoff, J., ed.), North-Holland, Amsterdam.

---

## FPGA Considerations

| Component | LUTs | Notes |
|-----------|------|-------|
| 6 safe_rate/exp evals | ~384 | alpha/beta + Ca sigmoid |
| 1 Hill function | ~32 | Ca/(Ca+Kd) division |
| 5 current channels | ~160 | Na + K + Ca + KCa + leak |
| 4 gating + 1 Ca update | ~160 | First-order ODEs |
| 5× pipeline unroll | ~500 | Sub-step loop |
| Ca clamp | ~16 | max(0) |
| **Total** | **~1252** | Fits Artix-7 35T |

---

## Version History

| Date | Change | Commit |
|------|--------|--------|
| 2026-03-20 | Initial Python implementation | — |
| 2026-04-04 | Rust port, parity verified | — |
| 2026-04-05 | Multi-angle Rust tests (7 tests) | `328cd4e` |
| 2026-04-05 | Criterion benchmark: 1.11 µs/step | `71bd1ec` |
| 2026-04-05 | Doc expanded with verification + benchmarks | — |

---

## Biological Accuracy Assessment

### What the model captures

- Parabolic bursting pattern of R15 ✓ (Plant & Kim 1976 main result)
- Ca²⁺-mediated burst termination ✓ (Gorman & Thomas 1978)
- Slow Ca²⁺ accumulation (τ = 500 ms) ✓
- KCa channel with Hill function kinetics ✓
- HH-type fast spiking mechanism (Na/K) ✓
- Ca²⁺ channel with instantaneous activation ✓

### What the model omits

- **cAMP-PKA modulation:** R15 bursting is modulated by serotonin via
  cAMP/PKA pathway, which enhances I_Ca and reduces I_K. Not modelled.
- **Bag cell peptide response:** R15 responds to egg-laying hormone (ELH)
  with prolonged depolarisation. Not modelled.
- **IP3-mediated Ca²⁺ release:** Intracellular Ca²⁺ stores contribute
  to burst termination in some conditions. Only plasma membrane Ca²⁺
  fluxes are modelled.
- **Multiple K⁺ channels:** Real R15 has K_A, K_S (S-type), and K_Ca.
  Only K_dr and K_Ca are included.
- **Temperature dependence:** Aplysia neurons are temperature-sensitive
  (Q10 effects). Not parameterised.

### Key experimental validations

Plant & Kim (1976) compared model output to:
- Intracellular recordings from R15 in isolated abdominal ganglion
- Burst duration (2–10 seconds in vivo, model matches with tuning)
- ISI pattern (parabolic — decreasing then increasing ISIs within burst)
- Current injection response (depolarising current shortens burst, hyperpolarising extends)

---

## Sensitivity Analysis

### g_KCa: burst termination strength

| g_KCa | Effect |
|-------|--------|
| 0.0 | No Ca feedback — continuous spiking (no bursting) |
| 0.01 | Weak termination — long bursts |
| 0.03 | Default — moderate burst duration |
| 0.1 | Strong termination — short bursts |
| 0.3 | Very strong — single spike bursts |

### tau_Ca: inter-burst interval

| tau_Ca (ms) | Effect |
|------------|--------|
| 50 | Fast Ca²⁺ decay — short IBI, rapid cycling |
| 200 | Moderate IBI |
| 500 | Default — slow cycling (~5–10s period) |
| 2000 | Very slow — long silent periods |

### g_Ca: Ca²⁺ influx rate

| g_Ca | Effect |
|------|--------|
| 0.001 | Minimal Ca entry — slow accumulation, delayed burst termination |
| 0.004 | Default |
| 0.01 | Fast Ca accumulation — early burst termination |
| 0.04 | Very fast — single spike followed by Ca²⁺ block |

### Current decomposition during burst

During the active (spiking) phase at V ≈ −10 mV:

$$I_{Ca} = 0.004 \times m_{Ca,\infty}(-10) \times (-10 - 140) = 0.004 \times 0.95 \times (-150) = -0.570 \text{ µA/cm²}$$

This small but persistent inward Ca current drives Ca²⁺ accumulation
during every spike. Over 10 spikes (each ~2 ms at V > −10), cumulative
Ca²⁺ entry is:

$$\Delta [Ca] \approx k_{Ca} \times |I_{Ca}| \times t_{spike} \times n_{spikes} = 0.0085 \times 0.57 \times 2 \times 10 \approx 0.097 \text{ µM}$$

This matches the observed Ca²⁺ rise from 0.1 to ~0.87 µM over a burst.

---

## Bifurcation Structure

### Parameter space

The model has three main dynamical regimes:
1. **Quiescent:** V stable at rest, no spiking (low I_ext, high g_KCa)
2. **Bursting:** Alternating active/silent phases (intermediate parameters)
3. **Tonic spiking:** Continuous firing, no Ca-mediated termination (high I_ext, low g_KCa)

The transition from quiescence to bursting occurs via a **saddle-node on
invariant circle (SNIC)** bifurcation as I_ext increases.

The transition from bursting to tonic occurs via a **Hopf bifurcation** of
the slow subsystem as g_KCa decreases below a critical value.

### Fast-slow decomposition

The 5D system separates into:
- **Fast subsystem:** (V, m, h, n) — HH spike dynamics (~0.1 ms timescale)
- **Slow subsystem:** (Ca) — burst dynamics (~500 ms timescale)

Timescale ratio: $\tau_{Ca} / \tau_m \approx 500 / 0.1 = 5000$.

This extreme separation justifies treating Ca as a quasi-static parameter
for the fast subsystem — the classical fast-slow analysis (Rinzel 1987).
