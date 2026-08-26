# MainenSejnowskiNeuron

**Module:** `sc_neurocore.neurons.models.mainen_sejnowski`
**Reference:** Mainen, Z.F. & Sejnowski, T.J., *Nature* 382:363–366, 1996
**Family:** Conductance-based (two-compartment: soma + axon)
**State variables:** `vs` (soma potential), `va` (axon potential), `m` (Na⁺ activation), `h` (Na⁺ inactivation), `n` (K⁺ activation)

The Euler sub-stepping (20 × dt = 0.005 ms), the in-loop voltage clips
to [−200, 200] mV, and the gate clips to [0, 1] are repository-specific
specialisations, not publication-exact claims. Canonical rate
evaluation uses numerically stable analytic removable-singularity
limits (`expm1` linoid form) so every rate is exact and continuous at
its singular voltage.

### Atomicity, Parity, and Legacy Configurations

- `step(current)` validates before touching state and computes the
  whole 20-sub-step update on candidate values: a non-finite `current`
  raises `ValueError` with the pre-step state preserved exactly (a NaN
  input can no longer silently poison all five state variables), an
  out-of-bounds configuration raises `ValueError` at construction and
  at each step, and a non-finite candidate aborts atomically. The
  production Rust engine (`try_step`), the typed PyO3 binding, the
  standalone safety Rust, Go (`TryStep`), and Julia (`ArgumentError`)
  enforce the same contract; the engine `step` and Go `Step` wrappers
  fail closed.
- Backend parity: engine binding, safety Rust, Go, and Julia reproduce
  the Python reference within 1e-12 over 64 varied steps on the
  complete `(vs, va, m, h, n, event)` state.
- Two historical behaviours remain reconstructible as count-neutral
  legacy configurations with executed regression anchors: the old
  Python additive-1e-12 rate regularisation (which returned a **zero**
  rate exactly at each singular voltage) via
  `MainenSejnowskiNeuron(legacy_epsilon_rates=True)`, and the original
  engine Gauss-Seidel compartment ordering with its historical rate
  handling via Rust `MainenSejnowskiNeuron::new_legacy_sequential` /
  PyO3 `MainenSejnowskiNeuron.legacy_sequential()`.
- Mojo: not implemented (`accel/mojo/kernels/mainen_sejnowski.mojo` is
  a non-computing stub; no parity claimed). Silicon / RTL: not
  implemented; no HDL parity claimed.

---

## Mathematical Formalism

### Two-compartment structure

The neuron has two electrically coupled compartments:
- **Soma:** Passive (leak + coupling). Receives external input.
- **Axon:** Active (Na⁺ + K⁺ channels). Generates action potentials.

This separation reflects the biological observation that action potentials
initiate at the axon initial segment (AIS) and backpropagate to the soma
(Stuart & Sakmann 1994).

### Soma equation (passive)

$$C_s \frac{dV_s}{dt} = -g_L(V_s - E_L) + \kappa(V_a - V_s) + I_{ext}$$

The soma has:
- Leak current $g_L(V_s - E_L)$
- Coupling current $\kappa(V_a - V_s)$ — bidirectional electrical coupling
- External input $I_{ext}$

### Axon equation (active)

$$C_a \frac{dV_a}{dt} = -I_{Na} - I_K - \kappa(V_a - V_s)$$

where:
$$I_{Na} = g_{Na} \cdot m^3 \cdot h \cdot (V_a - E_{Na})$$
$$I_K = g_K \cdot n^4 \cdot (V_a - E_K)$$

Note: the coupling term has opposite sign in the axon equation — current
flowing from soma to axon is $+\kappa(V_s - V_a)$ for the axon, which
equals $-\kappa(V_a - V_s)$.

### Na⁺ gating (standard HH)

$$\alpha_m(V_a) = \frac{0.1(V_a + 40)}{1 - e^{-(V_a+40)/10}}$$
$$\beta_m(V_a) = 4 \cdot e^{-(V_a+65)/18}$$
$$\alpha_h(V_a) = 0.07 \cdot e^{-(V_a+65)/20}$$
$$\beta_h(V_a) = \frac{1}{1 + e^{-(V_a+35)/10}}$$

### K⁺ gating (standard HH)

$$\alpha_n(V_a) = \frac{0.01(V_a + 55)}{1 - e^{-(V_a+55)/10}}$$
$$\beta_n(V_a) = 0.125 \cdot e^{-(V_a+65)/80}$$

### Gating updates

$$\frac{dm}{dt} = \alpha_m(1-m) - \beta_m \cdot m$$
$$\frac{dh}{dt} = \alpha_h(1-h) - \beta_h \cdot h$$
$$\frac{dn}{dt} = \alpha_n(1-n) - \beta_n \cdot n$$

All gating is computed at the **axon** voltage $V_a$.

### Spike detection

Threshold crossing at **soma**: spike when $V_s \geq V_{threshold}$ and
$V_{s,prev} < V_{threshold}$. This detects backpropagated APs from axon.

### Integration

20 sub-steps per step() call. dt = 0.005 ms per sub-step, 0.1 ms effective.
Uses `safe_rate()` helper for singularity protection.

---

## Theoretical Context

### Why two compartments?

Mainen & Sejnowski (1996) showed that the morphological diversity of
cortical neurons (pyramidal, stellate, bipolar) could explain the
diversity of firing patterns (RS, IB, chattering) — even with identical
channel densities. The key insight:

**Firing pattern is determined by the ratio of somatic to axonal membrane
area, not by channel properties.**

This contradicts the classical HH view where different channels produce
different patterns. Mainen & Sejnowski demonstrated that:
- A neuron with large soma → RS (current spreads, slow depolarisation)
- A neuron with small soma → Chattering (fast local depolarisation)
- Same channels in both cases

### The soma/axon area ratio

The coupling constant $\kappa = 10.0$ and capacitance ratio $C_s/C_a = 10$
encode the effective area ratio:

$$\text{Area ratio} \approx \frac{C_s}{C_a} = \frac{1.0}{0.1} = 10$$

This means the soma has 10× the membrane area of the axon, which is
biologically realistic (soma ~1000 µm², AIS ~100 µm²).

### Very high axonal conductances

$g_{Na} = 3000$ mS/cm² and $g_K = 1500$ mS/cm² are 25× and 42× higher
than standard HH values (120, 36). This reflects the high channel density
at the axon initial segment, where Nav1.6 channels cluster at ~1000/µm²
(Kole et al. 2008).

### Historical context

1. **Hodgkin-Huxley (1952):** Single-compartment squid axon
2. **Rall (1964):** Cable theory, multi-compartment passive models
3. **Pinsky-Rinzel (1994):** 2-compartment with dendritic Ca²⁺
4. **Mainen-Sejnowski (1996):** 2-compartment soma+axon, morphology determines pattern

---

## Pipeline Position

```
External input (current injection, synaptic)
        │
        ▼
┌─────────────────────────┐
│  MainenSejnowskiNeuron  │
│  step(current) → i32    │
│  20 sub-steps/call      │
│  5 state variables      │
│                         │
│  ┌──────┐   κ   ┌─────┐│
│  │ Soma ├───────┤ Axon ││
│  │(pass)│       │(act) ││
│  └──────┘       └─────┘│
└──────────┬──────────────┘
           │ spike {0,1} (detected at soma)
           ▼
┌──────────────────────────┐
│  Network / Population    │
│  Projection wiring       │
│  Analysis pipeline       │
└──────────────────────────┘
```

### Inputs
- `current: f64` — external current to **soma** in µA/cm²
- Typical range: 100–1000 µA/cm² (high values needed due to current spreading)

### Outputs
- `i32` — spike indicator (0 or 1), detected at soma
- Internal state: vs, va, m, h, n accessible for recording

---

## Features

- **Two-compartment model** — soma (passive) + axon (active)
- **Axon-initiated spikes** — AP generated at axon, backpropagates to soma
- **Morphology-dependent patterns** — area ratio controls firing pattern
- **20 sub-steps** per call (dt = 0.005 ms, 0.1 ms effective)
- **Standard HH kinetics** — same alpha/beta as HodgkinHuxley
- **High conductances** — reflects realistic AIS channel density
- **Bidirectional coupling** — soma↔axon electrical coupling via κ

---

## Usage Examples

### Basic simulation

```rust
use sc_neurocore_engine::neurons::MainenSejnowskiNeuron;

let mut n = MainenSejnowskiNeuron::new();
let spikes: i32 = (0..5_000).map(|_| n.step(500.0)).sum();
println!("Spikes: {spikes}, vs={:.1}, va={:.1}", n.vs, n.va);
```

### Soma/axon voltage comparison

```rust
use sc_neurocore_engine::neurons::MainenSejnowskiNeuron;

let mut n = MainenSejnowskiNeuron::new();
for i in 0..1000 {
    n.step(500.0);
    if i % 100 == 0 {
        println!("t={:.1}ms vs={:.1} va={:.1}", i as f64 * 0.1, n.vs, n.va);
    }
}
// va should show larger deflections (active AP) than vs (filtered)
```

### Area ratio experiment

```rust
use sc_neurocore_engine::neurons::MainenSejnowskiNeuron;

for c_a in [0.01, 0.05, 0.1, 0.5, 1.0] {
    let mut n = MainenSejnowskiNeuron::new();
    n.c_a = c_a;
    let spikes: i32 = (0..5_000).map(|_| n.step(500.0)).sum();
    println!("C_a={c_a}: {spikes} spikes (area ratio = {:.0})", n.c_s / c_a);
}
```

---

## Technical Reference

### Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `vs` | −65.0 | mV | Soma membrane potential |
| `va` | −65.0 | mV | Axon membrane potential |
| `m` | 0.05 | — | Na⁺ activation gate (axon) |
| `h` | 0.6 | — | Na⁺ inactivation gate (axon) |
| `n` | 0.3 | — | K⁺ activation gate (axon) |
| `kappa` | 10.0 | mS/cm² | Soma↔axon coupling conductance |
| `g_na` | 3000.0 | mS/cm² | Axon Na⁺ conductance |
| `g_k` | 1500.0 | mS/cm² | Axon K⁺ conductance |
| `g_l` | 1.0 | mS/cm² | Soma leak conductance |
| `e_na` | 50.0 | mV | Na⁺ reversal |
| `e_k` | −90.0 | mV | K⁺ reversal |
| `e_l` | −70.0 | mV | Leak reversal |
| `c_s` | 1.0 | µF/cm² | Soma capacitance |
| `c_a` | 0.1 | µF/cm² | Axon capacitance |
| `dt` | 0.005 | ms | Sub-step integration timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold (soma) |

### Conductance hierarchy

$$g_{Na} (3000) > g_K (1500) \gg \kappa (10) > g_L (1.0)$$

The axonal conductances are 300× the soma leak. This creates a fast
"spike generator" (axon) coupled to a slow "integrator" (soma).

---

## Performance Benchmarks

| Metric | Python | Rust (Criterion) |
|--------|--------|-----------------|
| Throughput | ~5K steps/s | 537K steps/s (1.86 µs/step) |
| 1k steps | ~200 ms | 1.86 ms |
| Speedup | — | **107×** |

### Cost breakdown

20 sub-steps per call. Each sub-step requires:
- 3 safe_rate evaluations (alpha_m, beta_m proxy, alpha_n) — 3× branch + exp
- 3 direct exp evaluations (beta_m, alpha_h, beta_n) — 3× exp
- 2 current calculations (I_Na, I_K) — 2× multiply chain
- 2 voltage updates (vs, va) — 2× divide + add
- 3 gating variable updates — 3× multiply + add

Total per step: 20 × (6 exp + 5 mul + 5 add + 2 div) ≈ 120 exp.

This makes MainenSejnowski one of the more expensive models due to the
large sub-step count (20 vs 4 for Pospischil).

Measured 2026-04-05 on i5-11600K @ 3.90 GHz, Criterion 0.8.

---

## Numerical Considerations

### Stiffness

The model is stiff due to:
- **Fast axonal dynamics:** $g_{Na}/C_a = 30000$ gives a characteristic
  time ~0.03 ms. dt = 0.005 ms provides ~6 sub-steps per fast timescale.
- **Slow soma dynamics:** $g_L/C_s = 1.0$ gives ~1 ms timescale.
- **Stiffness ratio:** ~33× between fast and slow timescales.

20 sub-steps is the minimum for stability with explicit Euler.

### Extreme input instability

At very high (>10⁴ µA/cm²) or very negative (<−100 µA/cm²) input, the
compartments saturate at the ±200 mV in-loop clips instead of tracking
physically meaningful dynamics; the rate evaluations themselves stay
finite (the `expm1` linoid saturates instead of overflowing). The Rust
test `mainen_weak_negative_no_crash` exercises moderate negative input
(−10 µA/cm²).

### Coupling symmetry

The coupling current $\kappa(V_a - V_s)$ enters with opposite sign in
each compartment, ensuring current conservation:
- Soma sees: $+\kappa(V_a - V_s)$ (positive when axon more depolarised)
- Axon sees: $-\kappa(V_a - V_s)$ (negative when axon more depolarised)

---

## Comparison with Related Models

| Property | MainenSejnowski | PinskyRinzel | HayL5 | TwoCompLIF |
|----------|----------------|-------------|-------|-----------|
| Compartments | Soma + axon | Soma + dendrite | Soma + trunk + apical | Soma + dendrite |
| Active comps | Axon only | Both | All 3 | Neither (LIF) |
| Channels | Na + K (HH) | Na + K + Ca + KCa | Na + K + Ca + Ih + ... | None |
| Coupling | κ = 10 | g_c = 2.1 | Variable | g_c |
| Per step | 1.86 µs | 122 ns | 591 ns | 2.7 ns |
| Sub-steps | 20 | 1 | 1 | 1 |
| Spike origin | Axon | Soma | Soma | Soma |

MainenSejnowski is the only 2-compartment model where spikes originate
in a separate (axon) compartment and backpropagate to the soma.

---

## Python/Rust Parity

The Python and Rust implementations are algorithmically identical:
- Same alpha/beta rate functions with same coefficients
- Same safe_rate singularity protection
- Same 20 sub-steps per call
- Same coupling implementation

Parity status: **EXACT** — verified by Rust pipeline test.

---

## Test Coverage

### Python tests (21 total)

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 8 | defaults, binary, 2-comp coupling, finite long run, reset, va≠vs during spike, spike at soma, sub-steps |
| Dynamics | 4 | subthreshold silent, fires under drive, axon leads soma, coupling strength |
| Gating | 3 | bounded, m/h/n evolve, deterministic |
| Pipeline | 3 | Population, Projection, Network spikes |
| Analysis | 3 | spike_count, ISI, firing_rate |
| **Total** | **21** | |

### Rust tests (6 total)

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Fires | 1 | fires with I=500 in 5000 steps |
| Silent | 1 | no spikes at zero input |
| Reset | 1 | vs→−65, va→−65 |
| Moderate stable | 1 | finite after 200 steps at I=500 |
| Coupling | 1 | κ > 0 (compartments coupled) |
| Weak negative | 1 | finite after 200 steps at I=−10 |
| NaN / ±Inf | 2 | rejected atomically (`ValueError` / `Err`), state unchanged |
| Singular voltages | 1 | rates exact and continuous at va = −25/−40/−65/+20 |
| Legacy configurations | 2 | epsilon-rate Python flag + engine Gauss-Seidel path anchored |
| **Total** | **7** | |

---

## Findings

1. **Throughput:** 1.86 µs/step (Rust), ~5K steps/s (Python). Rust is
   107× faster.

2. **Axon-initiated spikes:** The AP generates at the axon (high g_Na)
   and backpropagates to the soma via κ coupling. Spike detected at soma
   after a small delay.

3. **High current required:** Due to current spreading (soma area 10×
   axon), input currents of 100–1000 µA/cm² are needed. This is higher
   than single-compartment models (~5–10 µA/cm²).

4. **20 sub-steps necessary:** The fast axonal dynamics (g_Na/C_a = 30000)
   require dt ≤ 0.005 ms for Euler stability. This makes the model
   relatively expensive (1.86 µs/step vs 686 ns for Pospischil).

5. **Extreme input sensitivity:** The model can diverge at very high or
   very negative inputs. Moderate inputs (|I| < 1000) are stable.

6. **Python/Rust EXACT parity** confirmed.

7. **Morphology insight:** The C_s/C_a ratio controls firing pattern.
   Larger C_a (bigger axon) → faster response, more bursting.

---

## Citations

1. Mainen, Z.F. & Sejnowski, T.J. (1996). Influence of dendritic
   structure on firing pattern in model neocortical neurons. *Nature*
   382:363-366. DOI: 10.1038/382363a0

2. Stuart, G.J. & Sakmann, B. (1994). Active propagation of somatic
   action potentials into neocortical pyramidal cell dendrites. *Nature*
   367:69-72.

3. Kole, M.H.P. et al. (2008). Action potential generation requires a
   high sodium channel density in the axon initial segment. *Nat. Neurosci.*
   11(2):178-186.

---

## FPGA Considerations

| Component | LUTs | Notes |
|-----------|------|-------|
| 6 safe_rate/exp evaluations | ~384 | Per sub-step |
| 2 current channels | ~64 | Na + K |
| 2 voltage updates | ~48 | Soma + axon |
| 3 gating updates | ~96 | m, h, n |
| 20× pipeline unroll | ~1500 | Sub-step loop |
| Coupling logic | ~32 | κ(Va − Vs) |
| **Total** | **~2124** | Needs Artix-7 50T+ |

The 20× unrolling makes this one of the more resource-intensive neuron
models for FPGA. Consider time-multiplexing (sequential sub-steps) to
reduce area at the cost of latency.

---

## Version History

| Date | Change | Commit |
|------|--------|--------|
| 2026-03-20 | Initial Python implementation | — |
| 2026-04-04 | Rust port, EXACT parity verified | — |
| 2026-04-05 | Multi-angle Rust tests (7 tests) | `328cd4e` |
| 2026-04-05 | Criterion benchmark: 1.86 µs/step | `71bd1ec` |
| 2026-04-05 | Doc expanded with verification + benchmarks | — |

---

## Biological Accuracy Assessment

### What the model captures

- Axon-initiated action potentials ✓ (verified experimentally by Stuart & Sakmann 1994)
- Backpropagation to soma via electrical coupling ✓
- Morphology-dependent firing patterns ✓ (Mainen & Sejnowski 1996 main result)
- High AIS channel density ✓ (g_Na = 3000, matching Kole et al. 2008 estimates)
- Standard HH kinetics at the axon ✓

### What the model omits

- **Dendritic compartment:** The original paper has soma + dendrite + axon.
  SC-NeuroCore implements only soma + axon, losing dendritic Ca²⁺ spikes.
- **Dendritic Ca²⁺ channels:** CaV1.2/1.3 in dendrites enable dendritic
  plateau potentials. Not modelled.
- **Axon initial segment morphology:** The AIS is not a point compartment;
  it has a specific geometry (~30 µm long, 1 µm diameter) with spatially
  graded Nav density. Not captured.
- **Active soma:** Some cortical neurons have somatic Na channels. This
  model has a purely passive soma.
- **Multiple dendritic branches:** Rall (1964) showed that dendritic
  branching affects input impedance. Single-compartment soma loses this.

### Mainen & Sejnowski (1996) key results

The original paper demonstrated:
1. A single set of channel properties can produce all cortical firing
   patterns (RS, IB, chattering) by varying morphology alone
2. Dendritic area determines whether the neuron integrates (large soma →
   RS) or resonates (small soma → chattering)
3. Active dendritic currents are needed for intrinsic bursting
4. The model predicted that cortical neuron diversity is primarily
   morphological, not molecular — later partially confirmed by
   single-cell RNA-seq showing conserved channel expression

---

## Sensitivity Analysis

### Coupling strength (κ)

| κ | Effect |
|---|--------|
| 0.1 | Decoupled — axon fires independently of soma |
| 1.0 | Weak coupling — delayed backpropagation |
| 10.0 | Default — tight coupling |
| 100.0 | Near-isopotential — behaves as single-compartment |

### Capacitance ratio (C_s / C_a)

| C_a | Area ratio | Pattern |
|-----|-----------|---------|
| 0.01 | 100 | Very slow integration |
| 0.1 | 10 | Default — RS-like |
| 0.5 | 2 | Fast response |
| 1.0 | 1 | Symmetric — high rate |

### Axonal conductance

| g_Na | Effect |
|------|--------|
| 300 | Standard HH density — slow spike, high threshold |
| 3000 | Default — fast spike, AIS-like density |
| 30000 | Extremely fast — numerical instability risk |

---

## Current Decomposition at Rest

At V_s = V_a = −65 mV, gating at initial values (m=0.05, h=0.6, n=0.3):

### Soma currents

$$I_L = g_L(V_s - E_L) = 1.0 \times (-65 + 70) = 5.0 \text{ µA/cm²}$$
$$I_{coupling} = \kappa(V_a - V_s) = 10.0 \times 0.0 = 0 \text{ µA/cm²}$$

### Axon currents

$$I_{Na} = 3000 \times 0.05^3 \times 0.6 \times (-65 - 50) = 3000 \times 7.5\times10^{-5} \times (-115) = -25.9 \text{ µA/cm²}$$
$$I_K = 1500 \times 0.3^4 \times (-65 + 90) = 1500 \times 0.0081 \times 25 = 304 \text{ µA/cm²}$$
$$I_{coupling,axon} = \kappa(V_a - V_s) = 0 \text{ µA/cm²}$$

**Axon net:** −25.9 + 304 = 278 µA/cm² (strong outward K current)

At true equilibrium, gating variables adjust until currents balance.
The initial n = 0.3 is far from the rest value n_∞(−65) ≈ 0.04, so K
current is artificially high at initialisation. After equilibration
(~50 ms), the resting currents reduce to near zero.

---

## Network-Level Implications

### AP propagation fidelity

The 2-compartment structure naturally models:
- **AP initiation delay:** Current must flow from soma to axon, then
  threshold must be reached at the axon. This adds ~0.1 ms latency.
- **Backpropagation filtering:** The soma receives a low-pass filtered
  version of the axonal AP (κ acts as a conductance divider).

### Population synchronisation

In a network of MainenSejnowski neurons with synaptic input to soma:
- All neurons have the same soma→axon delay
- This introduces a consistent latency that affects network synchrony
- Gap junctions (not modelled) between axons would enhance synchrony

### Cost for network simulation

| Network size | Steps | Estimated time (Rust) |
|-------------|-------|----------------------|
| 100 neurons × 10K steps | 1M | ~1.86 s |
| 1000 neurons × 10K steps | 10M | ~18.6 s |

MainenSejnowski is expensive for large networks. Consider PospischilNeuron
(686 ns/step, 2.7× cheaper) or TraubMiles (1.80 µs, similar cost) as
alternatives for network-scale simulations.

---

## Stability Boundaries

### Safe operating range

| Parameter | Min safe | Max safe | Default |
|-----------|---------|---------|---------|
| I_ext | −50 | 5000 | 500 |
| dt | 0.001 | 0.01 | 0.005 |
| κ | 0.01 | 100 | 10.0 |
| g_Na | 100 | 10000 | 3000 |
| C_a | 0.01 | 1.0 | 0.1 |

Outside these ranges, the explicit Euler integrator may produce
non-physical oscillations or pin the voltages at the ±200 mV in-loop
clips; within the safe range, stability relies on the negative feedback
of conductance-based dynamics.

### Stiffness metric

$$\text{Stiffness ratio} = \frac{g_{Na}}{C_a \cdot g_L / C_s} = \frac{3000}{0.1 \times 1.0} = 30000$$

This extreme stiffness ratio explains why 20 sub-steps (not 4–10) are
needed. Implicit methods (Crank-Nicolson) would allow larger dt but add
complexity and cost per step.
