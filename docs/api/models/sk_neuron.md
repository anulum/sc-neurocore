# SKNeuron

**Module:** `engine/src/neurons/channels/sk.rs`
**Reference:** Stocker, *Nat Rev Neurosci* 5:758–770, 2004; Wang & Buzsáki, *J Neurosci* 16:6402–6413, 1996
**Family:** WB (Wang–Buzsáki) Na⁺/K⁺ base + SK (small conductance Ca²⁺-activated K⁺)
**State variables:** `v`, `h` (Na⁺ inactivation), `n` (K_dr activation), `ca` (intracellular Ca²⁺)

The model is a repository composite grounded in the Stocker (2004)
review: the threshold-reset event, the spike-triggered Ca²⁺ increment
(+0.2), and the Hill constants (n = 2, half-activation at Ca = 0.5)
are repository-specific specialisations, not a publication-exact
recurrence.

---

## Biological Context

### SK Channels: The Medium AHP

SK channels (KCa2.1, KCa2.2, KCa2.3, encoded by KCNN1-3) are a
family of K⁺ channels activated solely by intracellular Ca²⁺.  Unlike
BK channels (KCa1.1), SK channels have:

- **No voltage dependence:** activation is entirely determined by
  [Ca²⁺]ᵢ via constitutively bound calmodulin
- **Small single-channel conductance:** ~10 pS (vs ~250 pS for BK)
- **Slow kinetics:** activation/deactivation τ ≈ 5–15 ms, producing
  the medium afterhyperpolarisation (mAHP, 50–200 ms)

The mAHP is the dominant mechanism for **spike frequency adaptation**
in cortical pyramidal neurons, hippocampal CA1 neurons, and many
other cell types.  The sequence is:

1. Neuron fires action potentials
2. Each spike opens voltage-gated Ca²⁺ channels → Ca²⁺ entry
3. [Ca²⁺]ᵢ accumulates with each spike
4. Rising [Ca²⁺]ᵢ activates SK channels
5. SK outward current hyperpolarises the membrane
6. Interspike intervals lengthen → firing rate decreases
7. During pauses, [Ca²⁺]ᵢ decays → SK deactivates → firing resumes

### SK vs BK: Complementary Roles

| Property | SK (KCa2.x) | BK (KCa1.1) |
|----------|------------|-------------|
| Ca²⁺ dependence | Yes (Hill n≈2) | Yes (Hill n≈2) |
| Voltage dependence | **None** | Yes (Boltzmann) |
| Conductance | ~10 pS | ~250 pS |
| AHP component | mAHP (50–200 ms) | fAHP (<10 ms) |
| Time course | Slow (τ ~5–15 ms) | Fast (τ ~1 ms) |
| Ca²⁺ sensor | Calmodulin (CaM) | RCK domains |
| K_d for Ca²⁺ | ~0.3–0.7 µM | ~1–10 µM |
| Role | Adaptation, rhythmicity | Spike repolarisation |

### SK in Synaptic Plasticity

SK channels in dendritic spines act as a brake on NMDA receptor
activation (Faber, Delaney & Sah, 2005; Adelman, Maylie & Sah, 2012):

- Ca²⁺ entry through NMDA receptors activates spine SK channels
- SK hyperpolarisation limits further NMDA current (Mg²⁺ block)
- Blocking SK with apamin enhances LTP induction
- SK therefore acts as a negative feedback regulator of plasticity

This mechanism links intrinsic excitability to synaptic plasticity
and is implicated in learning and memory disorders.

### The Wang–Buzsáki (WB) Base Model

The SKNeuron uses the Wang–Buzsáki (1996) fast-spiking interneuron
model as its base:

- **INa:** m³h kinetics with instantaneous m (m_inf = α_m/(α_m+β_m))
- **IK:** n⁴ delayed rectifier
- **IL:** leak
- **φ = 5:** temperature factor accelerating gating kinetics

The WB model was originally designed for cortical fast-spiking
interneurons but serves here as a generic spiking base to which the
SK channel is added.

---

## Mathematical Analysis

### Membrane Equation

$$C_m \frac{dV}{dt} = -(I_{Na} + I_K + I_{SK} + I_L) + I_{ext}$$

### Base WB Currents

**Sodium (INa) — instantaneous m, dynamic h:**

$$I_{Na} = g_{Na} \cdot m_\infty^3(V) \cdot h \cdot (V - E_{Na})$$

$$m_\infty = \frac{\alpha_m}{\alpha_m + \beta_m}$$

$$\alpha_m = \frac{0.1(V + 35)}{1 - e^{-(V+35)/10}} \qquad \beta_m = 4 e^{-(V+60)/18}$$

$$\alpha_h = 0.07 e^{-(V+58)/20} \qquad \beta_h = \frac{1}{1 + e^{-(V+28)/10}}$$

$$\frac{dh}{dt} = \phi \cdot (\alpha_h(1-h) - \beta_h h)$$

**Delayed rectifier K⁺ (IK) — n⁴:**

$$I_K = g_K \cdot n^4 \cdot (V - E_K)$$

$$\alpha_n = \frac{0.01(V + 34)}{1 - e^{-(V+34)/10}} \qquad \beta_n = 0.125 e^{-(V+44)/80}$$

$$\frac{dn}{dt} = \phi \cdot (\alpha_n(1-n) - \beta_n n)$$

**Leak:**

$$I_L = g_L \cdot (V - E_L)$$

### SK Current

$$I_{SK} = g_{SK} \cdot sk_\infty([Ca^{2+}]) \cdot (V - E_K)$$

$$sk_\infty = \frac{[Ca^{2+}]^2}{[Ca^{2+}]^2 + K_d^2}$$

with K_d = 0.5 µM (since K_d² = 0.25 in the code).

The Hill coefficient n = 2 reflects the cooperative binding of Ca²⁺
to the two EF-hand domains on calmodulin (one per lobe), which is
constitutively associated with the SK channel.

**No voltage dependence:** sk_∞ depends only on [Ca²⁺]ᵢ.  This is
the defining property of SK channels — calmodulin senses Ca²⁺ and
directly gates the channel without any involvement of membrane
potential.

### Ca²⁺ Dynamics

$$\frac{d[Ca^{2+}]}{dt} = -\frac{[Ca^{2+}]}{\tau_{Ca}}$$

Between spikes, [Ca²⁺]ᵢ decays exponentially with τ_Ca = 150 ms.

**On each spike** (V crosses −20 mV):

$$[Ca^{2+}] \leftarrow [Ca^{2+}] + 0.2 \text{ µM}$$

The discrete Ca²⁺ increment models the brief Ca²⁺ entry through
voltage-gated Ca²⁺ channels during the action potential.  The
increment of 0.2 µM per spike is consistent with experimental
measurements of spine Ca²⁺ transients (~0.1–0.5 µM per AP in
hippocampal pyramidal neurons).

### Adaptation Mechanism

After N spikes in quick succession, [Ca²⁺]ᵢ ≈ 0.2N (if decay is
negligible during the burst).  The SK activation is:

$$sk_\infty(0.2N) = \frac{(0.2N)^2}{(0.2N)^2 + 0.25}$$

| Spikes (N) | [Ca²⁺] (µM) | sk_∞ | Effect |
|-----------|------------|------|--------|
| 0 | 0 | 0 | No SK current |
| 1 | 0.2 | 0.14 | Mild mAHP |
| 2 | 0.4 | 0.39 | Moderate mAHP |
| 3 | 0.6 | 0.59 | Strong mAHP |
| 5 | 1.0 | 0.80 | Near-maximal mAHP |
| 10 | 2.0 | 0.94 | Almost fully active |

This shows how SK produces progressively stronger adaptation:
the first spike barely activates SK, but by 5 spikes, the mAHP
is strong enough to substantially delay the next spike.

### Spike Frequency Adaptation

For constant input I, the instantaneous firing frequency f(t)
decreases over time as Ca²⁺ accumulates:

$$f(t) \approx f_0 - \Delta f \cdot (1 - e^{-t/\tau_{adapt}})$$

where f_0 is the initial rate (without SK), Δf is the adapted
reduction, and τ_adapt ≈ τ_Ca ≈ 150 ms.  The steady-state adapted
rate depends on the balance between Ca²⁺ entry per spike and decay:

$$[Ca]_{ss} = 0.2 \cdot f_{ss} \cdot \tau_{Ca}$$

At f_ss = 50 Hz: [Ca]_ss = 0.2 · 50 · 0.15 = 1.5 µM → sk_∞ ≈ 0.90.

---

## Parameters

| Parameter | Symbol | Type | Default | Units | Description |
|-----------|--------|------|---------|-------|-------------|
| `v` | V | State | −65.0 | mV | Membrane potential |
| `h` | h | State | 0.6 | — | Na⁺ inactivation |
| `n` | n | State | 0.32 | — | K_dr activation |
| `ca` | [Ca²⁺]ᵢ | State | 0.0 | µM | Intracellular calcium |
| `g_na` | g_Na | Param | 35.0 | mS/cm² | Na⁺ conductance |
| `g_k` | g_K | Param | 9.0 | mS/cm² | Delayed rectifier K⁺ |
| `g_sk` | g_SK | Param | 2.0 | mS/cm² | SK channel conductance |
| `g_l` | g_L | Param | 0.1 | mS/cm² | Leak conductance |
| `e_na` | E_Na | Param | 55.0 | mV | Na⁺ reversal |
| `e_k` | E_K | Param | −90.0 | mV | K⁺ reversal (shared by K_dr and SK) |
| `e_l` | E_L | Param | −65.0 | mV | Leak reversal |
| `c_m` | C_m | Param | 1.0 | µF/cm² | Membrane capacitance |
| `phi` | φ | Param | 5.0 | — | Temperature scaling factor |
| `tau_ca` | τ_Ca | Param | 150.0 | ms | Ca²⁺ decay time constant |
| `dt` | Δt | Step | 0.5 | ms | External time step |
| `v_threshold` | V_th | Thresh | −20.0 | mV | Spike threshold (with reset) |
| `gain` | g | Scale | 1.0 | — | Input current multiplier |

### Parameter Roles

**g_sk (2.0):** The SK conductance determines the strength of
adaptation.  Increasing g_SK from 0 to 5 mS/cm² progressively
slows steady-state firing rate.  At g_SK > ~8 mS/cm², tonic firing
is abolished entirely (SK is too strong, terminates all activity).

**tau_ca (150 ms):** The Ca²⁺ decay rate sets the timescale of
adaptation.  Shorter τ → faster adaptation recovery → less
frequency reduction.  Longer τ → sustained mAHP → stronger adaptation.
The default 150 ms matches experimental mAHP durations in hippocampal
CA1 neurons.

**phi (5.0):** The WB temperature factor speeds up gating kinetics
by 5×, enabling fast spiking.  Without φ, the WB model would fire
at ~10 Hz; with φ = 5, it can reach >100 Hz.

### Reset Mechanism

Unlike continuous HH-style models, the SKNeuron uses a **threshold
reset:** when V crosses −20 mV, V is reset to −65 mV and Ca²⁺ is
incremented by 0.2 µM.  This hybrid integrate-and-fire approach:

- Avoids the computational cost of simulating the full Na⁺ spike waveform
- Provides exact spike timing at the threshold crossing
- Models Ca²⁺ entry as a discrete event per spike

---

## Discrete-Time Implementation

### Sub-Stepping (50 sub-steps)

The WB α/β rate functions have fast dynamics (α_m at depolarised V
can exceed 10 ms⁻¹), requiring dt_sub = 0.01 ms for stability.

### Algorithm per Sub-Step

```
1. Compute WB rates at current V:
   α_m, β_m (Na activation — used only for m_inf)
   α_h, β_h (Na inactivation)
   α_n, β_n (K delayed rectifier)
2. Instantaneous Na activation:
   m_inf = α_m / (α_m + β_m)
3. SK activation (purely Ca²⁺-dependent):
   sk_inf = Ca² / (Ca² + 0.25)
4. Ca²⁺ decay:
   Ca += dt_sub · (-Ca / τ_Ca)
5. Gate updates:
   h += dt_sub · φ · (α_h(1-h) - β_h·h)
   n += dt_sub · φ · (α_n(1-n) - β_n·n)
6. Compute currents:
   I_Na = g_Na · m_inf³ · h · (V - E_Na)
   I_K = g_K · n⁴ · (V - E_K)
   I_SK = g_SK · sk_inf · (V - E_K)
   I_L = g_L · (V - E_L)
7. Update V:
   V += dt_sub · (-I_Na - I_K - I_SK - I_L + I_ext) / C_m
8. Spike check:
   If V ≥ V_th: fired = 1, V ← -65, Ca += 0.2
```

After all sub-steps: clamp V to [−100, 60], h and n to [0, 1], Ca ≥ 0,
and commit the candidate state.

### Invalid-Input Atomicity (Fail-Closed Contract)

`step(current)` validates before touching state and computes the whole
update on candidate values:

- A non-finite `current` (NaN, ±∞) raises `ValueError("current must be
  finite")` — no state field changes, and no spike is reported.
- A non-finite or out-of-bounds configuration or state field (see the
  descriptor ranges in `SKNeuron.toml`; `ca` must be finite and ≥ 0,
  with no artificial upper bound) raises `ValueError` at construction
  and again at each step if corrupted afterwards.
- A candidate state that becomes non-finite mid-integration raises
  `ValueError("SK candidate state became non-finite")` and the
  pre-step state is preserved exactly.

The production Rust engine (`try_step`), the PyO3 binding (typed
`ValueError`), the standalone safety Rust, Go (`TryStep`), and Julia
(`ArgumentError`) surfaces enforce the same contract; the engine's
`step` and Go's `Step` legacy wrappers fail closed by returning 0
without mutating state.

---

## Numerical Examples

### Example 1: Adapted Firing (I_ext = 2)

Initial: V = −65, h = 0.6, n = 0.32, Ca = 0

**Spike 1 (t ≈ 3 ms):** V reaches −20, resets. Ca → 0.2 µM.
sk_∞ = 0.04/(0.04+0.25) = 0.14. ISK = 2·0.14·(−65+90) = 7 nA.
Mild hyperpolarisation.

**Spike 2 (t ≈ 8 ms):** Ca ≈ 0.19 + 0.2 = 0.39 (partial decay).
sk_∞ ≈ 0.38. ISK ≈ 19 nA. Moderate hyperpolarisation → ISI slightly longer.

**Spike 5 (t ≈ 30 ms):** Ca ≈ 0.8 µM. sk_∞ ≈ 0.72.
ISK ≈ 36 nA → substantial mAHP → ISI ~10 ms (vs ~5 ms initially).

**Steady state (t > 500 ms):** Ca ≈ 1.2 µM, sk_∞ ≈ 0.85.
Adapted firing rate ~30–40 Hz (vs ~100 Hz without SK).

### Example 2: No Adaptation (g_SK = 0)

Without SK, the WB model fires at a constant rate determined by
I_ext.  At I = 2: ~80–100 Hz with regular ISIs.  No Ca²⁺
accumulation occurs (Ca stays at 0 since spikes add Ca but there
is no SK to read it — the Ca still accumulates and decays but has
no effect on V).

### Example 3: Strong SK (g_SK = 5)

With stronger SK, adaptation is more pronounced:
- Initial rate still ~100 Hz (Ca ≈ 0 initially)
- After 3 spikes: Ca ≈ 0.6, sk_∞ ≈ 0.59, ISK ≈ 74 nA
- Firing effectively pauses for ~200 ms until Ca decays
- Resulting pattern: burst of 3–5 spikes → 200 ms pause → repeat
- This creates **intrinsic bursting** from the SK feedback loop

---

## Analytical Properties

### Adaptation Index

The adaptation index (AI) quantifies how much the ISI increases:

$$AI = \frac{ISI_{last} - ISI_{first}}{ISI_{last} + ISI_{first}}$$

For the SKNeuron at I = 2, g_SK = 2: AI ≈ 0.4–0.6, indicating
moderate adaptation.  AI = 0 means no adaptation; AI = 1 means
the neuron stops firing.

### f-I Curve (Adapted)

The steady-state f-I curve with SK is:

$$f_{ss}(I) < f_0(I)$$

where f_0 is the unadapted (WB-only) rate.  The reduction increases
with I because higher rates produce more Ca²⁺:

| I (nA) | f_0 (Hz) | f_ss (Hz) | Reduction |
|--------|---------|----------|-----------|
| 0.5 | ~30 | ~25 | 17% |
| 1.0 | ~60 | ~40 | 33% |
| 2.0 | ~100 | ~45 | 55% |
| 5.0 | ~200 | ~60 | 70% |

The adapted f-I curve has a much shallower slope — SK compresses
the dynamic range, which helps prevent saturation in neural circuits.

### SK as a High-Pass Filter

Because SK is activated by accumulated Ca²⁺ (which integrates spikes
over τ_Ca ≈ 150 ms), the neuron preferentially transmits transient
inputs over sustained ones.  This creates an effective high-pass
filter on the input:

- **Transient input (< 150 ms):** SK barely activates → full spike output
- **Sustained input (> 150 ms):** SK reaches steady state → adapted
  (reduced) spike output
- **Cutoff frequency:** f_c ≈ 1/(2π·τ_Ca) ≈ 1 Hz

---

### Apamin Sensitivity and Channel Subtypes

The three SK subtypes have different apamin sensitivities:

| Subtype | Gene | K_d apamin | Expression |
|---------|------|-----------|-----------|
| SK1 (KCa2.1) | KCNN1 | ~10 nM | Cortex, hippocampus |
| SK2 (KCa2.2) | KCNN2 | ~40 pM | Hippocampus CA1 (dominant) |
| SK3 (KCa2.3) | KCNN3 | ~1 nM | Monoaminergic nuclei |

The model does not distinguish subtypes — the single g_SK parameter
represents the total SK conductance.  Apamin block is modelled by
setting g_SK = 0.

### Calcium Dynamics: Compartmental Considerations

The model uses a single-compartment Ca²⁺ model (whole-cell [Ca²���]).
In reality, Ca²⁺ microdomains near the membrane are much higher
(10–100 µM) than bulk cytosolic Ca²⁺ (0.1–1 µM).  SK channels
are activated by the nanodomain Ca²⁺ near voltage-gated Ca²⁺
channels, not the average cytosolic level.

The model captures this implicitly through the spike-triggered
increment (0.2 µM) which represents the effective Ca²⁺ "seen" by
SK channels during an action potential, not the true whole-cell
average.

### Bifurcation with g_SK

As g_SK increases from 0, the system transitions:
- g_SK = 0: tonic firing (WB model)
- g_SK = 1–3: adapted firing (decreasing rate)
- g_SK = 3–6: bursting (SK-mediated pauses between burst clusters)
- g_SK > 8: quiescent (SK too strong, suppresses all activity)

This sequence mirrors experimental observations when SK conductance
is pharmacologically modulated (e.g., progressive 1-EBIO application).

### Comparison with Other Adaptation Mechanisms

| Mechanism | Timescale | Molecular basis | Model |
|-----------|----------|----------------|-------|
| Na⁺ inactivation | 1–5 ms | Nav channel h gate | All HH models |
| M-current (Kv7) | 50–200 ms | Voltage-gated KCNQ | Kv7 models |
| **SK (this model)** | **50–200 ms** | **Ca²⁺-gated KCa2** | **SKNeuron** |
| AHP (slow, sIAHP) | 1–5 s | Unknown (KCNQ3?) | Not implemented |
| Na⁺/K⁺ pump | 1–10 s | Electrogenic pump | Metabolic models |

SK provides the dominant medium-timescale adaptation in most cortical
and hippocampal neurons.

---

## FPGA Implementation Estimates

### Resource Requirements (Zynq-7020, XC7Z020)

| Resource | Per neuron | Available | Max neurons |
|----------|-----------|-----------|-------------|
| LUT | ~120 | 53,200 | ~443 |
| FF | ~128 | 106,400 | ~831 |
| DSP48E1 | 5 | 220 | 44 |
| BRAM (18Kb) | 0 | 280 | N/A |

**Breakdown:**
- WB α/β rates (4 exp functions): ~60 LUT
- m_inf³ computation: 1 DSP
- n⁴ computation: 1 DSP
- SK Hill function: 1 DSP
- 4 current sums: ~20 LUT
- Ca²⁺ decay + increment: 1 DSP
- V update: 1 DSP
- State registers (V, h, n, Ca × 32-bit): ~128 FF
- Threshold comparator + reset: ~10 LUT
- Control + sub-step: ~30 LUT

### Fixed-Point Precision

**Q16.16 recommended:**
- V range [−100, 60]: 8 integer bits
- g_Na = 35: 6 integer bits
- Ca range [0, ~5]: 3 integer bits
- Exponentials in α/β: need careful range management

### Timing

At 100 MHz with 50 sub-steps:
- Per sub-step: ~10 cycles
- Total: 50 × 10 = 500 cycles = 5.0 µs
- Benchmark: CPU 2.79 µs/step → FPGA comparable single-neuron,
  but 443 in parallel → effective ~11.3 ns/neuron/step

---

## Validation

### Functional Checks

| Property | Expected | Measured | Status |
|----------|----------|---------|--------|
| Fires with I = 2 | Sustained spiking | Confirmed | ✅ |
| Silent at I = 0 | No spikes | Confirmed | ✅ |
| Spike frequency adaptation | Early ISI < late ISI | Confirmed | ✅ |
| SK inactive at Ca = 0 | sk_inf < 0.001 | 0.0 | ✅ |
| Removing g_SK increases rate | Monotonic | Confirmed | ✅ |
| Ca²⁺ ≥ 0 | Always | Confirmed | ✅ |
| V clamped [−100, 60] | Always | 10⁶ steps | ✅ |
| Non-finite input rejected atomically | `ValueError`, state unchanged | Confirmed | ✅ |
| Invalid configuration rejected atomically | `ValueError`, state unchanged | Confirmed | ✅ |
| Reset clears state | V=−65, Ca=0 | Confirmed | ✅ |
| mAHP duration ~50–200 ms | Matches | ~150 ms at default | ✅ |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels/sk.rs` (`try_step`, atomic rejection) |
| PyO3 wrapper | `engine/src/bindings/channels/sk_neuron.rs` (typed `ValueError`; state: v, h, n, ca) |
| NetworkRunner wired | `NeuronVariant::SK` (`engine/src/network_runner/model_factory.rs`) |
| `create_neuron("SK")` | Yes |
| `supported_models()` | Includes "SK" |
| Standalone safety Rust | `src/sc_neurocore/accel/rust/safety/sk_neuron.rs` (full recurrence) |
| Go service | `src/sc_neurocore/accel/go/services/sk_neuron.go` (`TryStep`, full recurrence) |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/sk_neuron.jl` (atomic `ArgumentError`) |
| Mojo | not implemented; no kernel exists and no parity is claimed |
| Silicon / RTL | not implemented; no HDL parity claimed |
| Module-owned tests | `tests/test_model_sk_neuron.py`, `tests/test_sk_neuron_backends.py` |
| Backend parity | Rust engine, safety Rust, Go, Julia vs Python: 64-step complete state ≤ 1e-12 |
| Benchmark | `sk_1k_steps`: **2.79 ms** (2.79 µs/step), i5-11600K |

---

## Network Coupling

### SK and Network Oscillations

In networks of excitatory neurons with SK, the adaptation creates
population-level dynamics:

- **Theta oscillations (4–8 Hz):** SK-mediated adaptation produces
  burst-pause patterns at ~5 Hz in hippocampal pyramidal cells
- **Adaptation-induced synchrony:** adapted neurons tend to fire in
  synchronised bursts, as the collective pause from SK allows recovery
  before the next burst
- **Gain modulation:** SK reduces the gain of the f-I curve, allowing
  modulatory systems (e.g., cholinergic) to control excitability by
  regulating SK channels (muscarinic receptors inhibit SK via PKC)

### Cholinergic Modulation

Acetylcholine (ACh) acting on muscarinic M1 receptors activates PKC,
which phosphorylates calmodulin and reduces its Ca²⁺ affinity.  This
effectively increases K_d for SK channels, reducing the mAHP.  In the
model, this is equivalent to increasing kd_kca (shifting the Hill
function rightward).

Cholinergic modulation of SK is a key mechanism for:
- Attentional gating (ACh from basal forebrain → cortex)
- Memory encoding (ACh suppresses mAHP → higher firing rates → LTP)
- Sleep–wake transitions (cholinergic tone modulates SK throughout cortex)

### Pharmacology

| Agent | Effect on SK | Network consequence |
|-------|-------------|-------------------|
| Apamin (bee venom) | Blocks SK | Enhanced excitability, impaired LTP regulation |
| 1-EBIO | Opens SK | Reduced firing, enhanced adaptation |
| NS309 | Positive modulator | Anticonvulsant-like effects |
| UCL1684 | Blocks SK | Epileptiform activity |

---

## References

1. Stocker, M. (2004). Ca²⁺-activated K⁺ channels: molecular
   determinants and function of the SK family. *Nat Rev Neurosci*,
   5(10), 758–770.

2. Wang, X. J. & Buzsáki, G. (1996). Gamma oscillation by synaptic
   inhibition in a hippocampal interneuronal network model. *J Neurosci*,
   16(20), 6402–6413.

3. Faber, E. S. L., Delaney, A. J. & Sah, P. (2005). SK channels
   regulate excitatory synaptic transmission and plasticity in the
   lateral amygdala. *Nat Neurosci*, 8(5), 635–641.

4. Adelman, J. P., Maylie, J. & Sah, P. (2012). Small-conductance
   Ca²⁺-activated K⁺ channels: form and function. *Annu Rev Physiol*,
   74, 245–269.

5. Xia, X. M., Fakler, B., Rivard, A., et al. (1998). Mechanism of
   calcium gating in small-conductance calcium-activated potassium
   channels. *Nature*, 395, 503–507.

6. Bond, C. T., Herson, P. S., Strassmaier, T., Hammond, R.,
   Stackman, R., Maylie, J. & Adelman, J. P. (2004).
   Small conductance Ca²⁺-activated K⁺ channel knock-out mice reveal
   the identity of calcium-dependent afterhyperpolarization currents.
   *J Neurosci*, 24(23), 5301–5306.

7. Madison, D. V. & Nicoll, R. A. (1984). Control of the repetitive
   discharge of rat CA1 pyramidal neurones in vitro. *J Physiol*,
   354, 319–331.

8. Pedarzani, P. & Stocker, M. (2008). Molecular and cellular basis of
   small- and intermediate-conductance, calcium-activated potassium
   channel function in the brain. *Cell Mol Life Sci*, 65(20),
   3196–3217.

9. Benda, J. & Herz, A. V. M. (2003). A universal model for
   spike-frequency adaptation. *Neural Computation*, 15(11),
   2523–2564.

10. Ha, G. E. & Cheong, E. (2017). Spike frequency adaptation in
    neurons of the central nervous system. *Exp Neurobiol*, 26(4),
    179–185.

11. Engel, J., Schultens, H. A. & Schild, D. (1999). Small conductance
    potassium channels cause an activity-dependent spike frequency
    adaptation and make the transfer function of neurons logarithmic.
    *Biophys J*, 76(3), 1310–1319.
