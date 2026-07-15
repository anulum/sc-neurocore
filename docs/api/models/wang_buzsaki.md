# WangBuzsakiNeuron

**Module:** `sc_neurocore.neurons.models.wang_buzsaki`
**Reference:** Wang & Buzsáki, J. Neurosci. 16(20), 1996
**Family:** Biophysical conductance-based (fast-spiking interneuron)
**State variables:** `v` (membrane potential), `h` (Na⁺ inactivation), `n` (K⁺ activation)

---

## 1. Mathematical Formalism

### Membrane potential

$$C_m \frac{dV}{dt} = -g_{Na}\, m_\infty^3\, h\,(V - E_{Na}) - g_K\, n^4\,(V - E_K) - g_L\,(V - E_L) + I_{ext}$$

where $C_m = 1\,\mu\text{F/cm}^2$.

### Gating variables

$$\frac{dh}{dt} = \phi\,[\alpha_h(V)(1 - h) - \beta_h(V) \cdot h]$$

$$\frac{dn}{dt} = \phi\,[\alpha_n(V)(1 - n) - \beta_n(V) \cdot n]$$

**Key simplification:** m is **instantaneous** — an algebraic function,
not a differential equation:

$$m_\infty(V) = \frac{\alpha_m(V)}{\alpha_m(V) + \beta_m(V)}$$

This eliminates one ODE (3 instead of HH's 4), reflecting the fact that
Na⁺ activation in fast-spiking interneurons is so rapid that it can be
treated as instantaneous on the timescale of network dynamics.

### Rate functions

| Rate | Formula | Singularity |
|------|---------|-------------|
| $\alpha_m$ | $\frac{0.1(V+35)}{1 - \exp(-(V+35)/10)}$ | V=−35: returns 1.0 |
| $\beta_m$ | $4 \exp(-(V+60)/18)$ | — |
| $\alpha_h$ | $0.07 \exp(-(V+58)/20)$ | — |
| $\beta_h$ | $\frac{1}{1 + \exp(-(V+28)/10)}$ | — |
| $\alpha_n$ | $\frac{0.01(V+34)}{1 - \exp(-(V+34)/10)}$ | V=−34: returns 0.1 |
| $\beta_n$ | $0.125 \exp(-(V+44)/80)$ | — |

Rate functions are shifted from HH originals (V+35 vs V+40 for $\alpha_m$,
V+58 vs V+65 for $\alpha_h$, V+34 vs V+55 for $\alpha_n$). These shifts
are from Wang & Buzsáki 1996, Table 1.

### Phi factor

The $\phi = 5$ parameter accelerates h and n gating by 5×. This makes
the model fire faster than standard HH — matching the ~40 Hz gamma
oscillation frequency of parvalbumin-positive (PV+) basket cells.

### Integration

Forward Euler with **50 sub-steps** per call ($\text{int}(0.5 / \text{dt}) = 50$).
Each call integrates 0.5 ms of biological time. The sub-step count adapts
to dt: $\text{int}(0.5 / \max(\text{dt}, 0.001))$.

### Spike detection

$$V \geq V_\text{threshold}(-20\,\text{mV}) \;\text{AND}\; V_\text{prev} < V_\text{threshold}$$

---

## 2. Theoretical Context

### Historical background

Wang & Buzsáki (1996) published "Gamma oscillation by synaptic
inhibition in a hippocampal interneuronal network model" in the
Journal of Neuroscience. The paper demonstrated that networks of
fast-spiking (FS) inhibitory interneurons, connected solely via
mutual GABAergic inhibition, can generate coherent gamma-frequency
(30–80 Hz) oscillations — the "interneuron network gamma" (ING)
mechanism.

The work addressed a fundamental question in systems neuroscience:
how do populations of purely inhibitory neurons synchronise? The
answer was that the interaction between intrinsic spike dynamics
($\phi$-accelerated gating) and synaptic inhibition (GABA_A with
5–10 ms decay) creates a natural oscillation period in the gamma
band. This required a neuron model that captures the essential
fast-spiking phenotype with minimal complexity.

### Fast-spiking interneuron phenotype

Parvalbumin-positive (PV+) basket cells in the hippocampus and
neocortex are characterised by:

- **Narrow action potentials** (half-width < 0.5 ms)
- **High firing rates** (sustained > 200 Hz)
- **No spike-frequency adaptation** (constant ISI)
- **Deep, fast afterhyperpolarisation**
- **Low input resistance** (rapid membrane time constant)

The Wang-Buzsáki model captures these features through the
$m_\infty$ approximation (eliminates the slow m ODE) and the
$\phi = 5$ acceleration (compresses h/n time constants by 5×).

### Excitability classification

The model exhibits Type-I excitability: smooth frequency onset near
threshold with a continuous f-I curve. The $m_\infty$ approximation
removes the mechanism for depolarisation block that occurs in the
full HH model at very high currents.

### ING vs PING mechanisms

Two distinct mechanisms generate gamma oscillations in cortical
networks:

| Mechanism | Description | Neuron types |
|-----------|-------------|-------------|
| **ING** | Interneuron Network Gamma | I→I only |
| **PING** | Pyramidal-Interneuron Network Gamma | E→I→E cycle |

The Wang-Buzsáki model was designed for ING: mutual inhibition
between FS cells creates synchronised gamma. In SC-NeuroCore,
combining WangBuzsaki (I) with TraubMiles or HodgkinHuxley (E)
populations enables PING architectures.

### Model family

| Model | Distinguishing feature |
|-------|----------------------|
| HodgkinHuxley | Original squid axon, 4 ODEs, all gates integrated |
| WangBuzsaki | FS interneuron, 3 ODEs, m instantaneous, φ=5 |
| TraubMiles | CA3 pyramidal, 4 ODEs, shifted rates, 10 sub-steps |
| ConnorStevens | A-current, 6 state vars, type-I excitability |
| PrescottNeuron | 2D reduction with adaptation (w), excitability switching |

---

## 3. Pipeline Position

```text
Input → Population(WangBuzsakiNeuron, n) → Projection → Network → Monitor
  ↑         ↓
  I_ext   step() → {0,1}
```

### Layer assignment

WangBuzsaki neurons serve as the **default fast-spiking inhibitory
interneuron** in SC-NeuroCore networks:

- **Feedforward inhibition:** Receives excitatory drive and provides
  rapid inhibition to principal cells
- **Lateral inhibition:** Within-layer competition (winner-take-all)
- **ING oscillation:** Mutual I→I inhibition generates gamma
- **Cross-inhibition:** Between competing decision populations

### Model zoo integration

Most-used inhibitory model — appears in 4 of 10 pre-configured
architectures:

| Architecture | Role | Count |
|-------------|------|-------|
| `decision_making_circuit` | Shared inhibitory pool | 1 population |
| `working_memory_circuit` | Uniform inhibition | 1 population |
| `auditory_processing` | Onset detection | 1 population |
| `visual_cortex_v1` | Complex cells | n_orientation pops |

### NetworkRunner compatibility

The `WangBuzsakiNeuron` has the standard `step(f64) → i32` signature
and is directly compatible with NetworkRunner. No wrapper macros needed.

### Analysis integration

All SC-NeuroCore analysis functions work with this model's spike output:
- `spike_count(monitor)` — total spikes
- `isi(monitor)` — inter-spike intervals (verifies gamma band)
- `firing_rate(monitor)` — mean rate across time bins

---

## 4. Features

### Instantaneous m simplification

By setting $\tau_m \to 0$ (instantaneous activation), the WB model
reduces from 4 ODEs (HH) to 3 ODEs. This is valid because:
- Na⁺ activation (m) is the fastest process in the HH model ($\tau_m \approx 0.1$ ms)
- On the 0.5 ms timescale of interneuron dynamics, m has reached
  steady state before h or n change appreciably
- The remaining h and n dynamics (accelerated by $\phi=5$) capture
  the essential spike shape

### Phi acceleration

$\phi=5$ makes h and n dynamics 5× faster than standard HH:
- $\phi=1$: slow recovery, lower maximum rate, broader spikes
- $\phi=5$: fast recovery, high maximum rate, narrow spikes

Verified by test: after 100 steps at I=1.0, $|\Delta h|$ with $\phi=5$
exceeds $|\Delta h|$ with $\phi=1$.

### Gamma frequency band

At moderate drive (I ≈ 0.5–1.0), the model fires in the gamma band
(30–80 Hz). This is by design — Wang & Buzsáki 1996 tuned the
parameters to reproduce the frequency of PV+ basket cell oscillations
observed in hippocampal slices.

### ISI regularity

CV(ISI) < 0.05 at I=1.0 — extremely regular firing. This contrasts
with HH (CV ≈ 0.26) and reflects the simplified dynamics: with m
instantaneous, the spike-generating feedback is faster and more
deterministic.

### f-I curve

| Current | Regime | Frequency |
|---------|--------|-----------|
| 0.0 | Resting | 0 Hz |
| 0.5 | Low gamma | ~30–40 Hz |
| 1.0 | High gamma | ~50–70 Hz |
| 2.0 | Fast spiking | ~80–120 Hz |
| 5.0 | Very fast | ~150+ Hz |
| 10.0 | Maximum rate | ~200+ Hz |

Monotonic f-I curve — no depolarisation block observed in the
tested range.

### Conductance comparison with HH

| Parameter | HH (1952) | WB (1996) | Ratio |
|-----------|-----------|-----------|-------|
| $g_{Na}$ | 120 | 35 | 0.29× |
| $g_K$ | 36 | 9 | 0.25× |
| $g_L$ | 0.3 | 0.1 | 0.33× |
| $E_{Na}$ | +50 | +55 | — |
| $E_K$ | −77 | −90 | — |
| $E_L$ | −54.4 | −65.0 | — |
| $\phi$ | 1 | 5 | 5× |

The WB model has ~3× lower conductances but 5× faster gating — the
net effect is faster, sharper spikes characteristic of cortical
fast-spiking interneurons.

### Reversal potential ordering

$$E_K (-90) < E_L (-65) < V_\text{threshold} (-20) < E_{Na} (55)$$

The wider $E_K$−$E_L$ gap (25 mV vs 23 mV in HH) produces a deeper
afterhyperpolarisation — characteristic of fast-spiking cells that
need rapid recovery for high-frequency firing.

---

## 5. Usage Examples

### Example 1: Gamma-band spiking measurement

```python
from sc_neurocore.neurons.models.wang_buzsaki import WangBuzsakiNeuron

neuron = WangBuzsakiNeuron()
spike_times = []

for t in range(40000):  # 20 seconds at 0.5 ms/step
    spike = neuron.step(1.0)  # moderate drive
    if spike:
        spike_times.append(t * 0.5)  # ms

if len(spike_times) > 1:
    isis = [
        spike_times[i + 1] - spike_times[i]
        for i in range(len(spike_times) - 1)
    ]
    mean_isi = sum(isis) / len(isis)
    freq = 1000.0 / mean_isi  # Hz
    print(f"Frequency: {freq:.1f} Hz (gamma: 30-80 Hz)")
    cv = (
        (sum((x - mean_isi) ** 2 for x in isis) / len(isis)) ** 0.5
        / mean_isi
    )
    print(f"CV(ISI): {cv:.4f} (expect < 0.05)")
```

### Example 2: ING (Interneuron Network Gamma) circuit

```python
from sc_neurocore.network import Network, Population, Projection
from sc_neurocore.neurons.models.wang_buzsaki import WangBuzsakiNeuron
from sc_neurocore.input import PoissonInput
from sc_neurocore.monitors import SpikeMonitor
from sc_neurocore.analysis import spike_count, firing_rate

# FS interneuron population
fs_pop = Population(WangBuzsakiNeuron, n=20)

# Mutual inhibition (ING mechanism)
mutual_inh = Projection(
    source=fs_pop, target=fs_pop,
    weight=-2.0,  # inhibitory
    probability=0.5,
)

# Tonic excitatory drive
drive = PoissonInput(rate=500.0, weight=2.0, dt=0.001, seed=42)

net = Network()
net.add_population("fs", fs_pop)
net.add_projection("inh", mutual_inh)
net.add_input("drive", drive, target="fs")

mon = SpikeMonitor()
net.add_monitor("spikes", mon, source="fs")

net.run(duration=1.0)

total = spike_count(mon)
rate = firing_rate(mon, duration=1.0)
print(f"Total spikes: {total}, Mean rate: {rate:.1f} Hz")
```

### Example 3: Phi parameter sweep

```python
from sc_neurocore.neurons.models.wang_buzsaki import WangBuzsakiNeuron

for phi_val in [1.0, 2.0, 5.0, 10.0]:
    n = WangBuzsakiNeuron()
    n.phi = phi_val
    spikes = sum(n.step(1.0) for _ in range(20000))
    freq = spikes / (20000 * 0.5e-3)  # Hz
    print(f"phi={phi_val:4.1f}: {spikes:4d} spikes, {freq:.0f} Hz")
```

---

## 6. Technical Reference

### Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane potential (initial) |
| `h` | 0.8 | — | Na⁺ inactivation gate |
| `n` | 0.1 | — | K⁺ activation gate |
| `g_na` | 35.0 | mS/cm² | Peak Na⁺ conductance |
| `g_k` | 9.0 | mS/cm² | Peak K⁺ conductance |
| `g_l` | 0.1 | mS/cm² | Leak conductance |
| `e_na` | 55.0 | mV | Na⁺ reversal potential |
| `e_k` | −90.0 | mV | K⁺ reversal potential |
| `e_l` | −65.0 | mV | Leak reversal potential |
| `c_m` | 1.0 | µF/cm² | Membrane capacitance |
| `phi` | 5.0 | — | Gating acceleration factor |
| `dt` | 0.01 | ms | Sub-step timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

### Rust parity

| Aspect | Python | Rust | Status |
|--------|--------|------|--------|
| State variables | v, h, n | v, h, n | **EXACT** |
| m computation | $m_\infty = \alpha_m / (\alpha_m + \beta_m)$ | same | **EXACT** |
| m integration | algebraic (no ODE) | algebraic (no ODE) | **EXACT** |
| Rate functions | α/β with singularity guards | `safe_rate()` helper | **EXACT** |
| Singularity threshold | 1e-6 | 1e-7 | Functionally equivalent |
| Sub-steps | `int(0.5/dt)` = 50 | `(0.5/dt) as usize` = 50 | **EXACT** |
| Phi acceleration | φ·(α(1-x) − β·x)·dt | same | **EXACT** |
| Spike detection | threshold crossing | threshold crossing | **EXACT** |
| Current powers | m_inf**3, n**4 | `.powi(3)`, `.powi(4)` | **EXACT** |

**No parity defects found.** Python and Rust produce numerically
equivalent spike trains.

### NetworkRunner integration

Direct compatibility — no wrapper macros needed.
Signature: `step(current: f64) → i32`.

### Source files

| File | Lines | Description |
|------|-------|-------------|
| `src/sc_neurocore/neurons/models/wang_buzsaki.py` | 71 | Python reference |
| `engine/src/neurons/biophysical/wang_buzsaki.rs` | (bounded) | Rust implementation |
| `tests/test_model_wang_buzsaki.py` | ~250 | 19 tests |

### Numerical considerations

- **50 sub-steps:** int(0.5/0.01) = 50 sub-steps per call. Each call
  integrates 0.5 ms of biological time.
- **dt stability:** Tested at dt = 0.005, 0.01, 0.02. All produce
  finite states after 10,000 steps.
- **Singularity protection:** $\alpha_m$ at V=−35 and $\alpha_n$ at
  V=−34 have removable singularities, handled with $|d| > 10^{-6}$
  guards (Python) and $|d| < 10^{-7}$ (Rust).
- **m_inf per sub-step:** m_inf is recomputed from V each sub-step —
  no accumulation error from Euler integration of m.
- **5 exp() per sub-step:** $\beta_m$, $\alpha_h$, $\beta_h$,
  $\alpha_n$, $\beta_n$ — totalling 250 exp() per call.

---

## 7. Performance Benchmarks

### Criterion benchmarks (local i5-11600K, measured 2026-04-05)

| Metric | Value |
|--------|-------|
| Test | `wang_buzsaki_1k_steps` (1,000 `step(2.0)` calls) |
| Median | 7,003 µs (7.0 ms) |
| Per-step | 7.0 µs |
| Throughput | ~143K steps/s |

### Python baseline (measured 2026-04-04)

| Metric | Value |
|--------|-------|
| Isolation | ~922 steps/s |
| Network (10 neurons, 1 s) | ~400 neuron-steps/s |
| Spikes (10K steps, I=5.0) | 951 |

### Rust speedup estimate

The Rust implementation processes ~143,000 steps/s vs Python's ~922
steps/s — approximately **155× speedup**.

### Comparison with other biophysical models

| Model | Criterion (1K steps) | Sub-steps | exp() per step |
|-------|---------------------|-----------|----------------|
| DestexheThalamic | 0.53 ms | 5 | ~35 |
| TraubMiles | 1.6 ms | 10 | ~60 |
| WangBuzsaki | 7.0 ms | 50 | ~250 |
| HodgkinHuxley | 11.2 ms | 100 | ~400 |

The WB model is between TraubMiles and HH in speed. Its 50 sub-steps
(vs 10 for TraubMiles) account for the 4.4× ratio. The per-exp()
cost is consistent across all biophysical models.

---

## 8. Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary output, 3-var evolution, finite (20K), reset |
| Gamma frequency | 2 | gamma band at I=1.0 (30–100 Hz), onset frequency near 30 Hz |
| f-I curve | 3 | subthreshold silent, monotonic (4-point), fast spiking at I=10 |
| HH properties | 5 | m instantaneous, phi accelerates gating, gating bounded, ISI regularity (CV<0.05), singularity protection |
| Parameters | 2 | dt stability (3 values), deterministic |
| Pipeline | 2 | Population construction, Network + monitoring |
| **Total** | **19** | **ALL PASSED** |

### Rust tests (engine)

| Test | What is verified |
|------|-----------------|
| `wb_silent_without_input` | No spikes at I=0 |
| `wb_reset_clears_state` | Reset restores defaults |
| `wb_extreme_bounded` | V finite at I=10⁴ |
| `wb_fires_with_drive` | Spikes at I=2 |
| `wb_negative_no_crash` | Stable at I=−10 |
| `wb_nan_no_panic` | No panic on NaN input |

See `tests/test_model_wang_buzsaki.py` (Python) and
`engine/src/neurons/biophysical/wang_buzsaki.rs` (Rust).

### Pipeline verification summary (measured 2026-04-04)

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | PASS | 3 state vars (v, h, n) |
| step() → int {0,1} | PASS | Upward crossing at −20 mV |
| 50 sub-steps | PASS | int(0.5/0.01) = 50 |
| m instantaneous | PASS | m_inf ∈ [0, 1] |
| phi acceleration | PASS | φ=5 verified vs φ=1 |
| State finite (20K) | PASS | At I=5 |
| Gating bounded | PASS | h, n ∈ [−0.01, 1.01] |
| Fires under drive | PASS | ≥1 spike at I=1 |
| Gamma band | PASS | 30–100 Hz at I=1 |
| ISI regularity | PASS | CV < 0.05 |
| f-I monotonic | PASS | Rate increases with I |
| reset() | PASS | All vars to defaults |
| Deterministic | PASS | Bit-exact |
| Population(n=20) | PASS | 20 instances |
| Network + PoissonInput | PASS | Spikes detected |
| spike_count | PASS | ≥ 0 |
| isi | PASS | All finite |
| firing_rate | PASS | ≥ 0 |

---

## 9. Gamma Oscillations and Network Dynamics

### ING mechanism (Wang & Buzsáki 1996)

The original paper's key finding: a network of 100 mutually
inhibitory WB neurons, driven by heterogeneous tonic excitation,
synchronises at gamma frequency through the ING mechanism:

1. All neurons fire roughly simultaneously
2. Mutual GABA_A inhibition suppresses firing for ~5–10 ms
3. As inhibition decays, neurons reach threshold again
4. Cycle repeats at ~30–80 Hz (set by GABA_A decay time)

The requirements for ING synchronisation are:
- Sufficiently fast spike dynamics (φ ≥ 3)
- GABA_A decay time constant ~5–10 ms
- Heterogeneous drive (prevents trivial synchrony)
- Sparse but sufficient connectivity (~20–50%)

### PING mechanism

When WB interneurons are embedded in an E-I network (e.g. with
TraubMiles excitatory cells), the PING mechanism emerges:

1. Excitatory neurons fire → excite interneurons
2. Interneurons fire → inhibit excitatory neurons
3. Excitatory neurons recover → fire again
4. Cycle period set by E→I→E delay + GABA_A decay

PING generates more regular gamma than ING and is thought to
be the dominant mechanism in neocortex (Tiesinga & Sejnowski 2009).

### Hippocampal theta-gamma coupling

In behaving animals, gamma oscillations are nested within theta
oscillations (4–12 Hz). Each theta cycle contains multiple gamma
cycles. The WB model's gamma-band dynamics, combined with slower
theta-paced input, can reproduce this coupling. Buzsáki & Wang (2012)
review the computational roles of theta-gamma coupling in hippocampal
information processing.

### Clinical relevance

Gamma oscillation disruption is a biomarker for several neurological
and psychiatric conditions:
- **Schizophrenia:** Reduced 40 Hz auditory steady-state response
- **Epilepsy:** Pathological high-frequency oscillations (>80 Hz)
- **Alzheimer's disease:** Reduced gamma power during memory tasks
- **Autism spectrum:** Altered gamma dynamics during sensory processing

The WB model provides a mechanistic tool for studying how changes
in PV+ interneuron properties (e.g. reduced $g_{Na}$, altered $\phi$)
affect network-level gamma oscillations.

---

## 10. Citations

1. Wang X-J, Buzsáki G (1996). Gamma oscillation by synaptic
   inhibition in a hippocampal interneuronal network model.
   *J Neurosci* 16(20):6402–6413.
   DOI: [10.1523/JNEUROSCI.16-20-06402.1996](https://doi.org/10.1523/JNEUROSCI.16-20-06402.1996)

2. Buzsáki G, Wang X-J (2012). Mechanisms of gamma oscillations.
   *Annu Rev Neurosci* 35:203–225.
   DOI: [10.1146/annurev-neuro-062111-150444](https://doi.org/10.1146/annurev-neuro-062111-150444)

3. Tiesinga P, Sejnowski TJ (2009). Cortical enlightenment: are
   attentional gamma oscillations driven by ING or PING?
   *Neuron* 63(6):727–732.
   DOI: [10.1016/j.neuron.2009.09.009](https://doi.org/10.1016/j.neuron.2009.09.009)

4. Hodgkin AL, Huxley AF (1952). A quantitative description of membrane
   current and its application to conduction and excitation in nerve.
   *J Physiol* 117(4):500–544.
   DOI: [10.1113/jphysiol.1952.sp004764](https://doi.org/10.1113/jphysiol.1952.sp004764)

5. Bartos M, Vida I, Jonas P (2007). Synaptic mechanisms of synchronized
   gamma oscillations in inhibitory interneuron networks. *Nat Rev Neurosci*
   8(1):45–56.
   DOI: [10.1038/nrn2044](https://doi.org/10.1038/nrn2044)

6. Cardin JA, Carlén M, Meletis K, Knoblich U, Zhang F, Deisseroth K,
   Tsai L-H, Moore CI (2009). Driving fast-spiking cells induces gamma
   rhythm and controls sensory responses. *Nature* 459(7247):663–667.
   DOI: [10.1038/nature08002](https://doi.org/10.1038/nature08002)

---

**ALL 19 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**
**Rust parity: EXACT (no defects found).**
**Criterion: 7,003 µs / 1K steps (7.0 µs/step, ~155× Python speedup).**
