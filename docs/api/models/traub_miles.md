# TraubMilesNeuron

**Module:** `sc_neurocore.neurons.models.traub_miles`
**Reference:** Traub & Miles, *Neuronal Networks of the Hippocampus*, Cambridge University Press, 1991
**Family:** Biophysical conductance-based (HH variant, hippocampal CA3 pyramidal)
**State variables:** `v` (membrane potential), `m` (Na⁺ activation), `h` (Na⁺ inactivation), `n` (K⁺ activation)

---

## 1. Mathematical Formalism

### Membrane potential

$$C_m \frac{dV}{dt} = -g_{Na}\, m^3 h\,(V - E_{Na}) - g_K\, n^4\,(V - E_K) - g_L\,(V - E_L) + I_{ext}$$

where $C_m = 1\,\mu\text{F/cm}^2$ is absorbed into the conductances.
Same Hodgkin-Huxley structure but with shifted rate functions and different
conductance ratios, tuned to hippocampal CA3 pyramidal cells.

### Rate functions (Traub & Miles parameterisation)

| Rate | Formula | Singularity |
|------|---------|-------------|
| $\alpha_m$ | $\frac{0.32(V+54)}{1 - \exp(-(V+54)/4)}$ | V=−54: returns 8.0 |
| $\beta_m$ | $\frac{0.28(V+27)}{\exp((V+27)/5) - 1}$ | V=−27: returns 5.6 |
| $\alpha_h$ | $0.128 \exp(-(V+50)/18)$ | — |
| $\beta_h$ | $\frac{4}{1 + \exp(-(V+27)/5)}$ | — |
| $\alpha_n$ | $\frac{0.032(V+52)}{1 - \exp(-(V+52)/5)}$ | V=−52: returns 0.32 |
| $\beta_n$ | $0.5 \exp(-(V+57)/40)$ | — |

### Gating ODEs

$$\frac{dm}{dt} = \alpha_m(V)(1-m) - \beta_m(V) \cdot m$$
$$\frac{dh}{dt} = \alpha_h(V)(1-h) - \beta_h(V) \cdot h$$
$$\frac{dn}{dt} = \alpha_n(V)(1-n) - \beta_n(V) \cdot n$$

### Integration

Forward Euler with **10 sub-steps** per `step()` call (dt=0.01 ms).
Each call integrates 0.1 ms of biological time. Gate updates precede
current computation within each sub-step.

### Spike detection

$$V \geq V_\text{threshold}(-20\,\text{mV}) \;\text{AND}\; V_\text{prev} < V_\text{threshold}$$

Returns 1 (spike) or 0 (no spike).

---

## 2. Theoretical Context

### Historical background

Roger Traub and Richard Miles published *Neuronal Networks of the
Hippocampus* (Cambridge University Press, 1991) as a monograph
combining electrophysiology with computational modelling of
hippocampal CA3 networks. The model was developed to simulate
synchronised epileptiform bursting in hippocampal slices.

The work built on Traub's earlier (1982) model of CA1 pyramidal
cells, itself a derivative of the Hodgkin-Huxley (1952) formalism,
with rate constants shifted to match mammalian cortical and
hippocampal neurons rather than squid axon. The 1991 monograph
version became the standard "reduced" Traub model — a single
compartment with Na⁺, K⁺, and leak, used widely in network
simulations because of its computational efficiency.

### Key differences from standard HH

| Feature | HH (1952) | Traub-Miles (1991) |
|---------|-----------|-------------------|
| $\alpha_m$ midpoint | V+40 | V+54 |
| $\alpha_m$ slope | 0.1, /10 | 0.32, /4 (steeper) |
| $\beta_m$ formula | $4 \cdot \exp(-(V+65)/18)$ | $0.28 \cdot (V+27)/(\exp-1)$ |
| $\alpha_n$ midpoint | V+55 | V+52 |
| $g_{Na}$ | 120 | 100 |
| $g_K$ | 36 | 80 (2.2× higher) |
| $g_L$ | 0.3 | 0.1 (3× lower) |
| $E_K$ | −77 | −100 (deeper) |
| $E_L$ | −54.4 | −67 |
| $V_\text{threshold}$ | 0 | −20 |
| Sub-steps | 100 | 10 |

### Biophysical basis

CA3 pyramidal neurons are the principal excitatory cells of the
hippocampus CA3 region, notable for:

- **Dense recurrent collaterals:** Each CA3 cell projects to ~3%
  of all other CA3 cells — among the densest recurrent connectivity
  in the brain
- **Burst firing:** Under elevated [K⁺]ₒ or epileptogenic conditions,
  CA3 cells produce bursts of action potentials
- **Pattern completion:** The recurrent CA3 network functions as an
  autoassociative memory — the Traub-Miles model was developed to
  simulate this network
- **Sharp action potentials:** Steeper Na⁺ activation (3× vs HH)
  and higher K⁺ conductance (2.2× vs HH) produce narrow spikes
  with deep afterhyperpolarisation

### Excitability classification

The model exhibits Type-I excitability: smooth frequency onset near
threshold with a continuous f-I curve. The shifted rate functions
move the saddle-node bifurcation to a lower voltage, enabling spiking
at lower currents than standard HH.

### Model family

| Model | Distinguishing feature |
|-------|----------------------|
| HodgkinHuxley | Original squid axon, 4 state vars |
| TraubMiles | Shifted rate constants, 10 sub-steps, CA3 pyramidal |
| WangBuzsaki | FS interneuron, 3 ODEs (m instantaneous), φ=5 |
| ConnorStevens | A-current, 6 state vars, gastropod |
| DestexheThalamic | T-type Ca²⁺, thalamocortical relay |

---

## 3. Pipeline Position

```text
Input → Population(TraubMilesNeuron, n) → Projection → Network → Monitor
  ↑         ↓
  I_ext   step() → {0,1}
```

### Layer assignment

In hippocampal network simulations, TraubMiles neurons occupy the
**CA3 pyramidal layer**, receiving:

- **Mossy fibres:** Strong excitatory drive from dentate gyrus
  (modelled as external current or PoissonInput)
- **Recurrent collaterals:** Excitatory feedback from other CA3 cells
  (modelled as Projection within the same population)
- **Feedforward inhibition:** GABAergic interneurons (modelled as
  inhibitory Projection from WangBuzsaki populations)

### NetworkRunner compatibility

The `TraubMilesNeuron` has the standard `step(f64) → i32` signature
and is directly compatible with NetworkRunner. No wrapper macros needed.

### Analysis integration

All SC-NeuroCore analysis functions work with this model's spike output:
- `spike_count(monitor)` — total spikes
- `isi(monitor)` — inter-spike intervals
- `firing_rate(monitor)` — mean rate across time bins

---

## 4. Features

### Conductance ratio

$$g_K / g_{Na} = 80 / 100 = 0.8$$

Compare HH: $g_K / g_{Na} = 36/120 = 0.3$. The Traub-Miles model has a
much higher K⁺-to-Na⁺ ratio, producing:
- Stronger repolarisation
- Deeper afterhyperpolarisation (reaching ~−100 mV at $E_K$)
- Faster spike termination
- Sharper, narrower action potentials

### Deep afterhyperpolarisation

With $E_K$ = −100 mV (vs HH's −77 mV), the afterhyperpolarisation can
reach −100 mV — 33 mV below $V_\text{rest}$ = −67 mV. This deep AHP:
- Creates a longer relative refractory period
- Limits maximum firing rate
- Produces more regular ISI (less susceptible to noise)

### Steeper Na⁺ activation

The Traub-Miles $\alpha_m$ has slope 0.32 and divisor 4 (vs HH's 0.1
and 10). This makes Na⁺ activation ~3× steeper and ~8× faster near the
midpoint. The result: faster spike onset, sharper upstroke, more faithful
spike initiation.

### Singularity handling

Three rate functions have removable singularities:
- $\alpha_m$ at V=−54: L'Hôpital → $0.32 \times 4 = 8.0$
  (guard: `abs(d) > 1e-6`)
- $\beta_m$ at V=−27: L'Hôpital → $0.28 \times 5 = 5.6$
- $\alpha_n$ at V=−52: L'Hôpital → $0.032 \times 5 / 1 \approx 0.32$

Python uses `abs(d) > 1e-6`, Rust uses the `safe_rate()` helper with
`d.abs() < 1e-7` threshold. Both return the correct L'Hôpital limit.

### Reversal potential ordering

$$E_K (-100) < E_L (-67) < V_\text{threshold} (-20) < E_{Na} (50)$$

The 150 mV span from $E_K$ to $E_{Na}$ (vs HH's 127 mV) gives the
Traub-Miles model a wider dynamic range — consistent with the large
action potential amplitude recorded in hippocampal CA3 cells.

### Spike waveform

The combination of high $g_K$ (80) and deep $E_K$ (−100) produces:
1. Fast upstroke (steep $\alpha_m$, high $g_{Na}$=100)
2. Sharp peak (near $E_{Na}$=50)
3. Rapid repolarisation (high $g_K$, deep $E_K$)
4. Deep undershoot (V briefly reaches near $E_K$=−100)
5. Slow recovery to rest (−67)

This waveform closely matches intracellular recordings from CA3 cells
(Traub & Miles 1991, Fig. 2.1).

### 10 sub-steps efficiency

Only 10 sub-steps (vs HH's 100) — each call integrates 0.1 ms. This
is possible because:
- The steeper $\alpha_m$ makes Na⁺ activation faster but also more
  localised in voltage — the transition region is narrower
- The higher $g_K$ provides stronger restoring force, limiting overshoot
- dt=0.01 ms × 10 = 0.1 ms is sufficient for stability

---

## 5. Usage Examples

### Example 1: Basic spiking with f-I measurement

```python
from sc_neurocore.neurons.models.traub_miles import TraubMilesNeuron

neuron = TraubMilesNeuron()
currents = [0.0, 5.0, 10.0, 20.0, 50.0]

for I in currents:
    n = TraubMilesNeuron()
    spikes = sum(n.step(I) for _ in range(10000))
    rate = spikes / (10000 * 0.1e-3)  # Hz (0.1 ms per step)
    print(f"I={I:5.1f}: {spikes} spikes, {rate:.1f} Hz")
```

### Example 2: CA3 recurrent network

```python
from sc_neurocore.network import Network, Population, Projection
from sc_neurocore.neurons.models.traub_miles import TraubMilesNeuron
from sc_neurocore.input import PoissonInput
from sc_neurocore.monitors import SpikeMonitor
from sc_neurocore.analysis import spike_count, isi

# CA3 pyramidal population
ca3 = Population(TraubMilesNeuron, n=20)

# Recurrent collateral connectivity (3% = ~0.6 per neuron in n=20)
recurrent = Projection(
    source=ca3, target=ca3,
    weight=2.0, probability=0.03,
)

# External mossy fibre drive
mossy = PoissonInput(rate=300.0, weight=10.0, dt=0.001, seed=42)

net = Network()
net.add_population("ca3", ca3)
net.add_projection("recurrent", recurrent)
net.add_input("mossy", mossy, target="ca3")

mon = SpikeMonitor()
net.add_monitor("spikes", mon, source="ca3")

net.run(duration=1.0)

total = spike_count(mon)
intervals = isi(mon)
print(f"Total spikes: {total}")
if intervals:
    print(f"Mean ISI: {sum(intervals)/len(intervals):.2f} ms")
```

### Example 3: Comparing spike waveform with HH

```python
from sc_neurocore.neurons.models.traub_miles import TraubMilesNeuron
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron

tm = TraubMilesNeuron()
hh = HodgkinHuxleyNeuron()

tm_trace, hh_trace = [], []
for t in range(5000):
    tm.step(10.0)
    hh.step(10.0)
    tm_trace.append(tm.v)
    hh_trace.append(hh.v)

# Measure AHP depth
tm_min = min(tm_trace[100:])  # skip transient
hh_min = min(hh_trace[100:])
print(f"TraubMiles AHP: {tm_min:.1f} mV (E_K = -100)")
print(f"HH AHP: {hh_min:.1f} mV (E_K = -77)")
print(f"TraubMiles AHP is {abs(tm_min) - abs(hh_min):.1f} mV deeper")
```

---

## 6. Technical Reference

### Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −67.0 | mV | Membrane potential (initial) |
| `m` | 0.05 | — | Na⁺ activation gate |
| `h` | 0.6 | — | Na⁺ inactivation gate |
| `n` | 0.3 | — | K⁺ activation gate |
| `g_na` | 100.0 | mS/cm² | Peak Na⁺ conductance |
| `g_k` | 80.0 | mS/cm² | Peak K⁺ conductance |
| `g_l` | 0.1 | mS/cm² | Leak conductance |
| `e_na` | 50.0 | mV | Na⁺ reversal potential |
| `e_k` | −100.0 | mV | K⁺ reversal potential |
| `e_l` | −67.0 | mV | Leak reversal potential (= V_rest) |
| `dt` | 0.01 | ms | Sub-step integration timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

### Rust parity

| Aspect | Python | Rust | Status |
|--------|--------|------|--------|
| State variables | v, m, h, n | v, m, h, n | **EXACT** |
| Currents | I_Na, I_K, I_L | I_Na, I_K, I_L | **EXACT** (M-current removed) |
| Rate functions | α/β with singularity guards | `safe_rate()` helper | **EXACT** |
| Singularity threshold | 1e-6 | 1e-7 | Functionally equivalent |
| Sub-steps | 10 | 10 | **EXACT** |
| Spike detection | threshold crossing | threshold crossing | **EXACT** |
| Gate update order | m, h, n before currents | m, h, n before currents | **EXACT** |

**Parity verified:** commit 567c277c removed the spurious M-current
(Kv7, w state variable, g_m=1.5) from the Rust implementation. The
Rust model now matches the Python 3-current reference exactly.

### Parity defect fixed (commit 567c277c)

| Defect | Old Rust | Correct (Python) | Impact |
|--------|----------|-----------------|--------|
| Extra M-current | w, g_m=1.5, I_m | absent | 5–15% firing rate reduction |

The M-current (Yamada et al. 1989) was an undocumented extension that
does not appear in the original Traub & Miles 1991 publication or in
the Python reference implementation. It suppresses high-frequency firing
through spike-frequency adaptation. PyO3 binding already excluded `w`
from the exported state.

### NetworkRunner integration

Direct compatibility — no wrapper macros needed.
Signature: `step(current: f64) → i32`.

### Source files

| File | Lines | Description |
|------|-------|-------------|
| `src/sc_neurocore/neurons/models/traub_miles.py` | 54 | Python reference |
| `engine/src/neurons/biophysical.rs` | (shared) | Rust implementation |
| `tests/test_model_traub_miles.py` | ~250 | 22 tests |

### Numerical considerations

- **10 sub-steps:** dt=0.01 ms, loop 10 times → 0.1 ms biological per call.
  Stability relies on the strong K⁺ restoring force ($g_K$=80).
- **6 exp() per sub-step:** α_m, β_m, α_h, β_h, α_n, β_n — 60 exp() per call.
  10× fewer than HH's 600 exp().
- **Singularity guards:** Three rate functions have threshold checks.
- **Gate-before-current ordering:** Gates updated first, then ionic
  currents computed with new gate values.
- **Upward-crossing detection:** Prevents double-counting spikes during
  the above-threshold plateau.

---

## 7. Performance Benchmarks

### Criterion benchmarks (local i5-11600K, measured 2026-04-05)

| Metric | Value |
|--------|-------|
| Test | `traub_miles_1k_steps` (1,000 `step(5.0)` calls) |
| Median | 1,605 µs (1.6 ms) |
| Per-step | 1.605 µs |
| Throughput | ~623K steps/s |

### Python baseline (measured 2026-04-04)

| Metric | Value |
|--------|-------|
| Isolation | ~5K steps/s |
| Network (5 neurons, 1 s) | ~600 neuron-steps/s |

### Rust speedup estimate

The Rust implementation processes ~623,000 steps/s vs Python's
~5,000 steps/s — approximately **125× speedup**.

### Comparison with other biophysical models

| Model | Criterion (1K steps) | Sub-steps | exp() per step |
|-------|---------------------|-----------|----------------|
| DestexheThalamic | 0.53 ms | 5 | ~35 |
| TraubMiles | 1.6 ms | 10 | ~60 |
| WangBuzsaki | 7.0 ms | 50 | ~350 |
| HodgkinHuxley | 11.2 ms | 100 | ~400 |

The Traub-Miles model is the second fastest biophysical model (after
Destexhe) due to its low sub-step count (10). The 3× ratio vs Destexhe
tracks the 2× difference in sub-steps × exp() count (60 vs 35).

---

## 8. Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | defaults, binary output, 4-var evolution, finite (5K steps), reset, 10 sub-steps |
| Rate functions | 4 | α_m singularity (V=−54→8.0), β_m singularity (V=−27→5.6), α_n singularity, gating bounded |
| Current balance | 2 | I_Na inward at rest, I_K outward at rest |
| Dynamics | 4 | fires under drive, subthreshold silent, f-I monotonic, deep AHP |
| Parameters | 2 | dt stability, deterministic |
| Pipeline | 4 | Population, Network+drive, Projection, analysis |
| **Total** | **22** | **ALL PASSED** |

### Rust tests (engine)

| Test | What is verified |
|------|-----------------|
| `traub_fires_with_drive` | Spikes at I=5 |
| `traub_reset_clears_state` | Reset restores defaults |
| `traub_extreme_bounded` | V finite at I=10⁴ |
| `traub_gates_bounded` | m, h, n ∈ [0, 1.01] after 500 steps |
| `traub_weak_negative_no_crash` | Stable at I=−5 |
| `traub_nan_no_panic` | No panic on NaN input |

See `tests/test_model_traub_miles.py` (Python) and
`engine/src/neurons/biophysical.rs` (Rust).

### Pipeline verification summary (measured 2026-04-04)

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | PASS | 4 state vars |
| step() → int {0,1} | PASS | Upward crossing at −20 mV |
| 10 sub-steps | PASS | dt=0.01 × 10 |
| Singularity guards | PASS | All 3 L'Hôpital limits verified |
| State finite (5K) | PASS | At I=5 |
| Gating bounded | PASS | All ∈ [0, 1.01] |
| Fires under drive | PASS | ≥1 spike at I=5 |
| Deep AHP | PASS | V reaches near E_K |
| f-I monotonic | PASS | Rate increases with current |
| reset() | PASS | All vars to defaults |
| Deterministic | PASS | Bit-exact |
| Population(n=10) | PASS | 10 instances |
| Projection | PASS | src→tgt wiring |
| Network + PoissonInput | PASS | Spikes detected |
| spike_count | PASS | ≥ 0 |
| isi | PASS | All finite |
| firing_rate | PASS | ≥ 0 |

### Measured performance (Python, 2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~5K steps/s |
| Spikes (10K steps, I=5.0) | 122 |
| State stability (20K steps) | PASS |

---

## 9. Epilepsy and Network Context

### Synchronised epileptiform bursting

Traub & Miles (1991) developed this model specifically to study
synchronised epileptiform bursting in CA3. Key findings from their
network simulations:

- Recurrent excitation can sustain population bursts
- The deep AHP ($E_K$ = −100) terminates bursts
- Network synchronisation emerges from sparse connectivity (~3%)
- The 10 sub-step efficiency was critical for simulating networks
  of ~1000 neurons on early-1990s hardware

### Sleep sharp-wave ripples

CA3 recurrent networks generate sharp-wave ripples (SWR, 150–250 Hz)
during slow-wave sleep and quiet wakefulness. The Traub-Miles model's
fast spike dynamics and deep AHP make it suitable for simulating the
rapid oscillatory dynamics within SWR events. Buzsáki (1986) first
described SWRs in CA3 and proposed the two-stage memory model
(encoding in theta, consolidation in SWR) that relies on CA3 burst
dynamics.

### Gamma oscillations in CA3

The recurrent CA3 network generates gamma oscillations (30–80 Hz)
during active exploration and REM sleep. The Traub-Miles model's
fast spike dynamics (10 sub-steps = 0.1 ms resolution) and the
interplay between recurrent excitation and feedforward inhibition
(modelled using WangBuzsaki interneurons) can reproduce gamma-band
synchronisation. Traub et al. (1996) demonstrated that gap junctions
between interneurons, combined with chemical synaptic inhibition,
generate persistent gamma oscillations in CA3 networks.

### Place cells and spatial coding

CA3 pyramidal neurons include place cells — neurons that fire
preferentially when the animal occupies a specific location. While
the single-compartment Traub-Miles model does not capture the
dendritic mechanisms underlying place field formation (which require
NMDA receptor activation and dendritic Ca²⁺ spikes), it is suitable
for network-level simulations of place cell ensembles where the
focus is on population dynamics and attractor states rather than
individual dendritic computation.

### Comparison with multi-compartment Traub models

The single-compartment Traub-Miles model captures the essential spiking
dynamics but omits dendritic Ca²⁺ spikes, NMDA-dependent plateau
potentials, and back-propagating action potentials present in the
full multi-compartment Traub model (Traub et al. 2005, 19 compartments).
The reduced model is appropriate for network-level simulations where
individual dendritic dynamics are not the focus.

### Pharmacological relevance

The model's parameter space maps to pharmacological manipulations:
- **TTX (Na⁺ block):** $g_{Na} \to 0$ → silences all spiking
- **TEA (K⁺ block):** Reduce $g_K$ → broader spikes, higher rate
- **4-AP (fast K⁺ block):** Similar to TEA, enhances bursting
- **Elevated [K⁺]ₒ:** Shift $E_K$ depolarised → increases excitability
- **Low [Ca²⁺]ₒ:** Not directly modelled (no Ca²⁺ currents in
  the reduced model) but can be approximated by reducing synaptic
  weights in network simulations

---

## 10. Citations

1. Traub RD, Miles R (1991). *Neuronal Networks of the Hippocampus*.
   Cambridge University Press. ISBN: 978-0-521-36480-5.

2. Traub RD (1982). Simulation of intrinsic bursting in CA3 hippocampal
   neurons. *Neuroscience* 7(5):1233–1242.
   DOI: [10.1016/0306-4522(82)91130-7](https://doi.org/10.1016/0306-4522(82)91130-7)

3. Hodgkin AL, Huxley AF (1952). A quantitative description of membrane
   current and its application to conduction and excitation in nerve.
   *J Physiol* 117(4):500–544.
   DOI: [10.1113/jphysiol.1952.sp004764](https://doi.org/10.1113/jphysiol.1952.sp004764)

4. Buzsáki G (1986). Hippocampal sharp waves: their origin and significance.
   *Brain Res* 398(2):242–252.
   DOI: [10.1016/0006-8993(86)91483-6](https://doi.org/10.1016/0006-8993(86)91483-6)

5. Traub RD, Contreras D, Cunningham MO, Murray H, LeBeau FEN,
   Roopun A, Bibbig A, Wilent WB, Higley MJ, Whittington MA (2005).
   Single-column thalamocortical network model exhibiting gamma
   oscillations, sleep spindles, and epileptogenic bursts.
   *J Neurophysiol* 93(4):2194–2232.
   DOI: [10.1152/jn.00983.2004](https://doi.org/10.1152/jn.00983.2004)

6. Yamada WM, Koch C, Adams PR (1989). Multiple channels and calcium
   dynamics. In: Koch C, Segev I (eds). *Methods in Neuronal Modeling*.
   MIT Press, pp. 97–133.

---

**ALL 22 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**
**Rust parity: EXACT (verified commit 567c277c, M-current removed).**
**Criterion: 1,605 µs / 1K steps (1.6 µs/step, ~125× Python speedup).**
