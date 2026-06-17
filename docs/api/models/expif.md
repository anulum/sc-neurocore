# ExpIFNeuron

**Module:** `sc_neurocore.neurons.models.expif`
**Reference:** Fourcaud-Trocmé, Hansel, van Vreeswijk & Brunel, J. Neurosci. 23(37), 2003
**Family:** Integrate-and-fire (exponential spike initiation)
**State variables:** `v` (membrane potential)

---

## Equations

### Membrane potential with exponential spike initiation

$$\tau \frac{dV}{dt} = -(V - V_{rest}) + \Delta_T \exp\!\left(\frac{V - V_{rh}}{\Delta_T}\right) + I$$

### Spike and reset

$$V \geq V_{threshold}: \quad V \leftarrow V_{reset}$$

### Implementation

```python
def step(self, current: float) -> int:
    if not math.isfinite(self.v):
        raise ValueError("runtime voltage state must be finite")
    if not math.isfinite(current):
        raise ValueError("current must be finite")
    k1 = self._rhs(self.v, current)
    k2 = self._rhs(self.v + 0.5 * self.dt * k1, current)
    k3 = self._rhs(self.v + 0.5 * self.dt * k2, current)
    k4 = self._rhs(self.v + self.dt * k3, current)
    next_v = self.v + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    if not math.isfinite(next_v):
        raise ValueError("RK4 update must remain finite")
    self.v = next_v
    if self.v >= self.v_threshold:
        self.v = self.v_reset
        return 1
    return 0
```

Candidate-first fourth-order Runge-Kutta (RK4), single macro-step. The exp()
argument is clipped to [−20, 20] to prevent IEEE overflow. Runtime validation
is fail-closed across the maintained Python reference and native safety entry
points: non-finite current, corrupted voltage state, invalid time constants,
and non-finite RK4 derivatives or candidates are rejected before membrane state
mutation. This is the
**Exponential Integrate-and-Fire** (EIF) — the AdEx without the adaptation
current w.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane potential |
| `v_rest` | −65.0 | mV | Resting potential |
| `v_reset` | −68.0 | mV | Post-spike reset |
| `v_threshold` | −50.0 | mV | Hard spike threshold |
| `v_rh` | −55.0 | mV | Rheobase voltage (exp centre) |
| `delta_t` | 2.0 | mV | Spike sharpness (slope factor) |
| `tau` | 20.0 | ms | Membrane time constant |
| `dt` | 0.1 | ms | Integration timestep |

### Δ_T = 2.0 mV (sharpness)

Controls how sharply the exponential term kicks in:
- Δ_T → 0: approaches hard threshold (LIF limit)
- Δ_T = 2 mV: moderate sharpness (default, biologically realistic)
- Δ_T > 5 mV: very soft onset

At V = V_rh: exp_term = Δ_T × exp(0) = Δ_T = 2.0 mV.
At V = V_rh + Δ_T: exp_term = Δ_T × e ≈ 5.44 mV.
At V = V_rh + 5Δ_T: exp_term = Δ_T × e⁵ ≈ 296 mV (explosive).

### V_rh = −55 mV (rheobase voltage)

The voltage at which the exponential term equals Δ_T. Below V_rh, the
exponential is small (< Δ_T) and the dynamics are LIF-like. Above V_rh,
the exponential grows rapidly → spike initiation.

The gap V_threshold − V_rh = −50 − (−55) = 5 mV = 2.5 × Δ_T. This means
the hard threshold is 2.5 Δ_T above the exponential centre — the exp
term is exp(2.5) ≈ 12.2 at threshold.

---

## Analytical Properties

### Exponential spike initiation

The exp term models the Na⁺ channel activation:
- Below V_rh: subthreshold, leak dominates
- Near V_rh: exponential growth begins (positive feedback)
- Above V_rh: exponential runaway → V shoots up → hits threshold

This is more realistic than LIF's hard threshold: real action potentials
show a smooth, exponential onset (Naundorf et al. 2006).

### Relationship to AdEx

The AdEx (Adaptive Exponential IF) is:
$$\tau \dot{V} = -(V-V_r) + \Delta_T \exp(...) - w + I$$
$$\tau_w \dot{w} = a(V-V_r) - w$$

The ExpIF is AdEx with **w ≡ 0** — no adaptation. This means:
- No spike-frequency adaptation
- No subthreshold adaptation
- Simpler (1 ODE vs 2)
- Identical spike initiation dynamics

### Steady-state voltage (subthreshold)

For constant I, setting dV/dt = 0:
$$0 = -(V_{ss} - V_r) + \Delta_T \exp\!\left(\frac{V_{ss} - V_{rh}}{\Delta_T}\right) + I$$

This is a transcendental equation — no closed-form solution. For V_ss
well below V_rh (exp ≈ 0): V_ss ≈ V_rest + I (LIF approximation).

### Rheobase current

The minimum constant current for spiking. At the bifurcation:
$$I_{rh} = (V_{rh} - V_{rest}) - \Delta_T$$

With defaults: I_rh = (−55 − (−65)) − 2 = 8 mV. Below I = 8, the
neuron is subthreshold (at steady state). Above I = 8, the exponential
term drives V to threshold.

### Exp clipping

The clip to [−20, 20] bounds the exp:
- exp(−20) ≈ 2 × 10⁻⁹ (negligible, safe)
- exp(20) ≈ 4.9 × 10⁸ (large but finite, prevents inf)

Without clipping, V far above V_rh would produce exp(huge) = inf → NaN.

---

## Behaviour

### Subthreshold at low current

Verified: at small I, zero spikes in 5000 steps. The leak term
(−(V−V_rest)) dominates the exponential, keeping V below threshold.

### Spiking at moderate current

Verified: at sufficient I, the exponential term drives V above
threshold. The neuron fires and resets.

### Monotonic f-I curve

Higher current → more spikes. Verified across multiple current values.

### ISI regularity

With constant input, the ISI is approximately constant (no adaptation
to create ISI lengthening). The ExpIF produces regular spiking.

### Exponential escape at V_rh

At V = V_rh, exp_term = Δ_T = 2.0. This term is additive to the leak:
total drive = −(V_rh − V_rest) + Δ_T + I = −(−55 − (−65)) + 2 + I = −8 + I.
For I > 8: net depolarisation → V rises → exp grows → spike.

---

## Comparison with Related Models

| Property | ExpIF | AdEx | LIF | EscapeRate |
|----------|-------|------|-----|-----------|
| State vars | 1 (V) | 2 (V, w) | 1 (V) | 1 (V) |
| Spike initiation | Exponential | Exponential | Hard threshold | Stochastic |
| Adaptation | None | w current | None | None |
| Exp per step | 1 | 1 | 0 | 1 (safe_exp) |
| Sharpness | Δ_T | Δ_T | — | Δu |
| V_rh | Yes | Yes | — | V_threshold |
| Pipeline | Compatible | Compatible | Compatible | Compatible |

ExpIF is the **simplest exponential-onset model** — one parameter
(Δ_T) adds biophysical spike initiation to the LIF.

---

## Numerical Considerations

- **4 exp() evaluations per RK4 step:** Each RHS evaluation is clipped to
  [−20, 20] → no overflow.
- **Single RK4 macro-step:** dt=0.1 ms. Adequate for τ=20 ms.
- **No sub-stepping:** The exponential term can cause V to overshoot
  the threshold by a large amount in a single step. The hard threshold
  check (V ≥ V_threshold) catches this.
- **V_reset < V_rest:** The reset to −68 mV (below rest −65) creates a
  brief hyperpolarisation after each spike — a simple refractory effect.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/expif.py`.
- **One state variable:** v.
- **Dataclass:** Uses `@dataclass`.
- **Simplest exponential model** in SC-NeuroCore (AdEx adds w).
- **Polyglot surfaces:** Python, Rust engine, Rust safety, Go, Julia, and
  Mojo use candidate-first RK4 with clipped exponential RHS evaluations and
  reset-on-threshold spike semantics.

---

## Measured Performance (2026-06-16)

Local non-isolated regression run. These numbers are recorded for
regression comparison only and are not production throughput claims.

| Metric | Value |
|--------|-------|
| Evidence class | Local regression, non-isolated workstation |
| Benchmark artefact | `benchmarks/results/local_python_2026-06-16_expif_rk4.json` |
| Workload | 200000 steps, 5 repeats, I=20.0 |
| Polyglot contract | Python, Rust engine, Rust safety, Go, Julia, and Mojo RK4 surfaces aligned, with explicit errors where supported |

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 16820.414835 | 16763.600625 | 18207.096945 | 881 |
| Rust engine | 105.96647 | 105.191365 | 123.606465 | 881 |
| Go service mirror | 154.5 | 151.1 | 159.0 | 881 |
| Julia mirror | 99.584425 | 97.48369 | 104.18271 | 881 |
| Mojo mirror | 126.94658493273892 | 126.58483494305983 | 131.48430996807292 | 881 |

Fast for a nonlinear spiking model: one state variable, four clipped
exponential RHS evaluations per RK4 macro-step, and no sub-stepping.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | construction, binary output, state evolves, state finite (50K), reset |
| Exponential | 5 | exp term at V_rh, exp drives spike, Δ_T controls sharpness, exp clipping [−20,20], negative extreme finite |
| Analytical | 2 | membrane equation 1-step, subthreshold V approaches rest |
| F-I | 4 | subthreshold silent, suprathreshold fires, monotonic f-I, ISI regularity |
| Parameters | 4 | τ affects rate, custom V_rh, dt stability [0.05,0.1,0.2] (parametrised), deterministic |
| Performance | 2 | isolation throughput, network throughput |
| Pipeline | 4 | Population, Network spikes, Projection wiring, analysis pipeline |
| **Total** | **51** | **ALL PASSED** |

See `tests/test_model_expif.py`.

---

## Findings

1. **Exp term at V_rh = Δ_T.** At V = V_rh = −55, exp_term = 2.0 (exact).

2. **Exp drives spike.** Near V_rh, the exponential term provides enough
   positive feedback to push V to threshold.

3. **Δ_T controls sharpness.** Different Δ_T values produce different
   spike initiation dynamics — smaller Δ_T → sharper onset.

4. **Exp clipping prevents overflow.** At extreme V, exp argument clipped
   to ±20. No overflow, no NaN. Safe for any input.

5. **Membrane equation verified.** 1-step dV matches analytical formula
   to machine precision.

6. **Monotonic f-I curve.** Higher current → more spikes. Continuous
   onset from zero rate at rheobase (Class I excitability).

7. **ISI regular.** No adaptation → approximately constant ISI at fixed
   input. The coefficient of variation CV_ISI ≈ 0 (deterministic).

8. **τ affects firing rate.** Smaller τ → faster membrane dynamics →
   higher firing rate at same current.

9. **Rheobase at I ≈ 8 mV.** Matches analytical prediction
   I_rh = (V_rh − V_rest) − Δ_T = 10 − 2 = 8.

10. **Network pipeline fully functional.** Population, Projection,
    PoissonInput, spike_count, ISI, firing_rate all verified.

---

## Theoretical Context

### Historical background

Fourcaud-Trocmé, Hansel, van Vreeswijk, and Brunel (2003) introduced
the Exponential Integrate-and-Fire (EIF) model in "How spike generation
mechanisms determine the neuronal response to fluctuating inputs" in the
*Journal of Neuroscience*. The key contribution was showing that a
**single parameter** (Δ_T, the spike sharpness factor) bridges the gap
between the discontinuous LIF threshold and the smooth Hodgkin-Huxley
spike onset.

### Why exponential?

The Na⁺ channel activation m∞(V) is a Boltzmann sigmoid:
$$m_\infty(V) = \frac{1}{1 + \exp(-(V-V_{1/2})/k)}$$

Near the activation midpoint, this is approximately exponential:
$$m_\infty(V) \approx \exp((V-V_{1/2})/k)$$

The EIF's exponential term directly captures this near-threshold
Na⁺ activation — it is not an arbitrary choice but a biophysically
motivated approximation. The slope factor $k$ of the Boltzmann maps
to Δ_T.

### Parameter fitting from experimental data

Δ_T and V_rh can be estimated from:
- **f-I curve:** Fit the rheobase and slope near onset
- **Voltage traces:** Measure spike onset sharpness (phase plot dV/dt vs V)
- **Hodgkin-Huxley reduction:** Linearise the Na⁺ current near threshold

Typical values from cortical pyramidal cells:
- Δ_T ≈ 1–3 mV (sharp onset)
- V_rh ≈ −50 to −55 mV

Badel et al. (2008) developed an automated fitting procedure that
extracts EIF parameters directly from intracellular recordings using
a dynamic I-V method — enabling neuron-specific EIF models.

### EIF as the foundation for AdEx

Brette & Gerstner (2005) extended the EIF by adding an adaptation
current $w$, creating the Adaptive Exponential IF (AdEx):

$$\tau_m \dot{V} = -(V - V_{rest}) + \Delta_T \exp\!\left(\frac{V-V_{rh}}{\Delta_T}\right) - w + I$$
$$\tau_w \dot{w} = a(V - V_{rest}) - w + b \sum_k \delta(t - t_k)$$

The EIF is AdEx with $w \equiv 0$ — no adaptation. This means the
ExpIF captures the spike initiation dynamics exactly, but cannot
reproduce adaptation phenomena (spike-frequency adaptation, bursting,
initial transients).

### Bifurcation analysis

The EIF exhibits a saddle-node bifurcation at the rheobase current:

$$I_{rh} = (V_{rh} - V_{rest}) - \Delta_T$$

- Below $I_{rh}$: one stable fixed point (resting state)
- At $I_{rh}$: saddle-node — the stable and unstable fixed points
  merge and annihilate
- Above $I_{rh}$: no fixed points — all trajectories escape to
  threshold → repetitive spiking

This is a **Class I** excitability mechanism (Hodgkin 1948): the
firing rate starts from zero at the rheobase and increases
continuously with current (no discontinuous onset).

### Noise sensitivity

The EIF's smooth threshold makes it **less sensitive to input noise**
than the LIF near threshold. The exponential ramp provides a "grace
period" where small voltage fluctuations are absorbed by the nonlinear
dynamics rather than producing premature or missed spikes. This makes
the EIF more reliable in noisy network simulations.

### Standard simulator support

The EIF is the default "intermediate complexity" model in major
simulators:
- **Brian2:** `NeuronGroup(... model='dv/dt = ...')`
- **NEST:** `iaf_psc_exp` (EIF variant)
- **PyNN:** `IF_curr_exp` family
- **NEURON:** available via NMODL mechanism

### Phase plane analysis

The EIF's nullcline (dV/dt = 0 curve) is:

$$V_{null} = V_{rest} - \Delta_T \exp\!\left(\frac{V - V_{rh}}{\Delta_T}\right) - I$$

This is a downward-bending curve (due to the exponential). The
intersection with the V-axis gives the resting state (stable fixed
point). As $I$ increases:

1. The nullcline shifts upward
2. At $I = I_{rh}$: the nullcline just touches the V-axis → saddle-node
3. Above $I_{rh}$: no intersection → no fixed point → repetitive firing

The phase portrait is one-dimensional (single variable V), so the
dynamics are fully characterised by the nullcline and the sign of dV/dt.

### Comparison with LIF near threshold

The key difference between EIF and LIF appears in the response to
fluctuating inputs near threshold:

- **LIF:** Binary response — subthreshold fluctuations have zero
  effect, suprathreshold fluctuations produce a spike. Sharp
  discontinuity in the transfer function.
- **EIF:** Graded response — the exponential ramp smoothly amplifies
  near-threshold fluctuations. The transfer function is continuous
  and differentiable at threshold.

This difference is critical for network dynamics: EIF networks exhibit
smoother, more stable dynamics near bifurcation points, while LIF
networks can show artificial sensitivity to noise near threshold.

### Mean-field theory with EIF

The Siegert formula for the EIF (Fourcaud-Trocmé et al. 2003) involves
a modified first-passage time calculation where the boundary condition
accounts for the exponential spike mechanism:

$$r_{EIF} = \left[\tau_{rp} + \tau_m \sqrt{\pi} \int_{-\infty}^{u_{th}} e^{u^2} (1 + \text{erf}(u)) du + \text{correction}(\Delta_T) \right]^{-1}$$

The correction term depends on Δ_T and smoothly interpolates between
the LIF Siegert formula (Δ_T → 0) and the noise-dominated regime.
This enables analytical mean-field analysis of EIF networks.

### Applications

The EIF model is used extensively in:

- **Cortical network modelling:** Replacing LIF with EIF in random
  recurrent networks produces more realistic spike onset dynamics
  and f-I curves (Ostojic 2014)
- **Parameter inference:** The EIF provides a compact parametric
  description of single-neuron dynamics that can be fitted to in vitro
  data (Badel et al. 2008) and used for Bayesian inference
- **Neuromorphic hardware:** The exponential function is implemented
  in analogue hardware (BrainScaleS, Schemmel et al. 2010) using
  transistor physics (exponential I-V characteristic of subthreshold
  MOSFETs matches Δ_T naturally)
- **Computational psychiatry:** EIF parameter changes (Δ_T, V_rh)
  can model altered spike initiation in channelopathies and
  neurodegenerative diseases affecting Na⁺ channel function

---

## Usage Examples

### Example 1: Basic spiking with constant input

```python
from sc_neurocore.neurons.models.expif import ExpIFNeuron

n = ExpIFNeuron()
spikes = 0
v_trace = []
for t in range(10000):
    spike = n.step(current=12.0)
    spikes += spike
    v_trace.append(n.v)

print(f"Spikes: {spikes}")
print(f"Firing rate: {spikes / (10000 * 0.1 / 1000):.1f} Hz")
```

### Example 2: F-I curve measurement

```python
from sc_neurocore.neurons.models.expif import ExpIFNeuron

for I in range(0, 30, 2):
    n = ExpIFNeuron()
    spikes = sum(n.step(float(I)) for _ in range(50000))
    rate = spikes / (50000 * 0.1 / 1000)
    print(f"I={I:3d} mV: {rate:6.1f} Hz")
```

### Example 3: Sharpness comparison (Δ_T sweep)

```python
from sc_neurocore.neurons.models.expif import ExpIFNeuron

for dt_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
    n = ExpIFNeuron(delta_t=dt_val)
    spikes = sum(n.step(15.0) for _ in range(20000))
    print(f"delta_t={dt_val:4.1f}: {spikes} spikes in 20K steps")
```

---

## Technical Reference

### Rust parity

| Aspect | Python | Rust | Status |
|--------|--------|------|--------|
| State variable | v (membrane potential) | same | **EXACT** |
| Exp term | Δ_T × exp(clip((v-v_rh)/Δ_T, -20, 20)) | same | **EXACT** |
| Euler integration | dt/tau | same | **EXACT** |
| All defaults | identical | identical | **EXACT** |

**No parity defects.** EXACT parity verified by automated scan.

### Source files

| File | Lines | Description |
|------|-------|-------------|
| `src/sc_neurocore/neurons/models/expif.py` | Python candidate-first RK4 reference |
| `engine/src/neuron.rs` | Rust engine RK4 mirror |
| `src/sc_neurocore/accel/go/services/expif.go` | Go service RK4 mirror |
| `src/sc_neurocore/accel/julia/neurons/expif.jl` | Julia RK4 mirror |
| `src/sc_neurocore/accel/mojo/kernels/expif.mojo` | Mojo RK4 mirror |
| `tests/test_model_expif.py` | Python RK4, validation, dynamics, and pipeline checks |
| `src/sc_neurocore/accel/go/services/expif_test.go` | Go RK4 reference and invalid-update checks |

---

## Performance Benchmarks

### RK4 local regression benchmark (measured 2026-06-16)

Reproduce with:

```bash
PYTHONPATH=src ./.venv/bin/python benchmarks/bench_model_expif.py
```

Artifact: `benchmarks/results/local_python_2026-06-16_expif_rk4.json`.
Rust PyO3 timing is opt-in via `SC_NEUROCORE_BENCH_RUST_PYO3=1` after
rebuilding/installing the local engine wheel, so the committed artifact does not
mix new Python source with a possibly stale installed Rust extension.

This is **non-isolated local regression evidence** on a loaded workstation. It
must not be promoted as a production speed claim without an isolated-core rerun.

### Historical Criterion benchmarks (local i5-11600K, measured 2026-04-05)

| Metric | Value |
|--------|-------|
| Test | `expif_10k_steps` |
| Median | 238.9 µs |
| Per-step | 23.9 ns |
| Throughput | ~41.8M steps/s |

### Python baseline

| Metric | Value |
|--------|-------|
| Isolation | ~220K steps/s |

The historical numbers measured the older one-stage Euler path. The maintained
production path now evaluates four EIF derivative stages per macro-step, so the
old speedup is retained only as historical context.

---

## Limitations

- **No adaptation:** Cannot reproduce spike-frequency adaptation,
  bursting, or initial transients. Use AdEx for these.
- **Hard upper threshold:** The hard threshold at V_threshold is
  unrealistic — real action potentials peak at +30 mV or higher.
  This is a standard simplification in IF models.
- **No refractory period:** Only the reset to V_reset < V_rest
  provides a brief relative refractory effect. No explicit
  absolute refractory period.
- **Explicit RK4:** The exponential term can still create sharp threshold
  approaches, but the maintained path no longer uses a raw one-stage Euler
  mutation. Implicit methods remain a possible future upgrade for very large
  timesteps.
- **Single compartment:** No dendritic processing, no axonal
  conduction delay.
- **No synaptic dynamics:** Input current is instantaneous —
  no synaptic filtering (AMPA, NMDA, GABA time constants). For
  models with explicit synaptic kinetics, couple with
  TsodyksMarkram synapses.

---

## Relationship to Other SC-NeuroCore Models

### Model hierarchy

The EIF sits in a hierarchy of increasing complexity:

| Model | Variables | Adaptation | Exp onset | Use case |
|-------|-----------|------------|-----------|----------|
| LIF | 1 (V) | No | No | Fastest, simplest |
| **ExpIF** | **1 (V)** | **No** | **Yes** | **Smooth onset** |
| AdEx | 2 (V, w) | Yes | Yes | Full adaptation |
| Izhikevich | 2 (v, u) | Yes | Quadratic | Classification |
| HH | 4+ (V, m, h, n) | Intrinsic | Biophysical | Detailed |

The ExpIF is the minimal model with biophysically realistic spike
initiation. It is the natural choice when LIF's hard threshold is
too crude but AdEx's adaptation is unnecessary. For most cortical
modelling scenarios, the ExpIF provides an excellent balance of
biophysical fidelity and computational cost.

---

## Citations

1. Fourcaud-Trocmé N, Hansel D, van Vreeswijk C, Brunel N (2003). How
   spike generation mechanisms determine the neuronal response to
   fluctuating inputs. *J Neurosci* 23(37):11628–11640.
   DOI: [10.1523/JNEUROSCI.23-37-11628.2003](https://doi.org/10.1523/JNEUROSCI.23-37-11628.2003)

2. Brette R, Gerstner W (2005). Adaptive exponential integrate-and-fire
   model as an effective description of neuronal activity. *J Neurophysiol*
   94(5):3637–3642.
   DOI: [10.1152/jn.00686.2005](https://doi.org/10.1152/jn.00686.2005)

3. Badel L, Lefort S, Brette R, Petersen CCH, Gerstner W, Richardson MJE
   (2008). Dynamic I-V curves are reliable predictors of naturalistic
   pyramidal-neuron voltage traces. *J Neurophysiol* 99(2):656–666.
   DOI: [10.1152/jn.01107.2007](https://doi.org/10.1152/jn.01107.2007)

4. Naundorf B, Wolf F, Volgushev M (2006). Unique features of action
   potential initiation in cortical neurons. *Nature* 440(7087):1060–1063.
   DOI: [10.1038/nature04610](https://doi.org/10.1038/nature04610)

5. Izhikevich EM (2007). *Dynamical Systems in Neuroscience: The Geometry
   of Excitability and Bursting.* MIT Press. Chapter 8: Simple models.
   ISBN: 978-0-262-09043-8.

---

**ALL 28 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**
**Rust parity: EXACT (no defects found).**
**Criterion: 238.9 µs / 10K steps (23.9 ns/step, ~41.8M steps/s).**
