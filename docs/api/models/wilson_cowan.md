# WilsonCowanUnit

**Module:** `sc_neurocore.neurons.models.wilson_cowan`
**Reference:** Wilson & Cowan, Biophys. J. 12(1), 1972
**Family:** Rate model (neural mass, excitatory–inhibitory population)
**State variables:** `e` (excitatory population rate), `i` (inhibitory population rate)

---

## Equations

### Excitatory population

$$\tau_E \frac{dE}{dt} = -E + S(w_{EE} E - w_{EI} I + I_{ext})$$

### Inhibitory population

$$\tau_I \frac{dI}{dt} = -I + S(w_{IE} E - w_{II} I)$$

### Sigmoid activation function

Published two-term form (Wilson & Cowan 1972):

$$S(x) = \frac{1}{1 + \exp(-a(x - \theta))} - \frac{1}{1 + \exp(a\theta)}$$

where $a$ is the sigmoid gain (steepness) and $\theta$ is the threshold
(midpoint). The subtracted baseline $\beta = 1/(1 + \exp(a\theta))$ makes
$S(0) = 0$ exactly; the range is $[-\beta,\, 1-\beta]$. This baseline
term is what allows the Hopf bifurcation that produces the paper's
canonical limit-cycle regime — an earlier one-term implementation
without the subtraction flattened the fixed-point structure and
suppressed oscillations.

### Implementation (as coded)

```python
def step(self, ext_input: float = 0.0) -> float:
    validate_state_and_parameters()
    se = self._sigmoid(self.w_ee * self.e - self.w_ei * self.i + ext_input)
    k1_e, k1_i = self._derivatives(self.e, self.i, drive)
    k2_e, k2_i = self._derivatives(
        self.e + 0.5 * self.dt * k1_e,
        self.i + 0.5 * self.dt * k1_i,
        drive,
    )
    k3_e, k3_i = self._derivatives(
        self.e + 0.5 * self.dt * k2_e,
        self.i + 0.5 * self.dt * k2_i,
        drive,
    )
    k4_e, k4_i = self._derivatives(
        self.e + self.dt * k3_e,
        self.i + self.dt * k3_i,
        drive,
    )
    next_e = self.e + self.dt * (k1_e + 2*k2_e + 2*k3_e + k4_e) / 6
    next_i = self.i + self.dt * (k1_i + 2*k2_i + 2*k3_i + k4_i) / 6
    self.e, self.i = validate_candidate(next_e, next_i)
    return self.e
```

Fixed-step RK4 integration, one update per call. **Returns float (E
rate), not binary spike.** This is a rate model, not a spiking model.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `e` | 0.1 | — | Excitatory rate (initial) |
| `i` | 0.05 | — | Inhibitory rate (initial) |
| `w_ee` | 10.0 | — | E→E recurrent excitation weight |
| `w_ei` | 6.0 | — | I→E cross-inhibition weight |
| `w_ie` | 10.0 | — | E→I feedforward excitation weight |
| `w_ii` | 1.0 | — | I→I recurrent inhibition weight |
| `tau_e` | 1.0 | ms | Excitatory time constant |
| `tau_i` | 2.0 | ms | Inhibitory time constant |
| `a` | 1.2 | — | Sigmoid gain (steepness) |
| `theta` | 4.0 | — | Sigmoid threshold (midpoint) |
| `dt` | 0.1 | ms | Integration timestep |

---

## Analytical Properties

### Sigmoid properties

- **At threshold:** $S(\theta) = 0.5 - \beta$, where
  $\beta = 1/(1 + \exp(a\theta))$ (verified by test).
- **At zero drive:** $S(0) = 0$ exactly because the baseline term is
  subtracted.
- **Range:** $S(x) \in [-\beta, 1-\beta]$ for all finite $x$.
- **Monotonic:** $S'(x) > 0$ — sigmoid is always increasing
- **Maximum slope:** $S'(\theta) = a/4$ — steepest at the midpoint
- **Gain controls transition:** Higher $a$ → sharper on/off switch.
  Lower $a$ → gradual graded response.

### Fixed points (nullclines)

Setting $dE/dt = 0$ and $dI/dt = 0$:

$$E^* = S(w_{EE} E^* - w_{EI} I^* + I_{ext})$$
$$I^* = S(w_{IE} E^* - w_{II} I^*)$$

These are transcendental equations (no closed-form solution). The number
of fixed points depends on the weight parameters:
- **One stable FP:** Low recurrence (w_ee < 1/a) → monostable
- **Three FPs:** Strong recurrence → bistable (two stable + one saddle)
- **Limit cycle:** E/I interaction with delay → oscillations

### Stability and oscillation conditions

The Jacobian at a fixed point $(E^*, I^*)$:

$$J = \begin{pmatrix} (-1 + w_{EE} S'_E)/\tau_E & -w_{EI} S'_E/\tau_E \\ w_{IE} S'_I/\tau_I & (-1 - w_{II} S'_I)/\tau_I \end{pmatrix}$$

Oscillations occur when $\text{tr}(J) < 0$ and $\det(J) > 0$ with complex
eigenvalues — i.e., when the E/I time constant ratio and weight magnitudes
create an oscillatory instability.

### Excitatory recurrence (w_ee)

Higher w_ee → higher positive feedback → higher E steady state:
- w_ee=5: weak recurrence, low E
- w_ee=15: high recurrence, high E
Verified by test.

### Inhibitory control (w_ei)

Higher w_ei → higher I→E suppression → lower E steady state:
- w_ei=3: weak inhibition, high E
- w_ei=10: high inhibition, low E
Verified by test.

### Steady-state convergence

At high constant input (I_ext=10), E and I converge to a stable fixed
point. After 10,000 steps, |ΔE| < 0.001 over the next 10,000 steps.
The sigmoid saturation (S → 1 for large arguments) guarantees bounded
behaviour.

---

## Behaviour

### E/I population dynamics

The Wilson-Cowan model captures the essential E/I interaction of a cortical
column at the mesoscopic level:

1. **External input → E increases:** S(... + I_ext) > S(... + 0)
2. **E → I follows:** w_ie·E enters the I sigmoid → I increases
3. **I → E suppresses:** w_ei·I subtracts from E input → E decreases
4. **Negative feedback loop:** E ↑ → I ↑ → E ↓ → I ↓ → E ↑ ...

This creates either:
- **Damped oscillation → fixed point** (default parameters)
- **Sustained oscillation** (high coupling: w_ee=16, w_ei=12, w_ie=15)

### Zero input: decay to low activity

Without external input, both E and I decay toward low values (E < 0.05,
I < 0.05). The sigmoid threshold θ=4.0 means that the internal
recurrence alone (w_ee×0.1 = 1.0 < θ) is insufficient to self-sustain
activity.

### E bounded in [0, 1]

The sigmoid output is in the published finite range
$[-\beta, 1-\beta]$. The implementation stages the complete RK4
candidate before mutation and accepts it only when both E and I remain
finite and inside that range.

### Oscillation

With enhanced coupling parameters (w_ee=16, w_ei=12, w_ie=15, θ=4.0)
and I_ext=5.0, the model can exhibit sustained oscillations. The
oscillation frequency depends on τ_e and τ_i — faster time constants
produce higher-frequency oscillations.

---

## Pipeline Compatibility

### Returns float, not int

**Critical limitation:** `step()` returns `float` (the excitatory rate E),
not `int` (binary spike). The SC-NeuroCore Network pipeline expects
`step() → int` for spike detection via Population.step_all().

When WilsonCowanUnit is placed in a Network:
- Population.step_all() calls step() for each neuron
- The returned float is cast to spike detection: any E > 0 registers as
  a "spike" — this is semantically incorrect
- SpikeMonitor counts will be inflated (every timestep with E > 0 = "spike")

**Recommended use:** Standalone simulation or with custom pipeline code
that interprets the returned E rate correctly. Not suitable for the
standard Population → Projection → SpikeMonitor pipeline without a
rate-to-spike conversion adapter.

### Population compatible

Population construction works: `Population(WilsonCowanUnit, n=10, label="wc")`
creates 10 independent Wilson-Cowan units.

---

## Comparison with Related Models

| Property | Wilson-Cowan | JansenRit | Siegert | LarterBreakspear |
|----------|-------------|-----------|---------|-----------------|
| Variables | 2 (E, I) | 3 (y0, y1, y2) | 1 (rate) | 3 (V, W, Z) |
| Type | Rate model | Neural mass | Mean-field | Neural mass |
| Activation | Sigmoid | Sigmoid | erf-based | tanh |
| Output | float (E rate) | float (EEG) | float (rate) | float |
| E/I | Explicit E, I vars | Implicit in y | Single pop | Ca, Na, K |
| Oscillation | Parameter-dependent | Intrinsic (alpha) | No | Chaotic possible |
| Pipeline | Float return (limited) | Float return (limited) | Float return (limited) | Float return (limited) |

All rate/neural mass models share the same pipeline limitation: they return
float, not binary spikes. The Wilson-Cowan model is the simplest and most
analytically tractable of the group.

---

## Historical Significance

Wilson & Cowan (1972) is one of the foundational papers in computational
neuroscience. It introduced the idea that the dynamics of cortical
populations can be described by coupled ODEs for excitatory and inhibitory
firing rates — a mean-field approximation that remains the basis of:

- Neural mass models (Jansen-Rit, David-Friston)
- Dynamic causal modelling (DCM) in neuroimaging
- Rate-based network models in theoretical neuroscience
- Population-level descriptions of cortical oscillations

The model predicts several key phenomena:
- E/I balance as a requirement for stable cortical activity
- Oscillatory instability from E/I interaction delays
- Hysteresis and bistability from recurrent excitation
- Gain control through inhibitory feedback

---

## Numerical Considerations

- **Fixed-step RK4:** The coupled E/I state advances through a staged
  fourth-order Runge-Kutta update with a shared external drive for the
  whole timestep.
- **Fail-closed candidate updates:** Python, Go, Julia, and Rust
  surfaces validate finite E/I state, non-negative coupling weights,
  positive time constants, positive sigmoid gain, positive timestep, and
  the Wilson-Cowan sigmoid range before mutation. Invalid runtime
  parameter mutation or non-finite external drive leaves the previous
  state intact and returns or raises the surface-specific failure signal.
- **dt stability:** Tested at dt = 0.05, 0.1, 0.2. All stable for
  10,000 steps under the default parameterisation.
- **Sigmoid overflow:** the scalar logistic implementation uses a
  sign-split form, so extreme finite drives saturate to the asymptotes
  without relying on platform overflow behaviour.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/wilson_cowan.py`.
- **Two state variables:** e (excitatory rate), i (inhibitory rate).
- **Dataclass:** Uses `@dataclass` for parameter storage.
- **Private sigmoid:** `_sigmoid(x)` method — shared sigmoid used by
  both E and I updates with different arguments.
- **Rust wiring:** Compatible for standalone dispatch but pipeline-limited
  (float return). Not in the Rust NeuronVariant enum.

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | 87,164 steps/s | 7,416,785 steps/s through PyO3 |
| Network | Limited (float return) | — |

The RK4 update evaluates four coupled derivative stages. Each stage
evaluates E and I sigmoid drives, giving eight sigmoid evaluations per
step in the Python reference. Native Rust, Julia, Go, and Mojo paths keep
the same arithmetic contract while reducing interpreter overhead.

---

## Test Evidence

| Category | What is verified |
|----------|-----------------|
| Isolation | defaults, float return, two-variable evolution, finite long run, reset |
| Sigmoid | threshold value, zero drive, monotonicity, published finite range |
| E/I dynamics | input response, inhibitory following, low-input decay, bounded state, steady-state convergence, RK4 reference point, recurrence and inhibition controls |
| Oscillation | enhanced-coupling finite state path |
| Parameters | constructor rejection, runtime corruption preservation, finite-drive saturation, timestep stability, determinism |
| Performance | isolation throughput remains above the documented floor |
| Pipeline | Population construction and float-return limitation |

See `tests/test_model_wilson_cowan.py`. No bugs found.

---

## Findings

1. **S(θ) = 0.5 exact:** The sigmoid midpoint equals the threshold
   parameter to machine precision.

2. **Sigmoid bounded and monotonic:** Verified across x ∈ [-100, 100].
   Always in (0, 1), always increasing.

3. **E increases with external input:** At I_ext=10, E > 0.5 after
   1000 steps. The excitatory drive raises the E population rate.

4. **I follows E:** I increases above initial 0.05 when E is driven —
   the w_ie coupling transfers excitatory activity to inhibition.

5. **Zero input → low activity:** Without input, E < 0.05 and I < 0.05
   after 10,000 steps. The recurrence alone is insufficient to self-sustain.

6. **w_ee controls E level:** Higher w_ee → higher E steady state,
   confirming the positive feedback loop.

7. **w_ei controls suppression:** Higher w_ei → lower E steady state,
   confirming the inhibitory feedback loop.

8. **Steady state convergence:** |ΔE| < 0.001 after 10K + 10K steps
   at I_ext=10. The system converges to a stable fixed point.

9. **Float return limitation documented:** The model returns float, not
   binary spike. Network pipeline interprets this incorrectly. This is
   inherent to rate models and is not a bug.

10. **Very fast performance:** ~163K steps/s — among the fastest models
    due to simple Euler step with 2 exp() calls and no sub-stepping.


---

## Theoretical Context

### Historical background

Wilson & Cowan (1972) published "Excitatory and inhibitory interactions
in localized populations of model neurons" in the *Biophysical Journal*.
This paper established the mathematical framework for describing the
mean-field dynamics of cortical populations as coupled ordinary
differential equations — a radical simplification from the individual
neuron level that remains the dominant paradigm in neural mass modelling.

The core insight is that the average firing rate of a large, densely
connected neural population can be described by a single variable
governed by a first-order ODE with a nonlinear (sigmoidal) activation
function. The sigmoid arises from the distribution of firing thresholds
across the population: at low mean input, few neurons exceed threshold;
at high input, nearly all fire.

### Influence on modern neuroscience

The Wilson-Cowan framework is the ancestor of:

- **Jansen-Rit model** (1995): extends Wilson-Cowan with post-synaptic
  potential kernels (second-order ODEs) and three populations (pyramidal,
  excitatory interneurons, inhibitory interneurons)
- **David-Friston model** (2003): generalises Jansen-Rit for DCM in
  neuroimaging, adding thalamocortical loops and laminar specificity
- **Spectral DCM** (Friston et al. 2012): linearised Wilson-Cowan
  dynamics embedded in a hierarchical Bayesian framework for resting-
  state fMRI analysis
- **Mean-field reductions** (Montbrió et al. 2015): exact reductions
  from spiking networks to Wilson-Cowan-like equations using the
  Ott-Antonsen ansatz

### E/I balance hypothesis

The Wilson-Cowan model provides the mathematical foundation for the
E/I balance hypothesis — the idea that cortical circuits operate near
the boundary between stable and oscillatory regimes. Disruptions of
this balance (too much E or too little I) correspond to pathological
states:

- **Excess E:** Seizure-like runaway excitation (epilepsy models)
- **Excess I:** Quiescence, loss of responsiveness
- **Balanced regime:** Asynchronous irregular activity matching cortical
  recordings

### Bifurcation analysis

The Wilson-Cowan system exhibits several bifurcation types as
coupling parameters vary:

1. **Saddle-node bifurcation:** Two stable fixed points emerge
   (bistability) as $w_{EE}$ increases past a critical value
2. **Hopf bifurcation:** A stable fixed point loses stability and a
   limit cycle is born as $w_{IE}$ increases — the onset of oscillations
3. **Saddle-node on invariant circle (SNIC):** Oscillations emerge
   from a saddle-node bifurcation on a limit cycle — produces
   excitable dynamics similar to Class I neurons

The default parameters ($w_{EE}=10, w_{EI}=6, w_{IE}=10, w_{II}=1$)
place the model near the Hopf boundary — a small increase in coupling
strength can induce oscillations, consistent with the cortical
operating point hypothesis.

### Connection to neural field theory

When the Wilson-Cowan equations are extended to include spatial
coordinates (Wilson & Cowan 1973), they become neural field equations:

$$\tau \frac{\partial u(x,t)}{\partial t} = -u + S\left(\int w(x-y) u(y,t) dy + I_{ext}\right)$$

These equations predict travelling waves, spiral waves, and Turing
patterns in cortical tissue — phenomena observed in voltage-sensitive
dye imaging and ECoG recordings.

### Adaptation and fatigue extensions

The original Wilson-Cowan model has no intrinsic adaptation. Several
extensions add a slow negative feedback variable:

- **Spike-frequency adaptation (SFA):** An additional variable $a$
  tracks cumulative E activity and subtracts from E input:
  $\tau_a \dot{a} = -a + c \cdot E$. This enables oscillatory bursting
  and slow wave patterns.
- **Synaptic depression:** A resource variable $x \in [0, 1]$ depletes
  with E activity and recovers on a slow timescale. This creates
  adaptation at the synaptic rather than intrinsic level.
- **Wilson-Cowan-Izhikevich hybrid:** Replacing the sigmoid with an
  Izhikevich-style quadratic nonlinearity yields exact mean-field
  reductions (Montbrió et al. 2015).

### Stochastic Wilson-Cowan

Adding multiplicative or additive noise to the Wilson-Cowan equations
yields a stochastic neural mass model:

$$\tau_E dE = (-E + S(\ldots)) dt + \sigma_E \sqrt{E(1-E)} dW_E$$

The multiplicative noise term $\sqrt{E(1-E)}$ respects the [0, 1]
bounds of the rate variable. Stochastic Wilson-Cowan models are used
in neuroimaging to model trial-to-trial variability and generate
synthetic EEG/MEG power spectra (Deco et al. 2008).

---

## Usage Examples

### Example 1: Basic E/I dynamics with external drive

```python
from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

wc = WilsonCowanUnit()

# Drive with external input for 2000 steps
e_trace = []
i_trace = []
for t in range(2000):
    wc.step(ext_input=5.0)
    e_trace.append(wc.e)
    i_trace.append(wc.i)

print(f"Final E: {wc.e:.4f}, Final I: {wc.i:.4f}")
print(f"E range: [{min(e_trace):.4f}, {max(e_trace):.4f}]")
```

### Example 2: Oscillatory regime with high coupling

```python
from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

# Enhanced coupling for oscillations
wc = WilsonCowanUnit(w_ee=16.0, w_ei=12.0, w_ie=15.0, w_ii=1.0)

e_trace = []
for t in range(5000):
    wc.step(ext_input=5.0)
    e_trace.append(wc.e)

# Check for oscillations: variance should be non-trivial
import numpy as np
variance = np.var(e_trace[1000:])  # skip transient
print(f"E variance: {variance:.6f}")
print(f"Oscillatory: {'Yes' if variance > 0.001 else 'No'}")
```

### Example 3: Bistability demonstration

```python
from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

# Strong recurrence for bistability
wc_low = WilsonCowanUnit(w_ee=15.0, e=0.01)  # start low
wc_high = WilsonCowanUnit(w_ee=15.0, e=0.9)  # start high

for t in range(5000):
    wc_low.step(ext_input=3.0)
    wc_high.step(ext_input=3.0)

print(f"From low init:  E = {wc_low.e:.4f}")
print(f"From high init: E = {wc_high.e:.4f}")
print(f"Different attractors: {abs(wc_low.e - wc_high.e) > 0.1}")
```

---

## Technical Reference

### Rust parity

| Aspect | Python | Rust | Status |
|--------|--------|------|--------|
| State variables | e, i (rates) | same | matched |
| Sigmoid function | two-term Wilson-Cowan form | same | matched |
| RK4 integration | four coupled stages | same | matched |
| All defaults | identical | identical | matched |

No parity defects were observed in the current automated parity suite.

### Source files

| File | Description |
|------|-------------|
| `src/sc_neurocore/neurons/models/wilson_cowan.py` | Python reference |
| `engine/src/wilson_cowan.rs` | Rust PyO3 multi-step simulator |
| `engine/src/neurons/special.rs` | Direct Rust neuron mirror |
| `tests/test_model_wilson_cowan.py` | Module-specific Python tests |

---

## Multi-language acceleration chain

### Kernel sources

| Backend | Source file | Binding |
|---------|-------------|---------|
| Python primary   | `src/sc_neurocore/neurons/models/wilson_cowan.py`             | — (reference) |
| Rust (PyO3)      | `engine/src/wilson_cowan.rs`                                  | `sc_neurocore_engine.py_wilson_cowan_simulate` |
| Julia (juliacall)| `src/sc_neurocore/accel/julia/neurons/wilson_cowan.jl`        | `sc_neurocore.accel.julia.neurons.simulate_wilson_cowan` |
| Go (cgo)         | `src/sc_neurocore/accel/go/wilson_cowan/wilson_cowan.go`      | `sc_neurocore.accel.go.wilson_cowan.simulate_wilson_cowan` |
| Mojo (FFI)       | `src/sc_neurocore/accel/mojo/wilson_cowan/wilson_cowan.mojo`  | `sc_neurocore.accel.mojo.wilson_cowan.simulate_wilson_cowan` |

Wilson-Cowan is deterministic (no stochastic noise) so all five
backends produce identical trajectories to machine epsilon given the
same external-input array.

### Multi-backend performance

Measured on local i5-11600K, `N = 100 000` steps, `benchmarks/
bench_wilson_cowan.py`. Numbers trace back to
`benchmarks/results/bench_wilson_cowan.json` committed alongside.

| Backend | Steps/s | Wall (ms) | Speedup vs Python | Parity vs Rust |
|---------|--------:|----------:|------------------:|---------------:|
| Python primary | 87 164 | 1147.26 |   1.00× | — |
| Rust PyO3      | 7 416 785 |  13.48 |  85.09× | reference |
| Julia (warm)   | 4 786 245 |  20.89 |  54.91× | Δ ≈ 1.1e-16 |
| Go cgo         | 3 201 920 |  31.23 |  36.73× | Δ ≈ 2.2e-16 |
| Mojo FFI       | 4 893 367 |  20.44 |  56.14× | Δ ≈ 7.1e-11 (libm vs f64::exp) |

### Backends

| Backend | Status | Rationale |
|---------|--------|-----------|
| Rust PyO3 | **USED** | default `auto` path; fastest measured + zero parity drift |
| Julia     | USED   | warm path ~65 % of Rust; preferred when juliacall is hot |
| Go cgo    | USED   | simplest cross-platform .so, slight FFI overhead |
| Mojo FFI  | USED   | warm path ~66 % of Rust; libm-exp ulp drift tolerated |

### Tests

| Backend | File | Tests | What is verified |
|---------|------|------:|------------------|
| Rust    | `tests/test_wilson_cowan_parity.py` (+ `engine/src/wilson_cowan.rs::tests`) | 16 | sigmoid regime + asymptotes, quiescent convergence, high-drive elevation, output-shape, length-panic, Python↔Rust bit-exact under 4 drive patterns, zero-length + one-step edge |
| Julia   | `tests/test_wilson_cowan_julia_parity.py` | 4 | Python↔Julia bit-exact, Rust↔Julia cross |
| Go      | `tests/test_wilson_cowan_go_parity.py`    | 4 | Python↔Go bit-exact, Rust↔Go cross |
| Mojo    | `tests/test_wilson_cowan_mojo_parity.py`  | 3 | Python↔Mojo within libm ulp drift, Rust↔Mojo |

### Sophisticated dynamics (`tests/test_wilson_cowan_dynamics.py`, 23)

Covers published properties beyond API parity: sigmoid regime (5),
quiescent fixed point (2), monotone response, time-constant
separation (τ_e < τ_i → E settles before I), **limit-cycle
oscillator regime** from Wilson-Cowan 1972 Fig 3 (repeated
zero-crossings around the post-transient mean — only possible with
the correct two-term sigmoid), bounded state over 5 extreme drives,
parameter sweep monotonicity, extreme-param cross-backend parity
(5 regimes), edge cases.

---

## Citations

1. Wilson HR, Cowan JD (1972). Excitatory and inhibitory interactions
   in localized populations of model neurons. *Biophys J* 12(1):1–24.
   DOI: [10.1016/S0006-3495(72)86068-5](https://doi.org/10.1016/S0006-3495(72)86068-5)

2. Wilson HR, Cowan JD (1973). A mathematical theory of the functional
   dynamics of cortical and thalamic nervous tissue. *Kybernetik*
   13(2):55–80.
   DOI: [10.1007/BF00288786](https://doi.org/10.1007/BF00288786)

3. Jansen BH, Rit VG (1995). Electroencephalogram and visual evoked
   potential generation in a mathematical model of coupled cortical
   columns. *Biol Cybern* 73(4):357–366.
   DOI: [10.1007/BF00199471](https://doi.org/10.1007/BF00199471)

4. Destexhe A, Sejnowski TJ (2009). The Wilson-Cowan model, 36 years
   later. *Biol Cybern* 101(1):1–2.
   DOI: [10.1007/s00422-009-0328-3](https://doi.org/10.1007/s00422-009-0328-3)

5. Montbrió E, Pazó D, Roxin A (2015). Macroscopic description for
   networks of spiking neurons. *Phys Rev X* 5(2):021028.
   DOI: [10.1103/PhysRevX.5.021028](https://doi.org/10.1103/PhysRevX.5.021028)

6. Deco G, Jirsa VK, Robinson PA, Breakspear M, Friston K (2008).
   The dynamic brain: from spiking neurons to neural masses and
   cortical fields. *PLoS Comput Biol* 4(8):e1000092.
   DOI: [10.1371/journal.pcbi.1000092](https://doi.org/10.1371/journal.pcbi.1000092)

---

---

## Limitations

- **No spike output:** Returns float rate, not binary spike. The
  standard Network pipeline misinterprets this as "always spiking"
  for any E > 0. Requires a rate-to-spike adapter for network use.
- **No adaptation:** The model has no intrinsic adaptation or
  fatigue mechanism. Sustained input produces sustained output
  with no accommodation.
- **No spatial structure:** Each unit is a point model. For spatial
  cortical dynamics, use the neural field extension (Wilson & Cowan
  1973) or couple multiple units with distance-dependent weights.
- **Single sigmoid:** Both E and I populations share the same sigmoid
  parameters (a, θ). Biological E and I populations may have
  different F-I curves.
- **No delays:** Axonal conduction delays are absent. For oscillatory
  dynamics that depend on propagation time, explicit delay terms
  are needed.

---

**ALL 21 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**
**Rust parity: EXACT (no defects found).**
**Criterion: 3.0 ms / 100K steps (30.5 ns/step, ~32.8M steps/s).**
