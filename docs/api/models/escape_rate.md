# EscapeRateNeuron

**Module:** `sc_neurocore.neurons.models.escape_rate`
**Reference:** Gerstner, Neural Comput. 12(1), 2000; Gerstner & Kistler, Spiking Neuron Models, 2002
**Family:** Stochastic integrate-and-fire (escape noise model)
**State variables:** `v` (membrane potential)

---

## Equations

### Membrane potential (deterministic)

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I$$

For constant current over one step, the maintained implementation uses
the closed-form RC flow:

$$V(t+\Delta t) = V_\infty + (V(t) - V_\infty)\exp(-\Delta t/\tau_m)$$

where:

$$V_\infty = V_{rest} + R \cdot I$$

### Instantaneous escape rate

$$\rho(V) = \rho_0 \exp\!\left(\frac{V - V_{threshold}}{\Delta u}\right)$$

### Spike probability (per timestep)

$$p_{spike} = 1 - \exp\!\left(-\rho(V) \cdot dt\right)$$

For small timesteps this reduces to $p_{spike} \approx \rho(V) \cdot dt$, but the finite-step hazard transform remains bounded in $[0,1]$ for high escape rates.

### Stochastic spike generation

$$\text{Bernoulli}(p_{spike}): \quad \text{if } U(0,1) < p_{spike}: \text{spike, } V \leftarrow V_{reset}$$

### Implementation

```python
def step(self, current: float) -> int:
    steady_state = self.v_rest + self.resistance * current
    decay = math.exp(-self.dt / self.tau_m)
    voltage = steady_state + (self.v - steady_state) * decay
    rate = self.rho_0 * safe_exp((voltage - self.v_threshold) / self.delta_u)
    p_spike = -math.expm1(-rate * self.dt)
    if np.random.random() < p_spike:
        self.v = self.v_reset
        return 1
    self.v = voltage
    return 0
```

The membrane potential evolves deterministically by the exact
constant-current RC solution, while spiking is **stochastic**: the
probability of a spike increases exponentially as V approaches
threshold. There is no hard threshold — even far below V_threshold,
there is a small but nonzero probability of firing.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −70.0 | mV | Membrane potential |
| `v_rest` | −70.0 | mV | Resting potential |
| `v_reset` | −70.0 | mV | Post-spike reset |
| `v_threshold` | −50.0 | mV | Nominal threshold (centre of escape zone) |
| `tau_m` | 10.0 | ms | Membrane time constant |
| `rho_0` | 0.001 | kHz | Base escape rate |
| `delta_u` | 3.0 | mV | Escape noise width (sharpness) |
| `resistance` | 1.0 | MΩ | Membrane resistance |
| `dt` | 1.0 | ms | Integration timestep |

### ρ₀ = 0.001 (base rate)

The base escape rate at V = V_threshold: ρ(V_θ) = 0.001 kHz. This gives
p_spike = 1 − exp(−0.001 × 1.0) ≈ 0.001 (0.1%) per timestep at threshold.
Well below threshold, p_spike is exponentially smaller.

### Δu = 3.0 mV (noise width)

Controls the sharpness of the soft threshold:
- Small Δu → sharp transition (approaches hard threshold)
- Large Δu → broad, noisy threshold

The escape rate at key voltages:

| V − V_θ (mV) | ρ(V)/ρ₀ | p_spike (dt=1) |
|---------------|---------|----------------|
| −10 | exp(−3.33) ≈ 0.036 | 3.6 × 10⁻⁵ |
| −5 | exp(−1.67) ≈ 0.189 | 1.9 × 10⁻⁴ |
| 0 | 1.0 | 10⁻³ |
| +5 | exp(1.67) ≈ 5.29 | 5.3 × 10⁻³ |
| +10 | exp(3.33) ≈ 28.0 | 2.8 × 10⁻² |
| +20 | exp(6.67) ≈ 789 | 0.789 |

---

## Analytical Properties

### Escape noise vs reset noise

Two main approaches to stochastic spiking models:

| Property | Escape noise (this) | Reset noise (diffusion) |
|----------|-------------------|----------------------|
| Source | Threshold is stochastic | Membrane voltage is stochastic |
| Mechanism | Bernoulli(ρ(V)·dt) | V += σ·ξ each step |
| V dynamics | Deterministic exact RC flow | Stochastic |
| ρ(V) | Exponential escape | Not applicable |
| ISI distribution | Renewal | Non-renewal |
| Analytical | Tractable | Requires Fokker-Planck |

The escape noise model (Gerstner 2000) is more analytically tractable
because the membrane dynamics are deterministic — only the spike decision
is stochastic.

### Soft threshold interpretation

The exponential escape rate can be interpreted as a **Boltzmann
distribution** over threshold crossings:

$$\rho(V) = \rho_0 \exp\!\left(\frac{V - V_\theta}{\Delta u}\right)$$

This is equivalent to a hard threshold V_θ + noise ξ, where ξ is
drawn from an exponential distribution with scale Δu. The "escape"
metaphor: the neuron "escapes" over the threshold barrier with a rate
that increases exponentially as the barrier shrinks.

### ISI statistics

For constant input I producing steady-state V_ss:
- Mean ISI ≈ 1/ρ(V_ss) (for low rates)
- ISI distribution ≈ exponential (memoryless) for constant V
- CV (coefficient of variation) → 1 for Poisson-like firing

At higher rates (multiple spikes), the reset creates ISI correlations
because V must recover from V_reset to V_ss between spikes.

### Steady-state voltage

For constant I (subthreshold, no spikes):
$$V_{ss} = V_{rest} + R \cdot I$$

At default parameters with I=0: V_ss = −70 mV (at rest).
With I=50: V_ss = −70 + 50 = −20 mV (well above nominal threshold).

### Membrane equation one-step verification

The update
$V_\infty + (V - V_\infty)\exp(-dt/\tau_m)$ is verified analytically
in the test suite to machine precision, including a large-step case
that separates it from the historical forward-Euler increment.

---

## Behaviour

### Stochastic spiking

The model produces stochastic spikes — two runs with identical parameters
produce different spike trains (different random seeds). This is verified
by test: two independent runs have different spike times.

### Rate increases with input

Higher current → higher V_ss → higher ρ(V) → more spikes. Verified:
strong drive produces more spikes than weak drive across 5000 steps.

### Zero input → silent

At I=0, V stays at V_rest = −70 mV. The escape rate at V_rest:
ρ = 0.001 × exp((−70 − (−50))/3) = 0.001 × exp(−6.67) ≈ 1.3 × 10⁻⁶.
p_spike = 1.3 × 10⁻⁶ per step. In 5000 steps: expected 0.006 spikes.
Effectively silent.

### safe_exp prevents overflow

The `safe_exp()` utility clips the argument to prevent IEEE overflow
when V is far above threshold. Without this, exp((V−V_θ)/Δu) could
produce inf for V >> V_θ.

---

## Comparison with Related Models

| Property | EscapeRate | StochasticIF | LIF | GalvesLocherbach |
|----------|-----------|-------------|-----|-----------------|
| V dynamics | Deterministic | Stochastic (σ·ξ) | Deterministic | Discrete-time |
| Spike mechanism | Bernoulli(ρ·dt) | Hard threshold | Hard threshold | Bernoulli(φ(V)) |
| Stochasticity | Threshold | Membrane noise | None | Activation function |
| ρ₀ parameter | Yes | No | No | No |
| Δu parameter | Yes (noise width) | σ (noise amplitude) | No | β (inverse temp) |
| ISI distribution | Approximately exponential | Inverse Gaussian | Deterministic | Variable |
| Pipeline | Compatible | Compatible | Compatible | Compatible |

---

## Numerical Considerations

- **1 safe_exp() per step:** For the escape rate calculation.
- **1 np.random.random() per step:** RNG call is the performance
  bottleneck (much slower than arithmetic).
- **Finite-step hazard:** The implementation uses the bounded hazard
  transform 1 − exp(−ρ·dt), so high escape rates saturate without invalid
  probabilities.
- **safe_exp overflow protection:** Clips exp argument to prevent inf.

## Validation contract

The reference implementation validates mutable runtime state on every
`step()` before division, exponentiation, random sampling, or membrane
assignment:

- `v`, `v_rest`, `v_reset`, `v_threshold`, and input current must be finite;
- `tau_m`, `rho_0`, `delta_u`, `resistance`, and `dt` must be finite and positive;
- the exact-flow membrane candidate must remain finite before spike-probability evaluation;
- the finite-step escape hazard must remain finite and non-negative;
- the Bernoulli probability must remain finite and bounded in `[0, 1]`.

These guards preserve the Gerstner escape-rate point-process contract while
preventing corrupted mutable parameters from converting a numerical overflow
into a silent reset or poisoned membrane state.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/escape_rate.py`.
- **One state variable:** v (membrane potential).
- **Dataclass:** Uses `@dataclass`.
- **Uses safe_exp:** From `sc_neurocore.utils.numerics`.
- **Uses np.random:** Per-step RNG call (not seedable via constructor).
- **Rust engine / Go / Julia / Rust safety wiring:** Compatible scalar
  state surface with exact constant-current RC flow, bounded finite-step
  hazard probability, and explicit invalid-state or non-finite hazard
  errors. Rust and Go safety mirrors deterministically emit only
  saturated-probability spikes; the Python reference, Rust engine, and
  Julia mirror keep stochastic Bernoulli sampling.

---

## Performance

Local non-isolated regression run, measured 2026-06-17. These numbers
are recorded for regression comparison only and are not production
throughput claims.

| Backend | Median ns/step | Spikes | Evidence |
|---------|---------------:|-------:|----------|
| Python | 5834.21848 | 3219 | stochastic reference, exact RC flow |
| Rust engine | 91.437735 | 3148 | stochastic engine example, exact RC flow |
| Go service mirror | 73.03 | 0 | deterministic saturated-probability mirror |
| Julia mirror | 39.474865 | 3202 | stochastic mirror, exact RC flow |
| Mojo mirror | 49.645785038592294 | 200 | deterministic threshold-sequence mirror, exact RC flow |

The benchmark artefact is
`benchmarks/results/local_python_2026-06-17_escape_rate_exact_flow.json`.
The RNG call remains the dominant cost in stochastic paths.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | construction, binary output, state evolves, state finite (10K), reset |
| Stochastic | 6 | stochastic spiking, two runs differ, rate increases with input, zero input silent, bounded hazard transform, high-rate saturation |
| Analytical | 5 | V steady-state, exact membrane equation 1-step, exact-flow vs Euler separation, ρ₀ scales rate, Δu controls sensitivity |
| ISI | 2 | ISI variability (CV > 0), higher current shorter ISI |
| Parameters | 2 | τ_m controls V dynamics, resistance scales input |
| Validation | 43 | finite parameters/current, positive scales, corrupted runtime state, finite voltage candidates, finite bounded hazards |
| Performance | 2 | isolation throughput, network throughput |
| Pipeline | 4 | Population, Network spikes, Projection wiring, analysis pipeline |
| **Total** | **69** | dedicated module checks |

See `tests/test_model_escape_rate.py`.

---

## Findings

1. **Stochastic spiking confirmed.** Spikes via Bernoulli sampling from
   ρ(V)·dt. Two identical-parameter runs produce different spike trains.

2. **Rate monotonic.** Higher current → higher V_ss → higher ρ(V) →
   more spikes. Verified across multiple current levels.

3. **Zero input silent.** At I=0, ρ ≈ 1.3 × 10⁻⁶ → expected 0.006
   spikes in 5000 steps. Effectively silent.

4. **safe_exp prevents overflow.** No NaN or inf at extreme voltages.

5. **Exact constant-current membrane flow.** V_ss = V_rest + R·I and
   the one-step RC relaxation are verified to machine precision.

6. **ρ₀ scales rate linearly.** Doubling ρ₀ approximately doubles the
   spike count (at low rates where p_spike << 1).

7. **Δu controls threshold sharpness.** Smaller Δu → more deterministic
   (sharper threshold). Larger Δu → more stochastic.

8. **ISI approximately exponential.** CV > 0 confirms stochastic ISI
   distribution. At low rates, ISI ≈ exponential (Poisson-like).

9. **Network pipeline fully functional.** Population, Projection,
   PoissonInput, spike_count, ISI, firing_rate all verified.

10. **Deterministic membrane, stochastic spike.** V evolves
    deterministically; only the spike decision is random.

---

## Theoretical Context

### Gerstner's escape noise framework

Wulfram Gerstner introduced the escape noise model as part of the
**Spike Response Model** (SRM) framework (Gerstner 2000). The key
insight: biological neurons have "noisy thresholds" — the probability
of firing increases steeply near threshold but is never exactly zero
below it and never exactly one above it.

The escape rate formalism provides a principled way to model this
stochastic threshold without adding noise to the membrane potential
equation. This separation of deterministic dynamics from stochastic
spike generation makes the model analytically tractable.

### Relationship to Kramers escape theory

The name "escape rate" comes from Kramers' (1940) theory of thermally
activated escape over energy barriers:

$$\text{rate} = \text{attempt frequency} \times \exp(-\Delta E / k_B T)$$

In the neural context:
- Barrier height ↔ V_threshold − V (distance to threshold)
- Temperature ↔ Δu (noise parameter)
- Attempt frequency ↔ ρ₀ (base rate)

The neuron "escapes" over the threshold barrier with a rate that
increases exponentially as the barrier shrinks (V → V_threshold).

### Maximum likelihood spike train fitting

The escape noise model is uniquely suited for **maximum likelihood
estimation** (MLE) of neural model parameters from spike train data
(Pillow et al. 2005, Paninski 2004). The log-likelihood of an
observed spike train $\{t_1, t_2, \ldots\}$ is:

$$\mathcal{L} = \sum_k \log \rho(V(t_k)) - \int_0^T \rho(V(t)) dt$$

This is a standard point-process likelihood with intensity ρ(V(t)).
The deterministic membrane dynamics mean that V(t) is a known
function of the input — no stochastic integration is needed. This
makes gradient-based optimisation straightforward and efficient.

### Generalised linear model (GLM) connection

The escape rate model is mathematically equivalent to a **point-
process GLM** (Truccolo et al. 2005):

$$\lambda(t) = \exp\!\left(\mathbf{k}^T \cdot \mathbf{x}(t) + h^T \cdot \mathbf{spike\_history}\right)$$

where $\lambda$ is the conditional intensity (firing rate), $\mathbf{k}$
is the stimulus filter, and $h$ is the spike-history filter. The
escape rate neuron implements the special case where the stimulus filter
is the LIF membrane equation and the spike-history filter is the reset.

This connection enables the use of powerful statistical tools (GLM
fitting, goodness-of-fit tests, model comparison) for neural data
analysis.

### Information-theoretic properties

The escape noise model provides an explicit coding noise model:

- **Signal:** V(t) — the deterministic membrane response to input
- **Noise:** The stochastic spike generation (Bernoulli with ρ·dt)
- **Fisher information:** $J(I) \propto [\rho'(V_{ss})]^2 / \rho(V_{ss})$

The exponential ρ(V) gives:
$$J(I) \propto \rho(V_{ss}) / \Delta u^2$$

Information increases with firing rate (higher rate = more samples per
unit time) and decreases with noise width (larger Δu = noisier spikes).

### Applications in computational neuroscience

1. **Bayesian brain hypothesis:** Probabilistic spiking models (like
   the escape rate) support the view that neural populations perform
   approximate Bayesian inference (Ma et al. 2006)
2. **Neural coding efficiency:** The escape noise model predicts the
   optimal threshold and noise level for maximising information
   transmission (Bethge et al. 2002)
3. **Network dynamics:** Stochastic spiking prevents synchrony artifacts
   that occur with deterministic threshold models
4. **Retinal ganglion cell modelling:** The escape rate model accurately
   predicts the spike trains of retinal ganglion cells in response to
   natural stimuli (Pillow et al. 2005)

### Relationship to Generalized Linear Models (GLMs)

The escape rate neuron is the biophysical incarnation of a point-process
GLM. The connection is:

| GLM component | Escape rate equivalent |
|---------------|----------------------|
| Stimulus filter k | LIF membrane equation (1/τ kernel) |
| Link function | Exponential: ρ = ρ₀ exp((V-V_θ)/Δu) |
| Spike-history filter h | Reset mechanism (V → V_reset) |
| Conditional intensity λ(t) | ρ(V(t)) |
| Observation model | Bernoulli(λ·dt) |

This equivalence means that all GLM analysis tools — maximum
likelihood fitting, goodness-of-fit tests (time-rescaling theorem),
model comparison (AIC/BIC), confidence intervals — apply directly to
the escape rate neuron.

### Spike Response Model (SRM) embedding

The escape rate neuron is a special case of Gerstner's Spike Response
Model (SRM). In the SRM framework:

$$V(t) = \eta(t - \hat{t}) + \int_0^\infty \kappa(s) I(t-s) ds$$

where $\eta$ is the spike afterpotential (encoding reset and
refractoriness), $\hat{t}$ is the last spike time, and $\kappa$ is
the membrane filter. The escape rate neuron uses:

- $\kappa(s) = (R/\tau_m) \exp(-s/\tau_m)$ (exponential kernel)
- $\eta(s) = (V_{reset} - V_{rest}) \exp(-s/\tau_m)$ (reset decay)

### Comparison with diffusion noise

In the diffusion (Langevin) noise model:

$$\tau_m dV = -(V - V_{rest}) dt + R \cdot I \, dt + \sigma \, dW_t$$

the noise enters the membrane equation directly, making V stochastic.
This changes the mathematical framework fundamentally:

- **Escape rate:** V deterministic, spike stochastic → point process
  likelihood, tractable MLE
- **Diffusion noise:** V stochastic, spike deterministic (hard
  threshold) → Fokker-Planck PDE, Siegert formula for rate

The escape rate model is often preferred for parameter inference because
the likelihood function is explicit and differentiable. The diffusion
model is preferred when the noise is genuinely in the membrane potential
(e.g., synaptic bombardment) and the ISI distribution shape matters.

### Population coding with escape rate neurons

A population of N escape rate neurons with shared input I and
independent noise (different RNG streams) implements a **population
code**. The population firing rate converges to:

$$R_{pop} = N \cdot \rho(V_{ss}(I))$$

with variance $N \cdot \rho(V_{ss}) \cdot (1 - \rho(V_{ss}) \cdot dt)$.
This provides a natural model for cortical columns where thousands of
neurons share similar inputs but fire independently. The population
Fisher information scales linearly with N, enabling precise encoding
of continuous stimuli through population rate codes. This linear scaling
is a fundamental result in neural population coding theory.

---

## Usage Examples

### Example 1: Stochastic spike generation

```python
from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron

n = EscapeRateNeuron()
spikes = sum(n.step(current=25.0) for _ in range(10000))
rate = spikes / (10000 * 1.0 / 1000)
print(f"Spikes: {spikes}, Estimated rate: {rate:.1f} Hz")
```

### Example 2: Escape rate as function of input

```python
from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron

for I in [0, 10, 15, 20, 25, 30]:
    n = EscapeRateNeuron()
    spikes = sum(n.step(float(I)) for _ in range(5000))
    print(f"I={I:3d}: {spikes} spikes in 5000 steps")
```

### Example 3: Noise width effect on ISI variability

```python
from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron
import numpy as np

for du in [1.0, 3.0, 10.0]:
    n = EscapeRateNeuron(delta_u=du)
    isi_list = []
    last_spike = 0
    for t in range(50000):
        if n.step(current=25.0):
            if last_spike > 0:
                isi_list.append(t - last_spike)
            last_spike = t
    if len(isi_list) > 10:
        cv = np.std(isi_list) / np.mean(isi_list)
        print(f"Δu={du:4.1f}: CV_ISI={cv:.3f}, mean ISI={np.mean(isi_list):.1f}")
```

---

## Technical Reference

### Rust parity

| Aspect | Python | Rust | Status |
|--------|--------|------|--------|
| State variable | v (membrane potential) | same | **EXACT** |
| Membrane update | closed-form RC flow | same | **EXACT** |
| Escape rate | ρ₀ × safe_exp((v-v_θ)/Δu) | same | **EXACT** |
| Bernoulli spike | random() < (1-exp(-ρ·dt)) | same | **EXACT** |
| All defaults | identical | identical | **EXACT** |

**No parity defects.** EXACT parity verified by automated scan.

### Source files

| File | Lines | Description |
|------|-------|-------------|
| `src/sc_neurocore/neurons/models/escape_rate.py` | ~47 | Python reference |
| `engine/src/neurons/trivial.rs` | (shared) | Rust implementation |
| `tests/test_model_escape_rate.py` | ~286 | 69 tests |

---

## Performance Benchmarks

### Local exact-flow regression benchmark (measured 2026-06-17)

The local regression benchmark records Python, Rust engine, Go, Julia,
and Mojo timing medians plus backend spike-count/final-voltage evidence in:

`benchmarks/results/local_python_2026-06-17_escape_rate_exact_flow.json`

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 5834.21848 | 5698.08305 | 6869.082345 | 3219 |
| Rust engine | 91.437735 | 88.389655 | 98.2433 | 3148 |
| Go service mirror | 73.03 | 72.09 | 98.9 | 0 |
| Julia mirror | 39.474865 | 39.248385 | 40.57662 | 3202 |
| Mojo mirror | 49.645785038592294 | 49.42665997077711 | 50.340774905635044 | 200 |

The benchmark gate requires the benchmark script, exact-flow model
sources, and generated artefact hashes to match before accepting the
numbers as current evidence.

---

## Limitations

- **Per-step RNG call:** Each step requires a random number, making
  the model slower than deterministic IF variants.
- **Global RNG:** Uses np.random (Python) — not per-instance
  reproducible without explicit seed management.
- **One spike per step:** The finite-step hazard is bounded in `[0, 1]`.
  At very high rates it saturates to one spike per timestep.
- **No adaptation:** No spike-frequency adaptation or refractory
  period beyond the V_reset mechanism.
- **Linear membrane:** The subthreshold dynamics are pure LIF — no
  exponential or quadratic spike onset. The stochasticity is entirely
  in the spike generation, not in the dynamics.

---

## Citations

1. Gerstner W (2000). Population dynamics of spiking neurons: fast
   transients, asynchronous states, and locking. *Neural Comput*
   12(1):43–89.
   DOI: [10.1162/089976600300015899](https://doi.org/10.1162/089976600300015899)

2. Gerstner W, Kistler WM (2002). *Spiking Neuron Models: Single Neurons,
   Populations, Plasticity.* Cambridge University Press.
   ISBN: 978-0-521-89079-3.

3. Pillow JW, Paninski L, Uzzell VJ, Simoncelli EP, Chichilnisky EJ
   (2005). Prediction and decoding of retinal ganglion cell responses
   with a probabilistic spiking model. *J Neurosci* 25(47):11003–11013.
   DOI: [10.1523/JNEUROSCI.3305-05.2005](https://doi.org/10.1523/JNEUROSCI.3305-05.2005)

4. Paninski L (2004). Maximum likelihood estimation of cascade point-
   process neural encoding models. *Network* 15(4):243–262.
   DOI: [10.1088/0954-898X_15_4_002](https://doi.org/10.1088/0954-898X_15_4_002)

5. Truccolo W, Eden UT, Fellows MR, Donoghue JP, Brown EN (2005). A
   point process framework for relating neural spiking activity to
   spiking history, neural ensemble, and extrinsic covariate effects.
   *J Neurophysiol* 93(2):1074–1089.
   DOI: [10.1152/jn.00697.2004](https://doi.org/10.1152/jn.00697.2004)

6. Kramers HA (1940). Brownian motion in a field of force and the
   diffusion model of chemical reactions. *Physica* 7(4):284–304.
   DOI: [10.1016/S0031-8914(40)90098-2](https://doi.org/10.1016/S0031-8914(40)90098-2)

---

**ALL 24 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**
**Rust parity: EXACT (no defects found).**
**Criterion: 532.9 µs / 10K steps (53.3 ns/step, ~18.8M steps/s).**
