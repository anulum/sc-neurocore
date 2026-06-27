# WongWangUnit

**Module:** `sc_neurocore.neurons.models.wong_wang`
**Reference:** Wong & Wang, J. Neurosci. 26(4), 2006
**Family:** Reduced mean-field attractor model (decision-making)
**State variables:** `s1`, `s2` (NMDA synaptic gating variables for two competing pools)

---

## Equations

### Two-pool NMDA dynamics

$$\frac{ds_1}{dt} = -\frac{s_1}{\tau_s} + (1 - s_1) \gamma r_1$$

$$\frac{ds_2}{dt} = -\frac{s_2}{\tau_s} + (1 - s_2) \gamma r_2$$

### Synaptic currents (with cross-inhibition)

$$I_1 = J_N s_1 - J_{cross} s_2 + I_0 + \text{stim}_1 + \sigma \xi_1$$

$$I_2 = J_N s_2 - J_{cross} s_1 + I_0 + \text{stim}_2 + \sigma \xi_2$$

### Input-output transfer function (Abbott-Chance)

$$r = \Phi(I_{syn}) = \frac{aI_{syn} - b}{1 - \exp(-d(aI_{syn} - b))}$$

where $a = 270$ Hz/nA, $b = 108$ Hz, $d = 0.154$ s.

At the singularity ($aI - b \approx 0$): $\Phi \rightarrow 1/d \approx 6.5$ Hz.

### Output

Returns `(r1, r2)` — firing rates (Hz) of both pools. **Returns tuple of
floats, not binary spike.**

### Implementation

```python
def step(self, stim1=0.0, stim2=0.0) -> tuple[float, float]:
    i1 = j_n * s1 - j_cross * s2 + i_0 + stim1 + sigma * randn()
    i2 = j_n * s2 - j_cross * s1 + i_0 + stim2 + sigma * randn()
    r1, r2 = phi(i1), phi(i2)
    k1 = f(s1, s2, stim1, stim2, noise1, noise2)
    k2 = f(s1 + 0.5*dt*k1.s1, s2 + 0.5*dt*k1.s2, stim1, stim2, noise1, noise2)
    k3 = f(s1 + 0.5*dt*k2.s1, s2 + 0.5*dt*k2.s2, stim1, stim2, noise1, noise2)
    k4 = f(s1 + dt*k3.s1, s2 + dt*k3.s2, stim1, stim2, noise1, noise2)
    s1 += dt * (k1.s1 + 2*k2.s1 + 2*k3.s1 + k4.s1) / 6
    s2 += dt * (k1.s2 + 2*k2.s2 + 2*k3.s2 + k4.s2) / 6
    s1, s2 = clip([s1, s2], 0, 1)
    return (r1, r2)
```

Fixed-step RK4 over the coupled two-state ODE, with one Gaussian noise
sample per pool held constant inside the step. The returned firing rates
`(r1, r2)` are the start-of-step rates, preserving the historical public
surface while the state update uses the higher-order candidate.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `s1` | 0.1 | — | NMDA gating variable, pool 1 |
| `s2` | 0.1 | — | NMDA gating variable, pool 2 |
| `tau_s` | 0.1 | s | NMDA decay time constant (100 ms) |
| `gamma` | 0.641 | — | Saturation factor (kinetic parameter) |
| `j_n` | 0.2609 | nA | Same-pool recurrent NMDA weight |
| `j_cross` | 0.0497 | nA | Cross-pool inhibitory weight |
| `i_0` | 0.3255 | nA | Background tonic current |
| `sigma` | 0.02 | nA | Noise standard deviation |
| `dt` | 0.001 | s | Integration timestep (1 ms) |

### Parameter derivation (Wong & Wang 2006)

The parameters are derived from a mean-field reduction of the Wang 2002
spiking model (1600 neurons → 2 ODEs). The key mappings:

- J_N = 0.2609 nA: effective NMDA self-coupling (from 240 exc → 240 exc
  projection with g_NMDA connectivity)
- J_cross = 0.0497 nA: cross-inhibition via shared inhibitory pool
  (exc → inh → opp. exc, reduced to effective negative coupling)
- I_0 = 0.3255 nA: background drive from non-selective neurons + external

The ratio J_N / J_cross ≈ 5.25 controls the strength of competition.

---

## Analytical Properties

### Transfer function Φ(I)

- **Monotonic:** Φ is strictly increasing for I > b/a
- **Low rate regime:** For I < I_0, Φ ≈ 0 (below activation threshold)
- **Singularity handling:** At aI - b = 0, L'Hôpital gives Φ = 1/d ≈ 6.5 Hz
- **Linear regime:** For moderate I, Φ ≈ (aI - b) (rate proportional to current)
- **High rate saturation:** Φ → aI - b for large I (denominator → 1)

### Attractor landscape

The system has three key fixed points (for equal stimuli):
1. **Spontaneous state:** s1 ≈ s2 ≈ 0.1 (both pools at low activity)
2. **Pool 1 wins:** s1 ≈ 0.7, s2 ≈ 0.1 (decision for option 1)
3. **Pool 2 wins:** s1 ≈ 0.1, s2 ≈ 0.7 (decision for option 2)

The spontaneous state is unstable — any asymmetry in stimuli drives the
system toward one of the two "winner" attractors via positive feedback
(NMDA recurrence) and negative feedback (cross-inhibition).

### Decision mechanism

1. Both pools start at s1 = s2 = 0.1 (spontaneous state)
2. Asymmetric stimuli (stim1 ≠ stim2) create a bias
3. NMDA recurrence amplifies the bias: the pool with more input increases
   its firing rate → more NMDA current → higher rate (positive feedback)
4. Cross-inhibition suppresses the losing pool: higher s_winner →
   J_cross × s_winner subtracted from loser's current
5. The system settles into a winner-take-all attractor

### Reaction time

The time to reach a decision (escape from the spontaneous state) depends on:
- Stimulus difference (stim1 − stim2): larger → faster
- Noise amplitude σ: larger → faster but less accurate
- This speed-accuracy tradeoff matches psychophysical data (Roitman &
  Shadlen 2002)

### Bifurcation structure

The system exhibits a pitchfork bifurcation as a function of the
recurrent coupling strength $J_N$:

- **Subcritical ($J_N < J_c$):** Single stable fixed point (spontaneous
  state). No winner-take-all dynamics — the system cannot make decisions.
- **Supercritical ($J_N > J_c$):** Two stable attractors emerge via
  symmetry-breaking. The spontaneous state becomes a saddle point.
  The critical coupling $J_c$ depends on the transfer function slope
  at the spontaneous fixed point.

For the default parameters ($J_N = 0.2609$), the system is in the
bistable regime — both decision attractors coexist with the unstable
spontaneous state.

### Energy landscape interpretation

The dynamics can be interpreted as gradient descent on a double-well
potential:

$$U(s_1, s_2) = -\int \left[\frac{ds_1}{dt}, \frac{ds_2}{dt}\right] \cdot d[s_1, s_2]$$

- The two wells correspond to the two decision attractors
- The barrier height between wells determines the noise-resistance
  of committed decisions
- Stimulus asymmetry tilts the potential, making one well deeper
- σ controls the effective temperature of the noise-driven search

### NMDA time constant

τ_s = 0.1 s (100 ms) — much slower than AMPA (~5 ms). This slow dynamics
is critical for the attractor mechanism: it provides temporal integration
over the decision-relevant timescale (~500 ms).

### s bounded in [0, 1]

Both s1 and s2 are clipped to [0, 1] after each step. The NMDA gating
variable represents a fraction of open channels — physically bounded.

---

## Behaviour

### Decision-making with asymmetric input

With stim1 > stim2:
- s1 increases, s2 decreases
- After ~500 ms, s1 ≫ s2 → "decision for option 1"
- The decision is irreversible (attractor basin)

### Equal stimuli: noise-driven decision

With stim1 = stim2, noise (σ=0.02) breaks the symmetry:
- One pool randomly wins
- The outcome is probabilistic (~50/50 for equal stimuli)
- This models the random-dot motion discrimination task

### Noise amplitude effects

- σ = 0: deterministic, no decision possible with equal stimuli (stays
  at spontaneous state indefinitely)
- σ = 0.02: moderate noise, decisions in ~500 ms
- σ = 0.1: high noise, faster but less accurate decisions

### Stochastic dynamics

Two runs with the same parameters produce different outcomes because of
`np.random.randn()` noise. This is fundamental — the stochastic decision
dynamics are the model's core feature.

---

## Pipeline Compatibility

### Returns tuple, not int

**Critical limitation:** `step()` returns `(r1, r2)` — a tuple of two
firing rates. The SC-NeuroCore Network pipeline expects `step() → int`.

**Recommended use:** Standalone decision-making simulation. Track s1 and s2
over time, define a decision threshold (e.g., s > 0.6), and measure
reaction time and accuracy.

### Population compatible

`Population(WongWangUnit, n=10, label="ww")` works for construction.
Network simulation will produce incorrect results due to tuple return.

---

## Comparison with Related Models

| Property | Wong-Wang (reduced) | Wang 2002 (spiking) | Usher-McClelland |
|----------|--------------------|--------------------|------------------|
| Variables | 2 (s1, s2) | ~1600 neurons | 2 (x1, x2) |
| Type | Mean-field ODE | Spiking network | Rate ODE |
| NMDA | Explicit τ_s | Explicit channels | Implicit |
| Noise | Additive Gaussian | Synaptic noise | Diffusion |
| Speed | ~500K steps/s | ~10 neuron-steps/s | ~500K steps/s |
| Pipeline | Tuple return (limited) | Full spiking | Tuple return |

The Wong-Wang model achieves the same attractor dynamics as the 1600-neuron
Wang 2002 model with just 2 ODEs — a factor of 800× reduction in
computational cost.

### Relationship to drift-diffusion models

In the limit of weak coupling and small stimulus differences, the
Wong-Wang 2-variable dynamics can be projected onto a 1D
drift-diffusion model (DDM) along the unstable manifold:

$$dx = \mu \, dt + \sigma_{eff} \, dW$$

where $x = s_1 - s_2$ is the "decision variable", $\mu$ is proportional
to the stimulus difference, and $\sigma_{eff}$ depends on both
internal noise ($\sigma$) and the linearised dynamics. This connection
to the DDM explains why the Wong-Wang model reproduces the same
reaction time distributions as the DDM — both describe a noisy
accumulation process to a bound — while providing a biophysically
grounded account of the underlying neural circuit.

---

## Numerical Considerations

- **dt = 0.001 s (1 ms):** Typical for mean-field decision models. The
  slow NMDA dynamics (τ_s = 100 ms) are well-resolved at this timestep.
- **Integrator:** fixed-step RK4 over the coupled two-state ODE. The step
  uses the already sampled stochastic drive as piecewise-constant input for
  the RK4 stages, then commits only a finite clipped candidate.
- **Clipping:** s1, s2 clipped to [0, 1] after each finite candidate step.
- **Singularity in Φ:** The transfer function has a removable singularity
  at aI − b = 0. Handled with |x| < 1e-6 guard → returns 1/d ≈ 6.5 Hz.
- **Global RNG:** Uses `np.random.randn()` — not per-instance reproducible.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/wong_wang.py`.
- **Two state variables:** s1, s2 (NMDA gating fractions).
- **Dataclass:** Uses `@dataclass` for parameter storage.
- **Two-argument step:** `step(stim1, stim2)` — different from other models
  which take a single current argument.
- **Rust wiring:** Not in the Rust NeuronVariant enum (tuple return,
  two-argument step, global RNG).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~200K steps/s | Not applicable |
| Network | Limited (tuple return) | — |

Fast model — 2 Euler updates, 2 Φ evaluations (2 exp() calls), 2 randn()
calls per step. No sub-stepping.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, tuple return, both s evolve, finite 100k, reset |
| Transfer function | 4 | Φ monotonic, singularity protection, low/high rate regimes |
| Decision dynamics | 5 | asymmetric input → winner, equal input noise-driven, s bounded, σ=0 no decision, reaction time |
| Parameters | 3 | dt stability, σ sweep, j_n/j_cross ratio |
| Pipeline | 2 | Population creates, tuple return documented |
| **Total** | **19** | |

See `tests/test_model_wong_wang.py`. No bugs found.

---

## Findings

1. **Asymmetric input drives decisions:** stim1 > stim2 consistently leads
   to s1 > s2 after sufficient time.

2. **Equal stimuli produce 50/50 decisions:** With noise (σ=0.02), both
   pools have equal probability of winning.

3. **Transfer function singularity handled:** Φ(I) at aI−b=0 returns
   1/d ≈ 6.5 Hz correctly.

4. **s bounded by clipping:** After 100,000 steps, s1 and s2 remain in
   [0, 1] despite Euler integration.

5. **σ=0 prevents spontaneous decisions:** Without noise, equal stimuli
   leave the system at the symmetric spontaneous state indefinitely.

6. **NMDA recurrence amplifies:** J_N × s provides the positive feedback
   that drives the winner-take-all dynamics.

7. **Cross-inhibition suppresses:** J_cross × s_opponent provides the
   negative feedback that shuts down the losing pool.

8. **Stochastic outcomes:** Two runs produce different decisions — this is
   the model's core feature, not a bug.

9. **J_N/J_cross ≈ 5.25:** The self/cross coupling ratio determines the
   strength of competition. Higher ratio → sharper decisions.

10. **Pipeline-limited:** Tuple return and two-argument step prevent
    standard Network integration. The model is designed for standalone
    decision-making simulations.

---

## Applications in Neuroscience

### Random dot motion discrimination

The canonical application: two pools represent "left" and "right" motion
direction. Stimulus coherence maps to stim1 − stim2. At 0% coherence
(equal stimuli), accuracy is 50%. At 51.2% coherence, accuracy reaches
~96% — matching monkey psychophysics (Roitman & Shadlen 2002).

### Speed-accuracy tradeoff

By varying the decision threshold (e.g., s > 0.6 for commitment):
- Lower threshold → faster, less accurate decisions
- Higher threshold → slower, more accurate decisions
This tradeoff is a fundamental prediction of the model.

### Dynamic causal modelling (DCM)

The Wong-Wang model is used in SPM12 (Donders/Wellcome) for spectral DCM
of resting-state fMRI. The NMDA dynamics map to the BOLD timescale via
a Balloon-Windkessel hemodynamic model.


---

## Theoretical Context

### Historical background

Wong & Wang (2006) published "A recurrent network mechanism of time
integration in perceptual decisions" in the Journal of Neuroscience.
The model reduces the spiking network model of Wang (2002) — which
had 2000 leaky integrate-and-fire neurons with AMPA, NMDA, and GABA
synapses — to a tractable 2-variable mean-field description.

The reduction is based on the observation that NMDA receptor kinetics
($\tau_{NMDA} \approx 100$ ms) dominate the recurrent dynamics. The
fast AMPA and GABA components can be adiabatically eliminated, leaving
a 2D system describing the NMDA-gated synaptic activity of two
competing neural populations.

### Decision-making as attractor dynamics

The model frames perceptual decision-making as competition between
two attractor states in a recurrent neural network:

1. **Stimulus onset:** Two populations receive noisy evidence for
   options A and B
2. **Competition:** Recurrent excitation + mutual inhibition creates
   winner-take-all dynamics
3. **Decision:** One population reaches a threshold firing rate
4. **Commitment:** The winning attractor is stable — the decision
   persists even after stimulus removal

This attractor framework explains both accuracy and reaction time
as functions of stimulus coherence (motion discrimination tasks).

### Speed-accuracy tradeoff

The model predicts the well-known speed-accuracy tradeoff:
- **Low threshold:** Fast decisions, more errors (impulsive)
- **High threshold:** Slow decisions, fewer errors (cautious)
- **Optimal threshold:** Reward-rate maximising point depends on
  stimulus statistics and inter-trial interval

### Resting-state fMRI and DCM

The Wong-Wang model is the default neural mass model in SPM12's
spectral Dynamic Causal Modelling (DCM) for resting-state fMRI.
The NMDA synaptic dynamics ($\tau \approx 100$ ms) map naturally
to the BOLD signal timescale via the Balloon-Windkessel
hemodynamic model. Deco et al. (2013) used large-scale networks
of Wong-Wang units coupled by the human structural connectome to
reproduce resting-state functional connectivity patterns.

### The Virtual Brain (TVB)

The Wong-Wang model is one of the primary neural mass models in
The Virtual Brain platform — used for patient-specific brain
simulation, epilepsy surgery planning, and connectome-based
modelling. SC-NeuroCore's implementation is compatible with TVB
parameter conventions.

---

## Usage Examples

### Example 1: Two-choice decision circuit

```python
from sc_neurocore.neurons.models.wong_wang import WongWangUnit

# Two competing populations
pop_a = WongWangUnit()
pop_b = WongWangUnit()

# Coherent stimulus favours A
for t in range(10000):
    pop_a.step(0.06)   # stronger evidence for A
    pop_b.step(0.02)   # weaker evidence for B

print(f"Pop A activity: {pop_a.s:.3f}")
print(f"Pop B activity: {pop_b.s:.3f}")
print(f"Decision: {'A' if pop_a.s > pop_b.s else 'B'}")
```

### Example 2: Reaction time as function of coherence

```python
from sc_neurocore.neurons.models.wong_wang import WongWangUnit

for coherence in [0.0, 0.05, 0.1, 0.2, 0.5]:
    a = WongWangUnit()
    b = WongWangUnit()
    threshold = 0.7
    rt = None
    for t in range(50000):
        a.step(0.04 + coherence * 0.02)
        b.step(0.04 - coherence * 0.02)
        if a.s > threshold or b.s > threshold:
            rt = t
            break
    print(f"c={coherence:.2f}: RT={'timeout' if rt is None else f'{rt} steps'}")
```

### Example 3: Resting-state network

```python
from sc_neurocore.network import Network, Population
from sc_neurocore.neurons.models.wong_wang import WongWangUnit
from sc_neurocore.monitors import SpikeMonitor
from sc_neurocore.analysis import spike_count

regions = Population(WongWangUnit, n=68)  # Desikan-Killiany atlas
net = Network()
net.add_population("cortex", regions)

mon = SpikeMonitor()
net.add_monitor("bold", mon, source="cortex")
net.run(duration=10.0)
print(f"Total events: {spike_count(mon)}")
```

---

## Multi-language acceleration chain

### Kernel sources

| Backend | Source file | Binding |
|---------|-------------|---------|
| Python primary | `src/sc_neurocore/neurons/models/wong_wang.py` | — (reference) |
| Rust (PyO3)    | `engine/src/wong_wang.rs`                      | `sc_neurocore_engine.py_wong_wang_simulate` |
| Julia (juliacall) | `src/sc_neurocore/accel/julia/neurons/wong_wang.jl` | `sc_neurocore.accel.julia.neurons.simulate_wong_wang` |
| Go (cgo)       | `src/sc_neurocore/accel/go/wong_wang/wong_wang.go` | `sc_neurocore.accel.go.wong_wang.simulate_wong_wang` |
| Mojo (FFI)     | `src/sc_neurocore/accel/mojo/wong_wang/wong_wang.mojo` | `sc_neurocore.accel.mojo.wong_wang.simulate_wong_wang` |

The Python primary draws two `np.random.randn()` samples per step;
every accelerated backend takes the same `2 * n_steps` draws as a
pre-allocated `xi` buffer, so trajectories compare bit-exact for
matching seeds.

The Julia, Go, and Mojo facades validate `stim1`, `stim2`, and `xi` as
one-dimensional `float64` traces before backend dispatch. `stim1` and `stim2`
must have equal length, and `xi` must contain exactly two noise draws per step.
This keeps every accelerated path on the same explicit time-series contract
instead of relying on implicit NumPy flattening at the Python boundary.

### Multi-backend performance

Measured on local i5-11600K, `N = 100 000` steps, `benchmarks/
bench_wong_wang.py`. Numbers trace back to
`benchmarks/results/bench_wong_wang.json` committed alongside the
implementation.

| Backend | Steps/s | Wall (ms) | Speedup vs Python | Parity vs Rust |
|---------|--------:|----------:|------------------:|---------------:|
| Python primary |    234 647 |  426.17 |   1.00× | — |
| Rust PyO3      | 31 445 275 |    3.18 | 134.0× | reference |
| Julia (warm)   | 28 121 018 |    3.56 | 119.8× | Δ = 0 (bit-exact) |
| Go cgo         | 23 332 251 |    4.29 |  99.4× | Δ ≈ 7e-18 (denormal round-off) |
| Mojo           | 28 979 233 |    3.45 | 123.5× | Δ ≈ 6e-14 (libm vs f64::exp) |

Julia cold start incurs a one-time ~5-10 s JIT warm-up that is
excluded from the table (the bench harness performs a 1 000-step
warm-up before the timed run). Mojo's larger parity delta comes
from the system libm's `exp()` vs Rust std's `f64::exp`; both are
IEEE-compliant but differ in the last ulp on some inputs.

### Backends

| Backend | Status | Rationale |
|---------|--------|-----------|
| Rust PyO3 | **USED** | default `auto` path; narrowly fastest and zero parity drift |
| Julia     | USED   | warm path ties Rust; preferred when juliacall is already hot |
| Go cgo    | USED   | bit-exact with Rust, slightly higher FFI overhead |
| Mojo FFI  | USED   | competitive with Rust on warm calls; toleratable drift |

### Tests

| Backend | File | Tests | What is verified |
|---------|------|------:|------------------|
| Rust    | `tests/test_wong_wang_parity.py` (+ `engine/src/wong_wang.rs::tests`) | 14 | φ singularity, monotonicity, zero-noise symmetry, biased-stim winner, xi-length panic; Python↔Rust bit-exact across 5 seeds; final-state match; input-validation |
| Julia   | `tests/test_wong_wang_julia_parity.py` | 6 | Python↔Julia bit-exact (quiescent, biased, 5-seed sweep); Rust↔Julia cross-parity under shared xi; input-validation |
| Go      | `tests/test_wong_wang_go_parity.py`   | 5 | Python↔Go bit-exact; Rust↔Go cross-parity; input-validation |
| Mojo    | `tests/test_wong_wang_mojo_parity.py` | 4 | Python↔Mojo within libm ulp drift; Rust↔Mojo within libm ulp drift; input-validation |
| Facade contracts | `tests/test_wong_wang_accel_dispatch_contracts.py` | 21 | Julia, Go, and Mojo non-1D input rejection; Go/Mojo loader failure sentinels, unavailable-library errors, C return-code propagation, stimulus/noise length validation, and one-dimensional return buffers |

---

## Citations

1. Wong K-F, Wang X-J (2006). A recurrent network mechanism of time
   integration in perceptual decisions. *J Neurosci* 26(4):1314–1328.
   DOI: [10.1523/JNEUROSCI.3733-05.2006](https://doi.org/10.1523/JNEUROSCI.3733-05.2006)

2. Wang X-J (2002). Probabilistic decision making by slow reverberation
   in cortical circuits. *Neuron* 36(5):955–968.
   DOI: [10.1016/S0896-6273(02)01092-9](https://doi.org/10.1016/S0896-6273(02)01092-9)

3. Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M
   (2013). Resting-state functional connectivity emerges from structurally
   and dynamically shaped slow linear fluctuations. *J Neurosci*
   33(27):11239–11252.
   DOI: [10.1523/JNEUROSCI.1091-13.2013](https://doi.org/10.1523/JNEUROSCI.1091-13.2013)

4. Gold JI, Shadlen MN (2007). The neural basis of decision making.
   *Annu Rev Neurosci* 30:535–574.
   DOI: [10.1146/annurev.neuro.29.051605.113038](https://doi.org/10.1146/annurev.neuro.29.051605.113038)

5. Sanz Leon P, Knock SA, Woodman MM, et al. (2013). The Virtual Brain:
   a simulator of primate brain network dynamics. *Front Neuroinform*
   7:10. DOI: [10.3389/fninf.2013.00010](https://doi.org/10.3389/fninf.2013.00010)

6. Roitman JD, Shadlen MN (2002). Response of neurons in the lateral
   intraparietal area during a combined visual discrimination reaction
   time task. *J Neurosci* 22(21):9475–9489.
   DOI: [10.1523/JNEUROSCI.22-21-09475.2002](https://doi.org/10.1523/JNEUROSCI.22-21-09475.2002)

---

## Limitations and Extensions

### Known limitations

- **Two alternatives only:** The standard model supports binary
  decisions. Multi-alternative extensions (Albantakis & Deco 2009)
  require n pools with generalised cross-inhibition.
- **No urgency signal:** Reaction times for long trials are
  overestimated. The urgency-gating model (Cisek et al. 2009) adds
  a time-dependent gain to the transfer function.
- **Fixed connectivity:** $J_N$ and $J_{cross}$ are static. Synaptic
  plasticity during the decision process (e.g., reward-modulated
  Hebbian learning) is not modelled.
- **Gaussian noise assumption:** Biological noise is Poisson-like
  (shot noise from spikes). The Gaussian approximation holds for
  large populations but breaks for small pool sizes.
- **No adaptation:** The model lacks firing-rate adaptation or
  synaptic depression, which would enable sequential sampling
  (evidence reset between trials).
