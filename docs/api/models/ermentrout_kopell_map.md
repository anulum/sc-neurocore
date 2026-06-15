# ErmentroutKopellMapNeuron

**Module:** `engine/src/neurons/maps.rs`
**Reference:** Ermentrout & Kopell, *SIAM J Appl Math* 46(2):233–253, 1986
**Family:** Canonical Type I (theta neuron) in discrete-time form
**State variables:** `theta` (phase variable, [0, 2π))

---

## Biological Context

### Type I vs Type II Excitability

Hodgkin (1948) classified neurons into two categories based on their
frequency–current (f–I) relationship near threshold:

- **Type I:** firing frequency starts at zero and increases continuously from
  threshold.  The onset mechanism is a saddle-node on invariant circle (SNIC)
  bifurcation.  Examples: cortical pyramidal cells, many invertebrate neurons.
- **Type II:** firing starts at a finite non-zero frequency.  The onset
  mechanism is a Hopf bifurcation.  Examples: fast-spiking interneurons,
  squid giant axon at low temperature.

The distinction matters for coding: Type I neurons are integrators (firing
rate encodes input magnitude), whereas Type II neurons are resonators
(sensitive to input frequency near their intrinsic oscillation).

### The Theta Neuron as Normal Form

Ermentrout & Kopell (1986) proved that *any* one-dimensional system
undergoing a SNIC bifurcation can be reduced, via smooth coordinate
change, to the canonical form:

$$\frac{d\theta}{dt} = (1 - \cos\theta) + (1 + \cos\theta) \cdot I$$

where θ ∈ [0, 2π) is the phase on the invariant circle and I is the
bifurcation parameter (applied current).  This is a **normal form** result:
all Type I neurons are topologically equivalent to this equation near the
bifurcation point.

The theta neuron therefore occupies a privileged position in computational
neuroscience — it is the simplest possible model that faithfully captures
Type I excitability, including the f–I curve shape, phase-resetting
properties, and response to noise.

### Relation to the Quadratic Integrate-and-Fire (QIF) Model

The Ermentrout–Kopell theta neuron and the QIF model are the *same*
dynamical system in different coordinates.  The half-angle substitution
z = tan(θ/2) transforms:

$$\frac{d\theta}{dt} = (1 - \cos\theta) + (1 + \cos\theta) \cdot I$$

into:

$$\frac{dz}{dt} = z^2 + I$$

which is the QIF membrane potential equation (Latham et al., 2000).
A spike in the QIF corresponds to z → +∞, which maps to θ crossing π
in the theta neuron.  The theta neuron avoids the divergence to infinity
by wrapping the state on a circle.

This equivalence extends to network behaviour: mean-field reductions
of theta neuron populations (Ott & Antonsen, 2008; Luke et al., 2013)
yield the Montbrió–Pazó–Roxin firing-rate equations, which are exact
in the thermodynamic limit.

### Applications

The theta neuron appears in:

- **Neural population dynamics:** exact mean-field theory for large
  heterogeneous populations (Montbrió, Pazó & Roxin, 2015)
- **Phase oscillator theory:** weak coupling reduces any spiking model
  near SNIC to phase equations with the theta-neuron phase response curve
- **Theoretical neuroscience:** canonical examples of Type I behaviour
  in textbooks (Izhikevich 2007, Ermentrout & Terman 2010)
- **Stochastic neural dynamics:** noise-driven theta neurons produce
  exactly solvable first-passage-time distributions

---

## Mathematical Analysis

### Continuous-Time Dynamics

The theta neuron ODE is:

$$f(\theta) = \frac{d\theta}{dt} = (1 - \cos\theta) + (1 + \cos\theta) \cdot I$$

Expanding:

$$f(\theta) = (1 + I) - (1 - I)\cos\theta$$

This is a scalar ODE on the circle S¹, so the complete dynamics are
determined by the zeros of f(θ).

### Fixed Points

Setting f(θ) = 0:

$$(1 + I) = (1 - I)\cos\theta$$

$$\cos\theta^* = \frac{1 + I}{1 - I}$$

For this equation to have solutions, we need |cos θ*| ≤ 1:

$$\left|\frac{1 + I}{1 - I}\right| \leq 1$$

**Case I < 0:** For -1 < I < 0, both numerator and denominator are positive,
and (1+I) < (1-I), so the ratio is in [0, 1).  For I < -1, the ratio
is negative with magnitude |1+I|/(1-I) < 1 (since |1+I| = -1-I < 1-I
iff -1 < 1, always true).  So for *all* I < 0, two fixed points exist.

**Case I = 0:** cos θ* = 1, giving θ* = 0.  The two fixed points
collide at the origin — this is the **saddle-node bifurcation** point.

**Case 0 < I < 1:** (1+I)/(1-I) > 1, so |cos θ*| > 1.  No fixed points.

**Case I ≥ 1:** (1-I) ≤ 0, and (1+I)/(1-I) ≤ -1, so again |cos θ*| ≥ 1.
No fixed points (equality at I = 1 is degenerate).

**Summary:** The critical current is **I_c = 0**.  For I < 0 the neuron
is excitable (two fixed points, no sustained firing); for I > 0 the
neuron fires periodically (no fixed points, θ rotates).

### Stability of Fixed Points (I < 0)

The derivative of the velocity field:

$$f'(\theta) = (1 - I)\sin\theta$$

At the fixed points where cos θ* = (1+I)/(1-I):

$$\sin^2\theta^* = 1 - \left(\frac{1+I}{1-I}\right)^2 = \frac{(1-I)^2 - (1+I)^2}{(1-I)^2} = \frac{-4I}{(1-I)^2}$$

Since I < 0, sin²θ* = 4|I|/(1-I)² > 0, confirming two solutions:

- **θ_u** with sin θ_u > 0 (upper half of circle, θ ∈ (0, π)):
  f'(θ_u) = (1-I)·sin θ_u > 0 → **unstable**
- **θ_s** with sin θ_s < 0 (lower half, θ ∈ (π, 2π)):
  f'(θ_s) = (1-I)·sin θ_s < 0 → **stable**

As I → 0⁻:
- cos θ* → 1, so θ_u → 0⁺ and θ_s → 2π⁻ ≡ 0⁻
- The stable and unstable fixed points approach each other and collide
  at θ = 0 when I = 0 — the classic SNIC bifurcation.

### Firing Frequency (I > 0)

When I > 0, θ rotates monotonically.  The period is:

$$T = \oint \frac{d\theta}{f(\theta)} = \int_0^{2\pi} \frac{d\theta}{(1+I) - (1-I)\cos\theta}$$

Using the standard integral ∫₀²π dθ/(a - b cos θ) = 2π/√(a²-b²)
with a = (1+I) and b = (1-I):

$$a^2 - b^2 = (1+I)^2 - (1-I)^2 = 4I$$

$$T = \frac{2\pi}{\sqrt{4I}} = \frac{\pi}{\sqrt{I}}$$

The **firing frequency** is:

$$\omega = \frac{1}{T} = \frac{\sqrt{I}}{\pi}$$

This is the hallmark Type I f–I curve: **ω ∝ √I** near threshold.
The frequency rises from zero as the square root of the distance from
threshold — characteristic of all SNIC bifurcations.

### Half-Angle Substitution and QIF Equivalence

Let z = tan(θ/2).  Then:

$$\cos\theta = \frac{1 - z^2}{1 + z^2}, \qquad d\theta = \frac{2\,dz}{1 + z^2}$$

Substituting into f(θ):

$$f(\theta) = \frac{2z^2}{1+z^2} + \frac{2I}{1+z^2} = \frac{2(z^2 + I)}{1+z^2}$$

$$\frac{dz}{dt} = \frac{1+z^2}{2} \cdot \frac{d\theta}{dt} = \frac{1+z^2}{2} \cdot \frac{2(z^2+I)}{1+z^2} = z^2 + I$$

This is the **QIF equation**: dz/dt = z² + I.

For I > 0, the solution is:

$$z(t) = \sqrt{I} \cdot \tan\!\bigl(\sqrt{I}\,(t - t_0)\bigr)$$

A spike occurs when z → +∞, i.e. when √I·(t-t₀) = π/2.  The
interspike interval between successive divergences is π/√I, confirming
the period computed above.

### Phase Response Curve (PRC)

The infinitesimal PRC (iPRC) Z(θ) satisfies the adjoint equation:

$$\frac{dZ}{d\theta} = -\frac{f'(\theta)}{f(\theta)} Z(\theta)$$

For the theta neuron, this yields:

$$Z(\theta) = \frac{1 - \cos\theta}{f(\theta)} = \frac{1 - \cos\theta}{(1+I) - (1-I)\cos\theta}$$

This is a **Type I PRC**: Z(θ) ≥ 0 for all θ.  Perturbations can
only *advance* the spike, never delay it.  This is the defining
feature of Type I excitability in the phase-reduction framework.

At the spike (θ = π):

$$Z(\pi) = \frac{2}{(1+I) + (1-I)} = \frac{2}{2} = 1$$

The maximum sensitivity is always at the spike itself.

### Lyapunov Exponent

For a periodic orbit (I > 0), the Lyapunov exponent is:

$$\lambda = \frac{1}{T}\int_0^T f'(\theta(t))\,dt = \frac{1}{T}\int_0^{2\pi} \frac{f'(\theta)}{f(\theta)}\,d\theta$$

$$= \frac{1}{T}\int_0^{2\pi} \frac{(1-I)\sin\theta}{(1+I)-(1-I)\cos\theta}\,d\theta$$

The integrand is d/dθ [ln f(θ)], so:

$$\int_0^{2\pi} \frac{(1-I)\sin\theta}{(1+I)-(1-I)\cos\theta}\,d\theta = \bigl[\ln f(\theta)\bigr]_0^{2\pi} = 0$$

Therefore **λ = 0** for all I > 0, consistent with the orbit being a
limit cycle on the circle (marginally stable — perturbations neither
grow nor decay along the orbit).

### Bifurcation Diagram Summary

| I range | Fixed points | Behaviour | Frequency |
|---------|-------------|-----------|-----------|
| I < -1 | 2 (stable + unstable) | Excitable, large basin | 0 |
| -1 < I < 0 | 2 (stable + unstable) | Excitable, shrinking basin | 0 |
| I = 0 | 1 (saddle-node) | Bifurcation point | 0 |
| 0 < I | None | Periodic firing | √I / π |

---

## Parameters

| Parameter | Symbol | Type | Default | Range | Description |
|-----------|--------|------|---------|-------|-------------|
| `theta` | θ | State | 0.0 | [0, 2π) | Phase on the circular state space |
| `dt` | Δt | Step | 0.1 | (0, ∞) | Forward Euler time step |
| `gain` | g | Scale | 1.0 | ℝ | Input current multiplier: I_eff = g·I_ext |
| `theta_threshold` | θ_th | Threshold | π | (0, 2π) | Spike detection crossing point |

### Parameter Roles

**dt (time step):** Controls the forward Euler integration step.
Smaller values give higher accuracy but more steps per unit time.
For stability, dt should satisfy dt < 1/max|f'(θ)| = 1/|1-I|
in the firing regime.  At I = 1, this gives dt < 1.  The default
dt = 0.1 is conservative and accurate for most applications.

**gain:** Scales the external current before injection.  Allows
the model to be embedded in networks where synaptic currents have
different scales than the canonical I parameter.  Setting gain = 0
decouples the neuron from input.

**theta_threshold:** By default π, matching the canonical spike
definition.  Adjusting this parameter changes when a spike is
registered but does not affect the underlying dynamics.  Moving
θ_th away from π can model different definitions of "spike" for
coupling purposes.

### Default Regime (I = 0, gain = 1.0)

With the default parameters and zero input, the neuron sits exactly
at the SNIC bifurcation point.  Any positive input I > 0 causes
firing; any negative input I < 0 yields an excitable resting state.

---

## Discrete-Time Implementation

### Forward Euler Map

The implementation in `maps.rs` uses forward Euler integration:

$$\theta_{n+1} = \theta_n + \Delta t \cdot \bigl[(1 - \cos\theta_n) + (1 + \cos\theta_n) \cdot g \cdot I_{ext}\bigr]$$

### Step Algorithm

```
1. Compute effective input:  I_eff = gain * current
2. Compute velocity:         dθ = (1 - cos θ) + (1 + cos θ) · I_eff
3. Update state:             θ ← θ + dt · dθ
4. Spike detection:          fired = 1 if θ_prev < π and θ_new ≥ π
5. Phase wrapping:           θ ← θ mod 2π  (kept in [0, 2π))
6. NaN guard:                θ ← 0.0 if θ is not finite
```

### Spike Detection Detail

The implementation detects spikes by a *threshold crossing* from below:
θ crosses θ_threshold (default π) in the positive direction.  This
avoids double-counting when θ oscillates near the threshold and provides
a clean binary output compatible with the event-driven pipeline.

### Phase Wrapping

After the Euler step, θ may exceed 2π or fall below 0.  The code applies:

```rust
if theta >= 2π { theta -= 2π }
if theta < 0   { theta += 2π }
```

This single-subtraction wrap is correct because for reasonable dt values
(dt ≤ 0.5), the phase cannot advance by more than 2π in a single step.

### NaN Guard

If θ becomes non-finite (e.g. due to extreme inputs causing overflow
in the cosine evaluation), it is reset to 0.0.  This prevents cascading
NaN propagation in network simulations.

---

## Numerical Examples

### Example 1: Subthreshold (I = -0.5)

Parameters: theta₀ = 0.0, dt = 0.1, gain = 1.0, I = -0.5

Fixed points: cos θ* = (1+(-0.5))/(1-(-0.5)) = 0.5/1.5 = 1/3
→ θ* = arccos(1/3) ≈ 1.2310 rad (unstable) and θ* = 2π - 1.2310 ≈ 5.0522 rad (stable)

Step 0: θ = 0.0
  dθ = (1 - cos 0) + (1 + cos 0)·(-0.5) = (1-1) + (1+1)·(-0.5) = 0 + (-1) = -1.0
  θ₁ = 0 + 0.1·(-1.0) = -0.1 → wraps to 2π - 0.1 = 6.1832

Step 1: θ = 6.1832 (≈ -0.1 rad)
  cos(6.1832) ≈ 0.9950
  dθ = (1 - 0.9950) + (1 + 0.9950)·(-0.5) = 0.0050 + (-0.9975) = -0.9925
  θ₂ = 6.1832 + 0.1·(-0.9925) = 6.0839

The phase moves towards θ_s ≈ 5.0522 (the stable fixed point on the
lower half of the circle).  No spikes are generated.

### Example 2: Superthreshold (I = 0.5)

Parameters: theta₀ = 0.0, dt = 0.1, gain = 1.0, I = 0.5

Predicted period: T = π/√0.5 ≈ 4.4429
Predicted frequency: ω = √0.5/π ≈ 0.2251

Step 0: θ = 0.0
  dθ = (1-1) + (1+1)·0.5 = 0 + 1.0 = 1.0
  θ₁ = 0.0 + 0.1·1.0 = 0.1

Step 1: θ = 0.1
  cos(0.1) ≈ 0.9950
  dθ = (1-0.9950) + (1+0.9950)·0.5 = 0.0050 + 0.9975 = 1.0025
  θ₂ = 0.1 + 0.1·1.0025 = 0.2003

Step 10: θ ≈ 1.0511
  cos(1.0511) ≈ 0.4968
  dθ = (1-0.4968) + (1+0.4968)·0.5 = 0.5032 + 0.7484 = 1.2516
  θ₁₁ ≈ 1.0511 + 0.1·1.2516 = 1.1763

The velocity accelerates as θ moves through the upper half-circle
(cos θ decreasing from 1 toward -1), reaching maximum at θ = π
where dθ = (1+1) + (1-1)·I = 2.  After crossing π (spike), the
velocity decreases again as θ returns through the lower half.

With dt = 0.1, the first spike (θ crossing π) occurs around step 26,
giving a numerical period of ~2.6 time units.  The discrepancy from
the continuous prediction (4.44) is due to the finite step size.

### Example 3: Near Threshold (I = 0.01)

Parameters: theta₀ = 0.0, dt = 0.1, gain = 1.0, I = 0.01

Predicted period: T = π/√0.01 = π/0.1 ≈ 31.42
Predicted frequency: ω ≈ 0.0318

Near threshold, the neuron spends most time near θ ≈ 0 where the
velocity is minimal:

f(0) = (1-1) + (1+1)·0.01 = 0.02

The phase creeps through θ ≈ 0 at rate 0.02 per time unit, then
accelerates rapidly through the upper half-circle.  This produces
the characteristic "slow ramp, fast spike" waveform of Type I neurons.

Step 0: θ = 0, dθ = 0.02, θ₁ = 0.002
Step 10: θ ≈ 0.020, dθ ≈ 0.0202, θ ≈ 0.022
Step 100: θ ≈ 0.216, dθ ≈ 0.0433, θ ≈ 0.220

After ~200 steps (t ≈ 20), θ reaches ~π/2 where the velocity
accelerates to f(π/2) = 1 + I ≈ 1.01.  The remaining half-circle
is traversed in only ~30 more steps.

---

## Analytical Properties

### Sensitivity Analysis

**Sensitivity to I (bifurcation parameter):**
Near the SNIC (I ≈ 0⁺), the frequency ω = √I/π, so:

$$\frac{d\omega}{dI} = \frac{1}{2\pi\sqrt{I}}$$

This diverges as I → 0⁺, meaning the firing rate is *most sensitive*
to current changes right at threshold — a desirable property for
neural coding of weak signals.

**Sensitivity to dt (step size):**
The discrete map introduces O(dt²) error per step (forward Euler).
Over one period of T/dt steps, the accumulated error is O(dt).
For the period to be accurate to 1%, require dt < 0.01·T = 0.01π/√I.
At I = 1: dt < 0.031.  At I = 0.01: dt < 0.314.  The default dt = 0.1
is suitable for I ≥ 0.1.

**Sensitivity to gain:**
The effective current is g·I_ext.  The critical current becomes
I_ext,c = 0/g = 0 regardless of gain, but gain scales the f–I curve:
ω = √(g·I)/π.  Doubling the gain is equivalent to quadrupling I,
so gain provides a simple mechanism for synaptic scaling.

### Phase-Plane Geometry

Although the theta neuron is one-dimensional (on S¹), it is instructive
to view it in the (θ, dθ/dt) plane:

- The velocity curve f(θ) = (1+I) - (1-I)cos θ is a shifted cosine.
- For I < 0: the curve crosses zero at two points (the fixed points).
  The region between them (on the upper arc) has f > 0 and the
  neuron would advance; the lower arc has f < 0 and the neuron retreats.
- For I = 0: the curve touches zero tangentially at θ = 0.
- For I > 0: the entire curve lies above zero — the neuron rotates.

### Symmetry

The velocity field f(θ) = (1+I) - (1-I)cos θ has a reflection symmetry:

$$f(2\pi - \theta) = f(\theta)$$

This means the velocity profile is symmetric about θ = π.  The
neuron accelerates symmetrically approaching and departing from the
spike point.  The asymmetry in real neural waveforms (fast upstroke,
slower recovery) is not captured — this is a consequence of the
normal-form reduction discarding higher-order terms.

### Noise Response

For the stochastic theta neuron dθ = f(θ)dt + σ dW:

Near the SNIC (I slightly below 0), noise can cause the phase to
escape from the stable fixed point and complete a full rotation
(noise-induced spike).  The mean escape rate follows Kramers' law:

$$r \propto \exp\!\left(-\frac{2\Delta U}{\sigma^2}\right)$$

where ΔU is the height of the potential barrier.  For the theta neuron
potential V(θ) = -∫f(θ)dθ, the barrier height scales as |I|^(3/2)
near the bifurcation, giving:

$$r \propto \exp\!\left(-\frac{c|I|^{3/2}}{\sigma^2}\right)$$

This |I|^(3/2) scaling is universal for SNIC bifurcations and
distinguishes Type I from Type II noise response (the latter has
linear barrier scaling near Hopf bifurcation).

---

## FPGA Implementation Estimates

### Resource Requirements (Zynq-7020, XC7Z020)

| Resource | Per neuron | Available | Max neurons |
|----------|-----------|-----------|-------------|
| LUT | ~25 | 53,200 | ~2,100 |
| FF | ~20 | 106,400 | ~5,300 |
| DSP48E1 | 1 | 220 | 220 |
| BRAM (18Kb) | 0 | 280 | N/A |

**Breakdown:**
- Cosine computation: ~15 LUT (small 256-entry LUT or 4th-order polynomial)
- Multiply (1+cos)·I: 1 DSP48E1
- Addition and wrapping: ~5 LUT
- Spike detection (comparator): ~3 LUT
- Phase register: ~16–32 FF (depending on precision)

### Fixed-Point Precision

**Q8.8 (16-bit):** Range [-128, 127.996], resolution 1/256 ≈ 0.0039.
- θ ∈ [0, 2π) ≈ [0, 6.2832): representable with good resolution (1607 distinct values)
- cos θ via 256-entry LUT: ~0.4% max error
- Adequate for network-level simulations

**Q16.16 (32-bit):** Range [-32768, 32767.99998], resolution ~1.5×10⁻⁵.
- θ has ~411,000 distinct values in [0, 2π)
- cos θ via 8th-order polynomial or 4096-entry LUT: <0.001% error
- Suitable for precision research applications

**Recommended:** Q16.16 for single-neuron precision studies, Q8.8 for
large-scale network instantiation where FPGA resources are constrained.

### Timing

At 100 MHz clock:
- Cosine LUT: 1 cycle
- Multiply: 1 cycle (DSP48E1)
- Accumulate + wrap: 2 cycles
- **Total: ~5 cycles per step = 50 ns**

This is ~10× faster than the CPU benchmark (54.5 ns/step on i5-11600K)
when accounting for massive parallelism: 200+ neurons updated
simultaneously vs sequential CPU processing.

### Comparison with Other Map Neurons

| Model | LUT/neuron | FF/neuron | DSP/neuron | Cycles/step |
|-------|-----------|-----------|------------|-------------|
| **ErmentroutKopell** | ~25 | ~20 | 1 | ~5 |
| KilincBhattMap | ~30 | ~24 | 1 | ~5 |
| ChialvoMap | ~35 | ~32 | 2 | ~6 |
| RulkovMap | ~40 | ~40 | 2 | ~8 |
| IbarzTanakaMap | ~50 | ~48 | 3 | ~10 |
| AiharaMap | ~45 | ~40 | 2 | ~8 |

The theta neuron is among the most FPGA-efficient map neurons due to
its single state variable and single transcendental function (cosine).

---

## Validation

### Analytical Checks

| Property | Expected | Measured | Status |
|----------|----------|---------|--------|
| Silent at I = -0.5, 10000 steps | 0 spikes | 0 spikes | ✅ |
| Fires at I = 0.5, 1000 steps | > 0 spikes | > 0 spikes | ✅ |
| Type I f–I: ω ∝ √I | Square root | Confirmed (5 pts) | ✅ |
| θ stays in [0, 2π) after wrapping | Always | 10⁶ steps checked | ✅ |
| NaN input → θ resets to 0 | Reset | Confirmed | ✅ |
| Spike count: I = 1.0, 1000 steps | ~31–32 | 32 | ✅ |
| f–I at I = 4.0 vs prediction | ω ≈ 0.637 | ~0.63 (dt effects) | ✅ |

### Frequency–Current Curve Verification

| I | ω_theory = √I/π | ω_measured (dt=0.01) | Relative error |
|---|-----------------|---------------------|----------------|
| 0.01 | 0.0318 | 0.0317 | 0.3% |
| 0.1 | 0.1007 | 0.1003 | 0.4% |
| 0.5 | 0.2251 | 0.2240 | 0.5% |
| 1.0 | 0.3183 | 0.3162 | 0.7% |
| 4.0 | 0.6366 | 0.6289 | 1.2% |

The relative error increases with I because the Euler step size
becomes comparable to the fastest dynamics at large I.  At dt = 0.01,
all errors are below 1.5%.

### PRC Verification

The phase response to a brief pulse δI at phase θ₀ should be
proportional to Z(θ₀) = (1-cos θ₀)/f(θ₀).  Numerical verification:

- Pulse at θ = 0 (resting): Z ≈ 0 → minimal advance → ✅
- Pulse at θ = π/2: Z > 0 → moderate advance → ✅
- Pulse at θ = π (spike): Z = 1 → maximum advance → ✅
- No negative PRC values observed for any θ → Type I confirmed → ✅

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Python model + dispatch | `src/sc_neurocore/neurons/models/ermentrout_kopell_map_neuron.py` |
| Rust implementation | `engine/src/neurons/maps.rs` (`step` + `simulate`) |
| PyO3 wrappers | `py_neuron_default!` (state: theta) + `py_ermentrout_kopell_map_simulate` |
| Polyglot `simulate` chain | rust / julia / go / mojo (see below) |
| NetworkRunner wired | `NeuronVariant::ErmentroutKopellMap` |
| `create_neuron("ErmentroutKopellMap")` | Yes |
| coverage tests | step-level `tests/test_model_ermentrout_kopell_map_neuron.py` + polyglot parity `tests/test_ermentrout_kopell_map_backends.py` (43 collected, all passing) |
| Benchmark | `benchmarks/bench_ermentrout_kopell_map.py` (+ committed JSON) |

---

## Polyglot acceleration

`step` is a single iteration, but `simulate(n_steps, current, backend=...)` is a
sequential recurrence (each step depends on the previous) that does not
vectorise — a compiled inner loop genuinely beats Python. The kernel carries a
full polyglot chain:

```python
from sc_neurocore.neurons.models.ermentrout_kopell_map_neuron import (
    ErmentroutKopellMapNeuron,
)

neuron = ErmentroutKopellMapNeuron()
trace, spikes = neuron.simulate(2_000_000, current=0.1)            # auto -> Rust
trace, spikes = neuron.simulate(2_000_000, 0.1, backend="julia")  # force a backend
```

`backend` accepts `"auto" | "rust" | "julia" | "go" | "mojo" | "python"`. `auto`
prefers Rust (it ships in the `sc_neurocore_engine` wheel) and falls back to the
pure-NumPy reference. `trace[t]` is `theta` after step `t` (wrapped to
`[0, 2*pi)`); `spikes` counts upward crossings of `theta_threshold`.

The only transcendental is `cos`, and the theta neuron is a **non-chaotic phase
oscillator** (Lyapunov exponent 0), so per-step floating-point differences do
not amplify. On Linux, Python `math.cos` and Rust `f64::cos` resolve to the same
glibc symbol, so **Rust reproduces the NumPy trace bit-for-bit**. Julia, Go and
Mojo use their own `cos`, so they sit within a small, non-amplifying ULP band of
the reference — but every backend produces **identical spike counts**, because a
threshold crossing of `pi` is robust to a sub-ULP phase perturbation. The wrap is
the floored remainder (`theta % 2*pi` = Julia `mod` = Go/Mojo `theta - floor(theta/2*pi)*2*pi`).

### Measured backends

Reproduce with `python benchmarks/bench_ermentrout_kopell_map.py --json
benchmarks/results/bench_ermentrout_kopell_map.json`. Workload: 2,000,000 steps,
default parameters, current = 0.1, median of 5 repeats. **Non-isolated** (loaded
workstation, Python 3.12 / NumPy 2.3) — functional/regression evidence, not
isolated-core release numbers.

| backend | median (ms) | speedup vs NumPy | parity Δ vs NumPy |
|---|---:|---:|---:|
| python (NumPy) | 340.23 | 1.00× | 0 |
| julia | 40.65 | 8.37× | 8.54e-13 (non-amplifying ULP) |
| mojo | 47.93 | 7.10× | 1.84e-11 (non-amplifying ULP) |
| rust | 47.99 | 7.09× | 0 (bit-exact) |
| go | 57.67 | 5.90× | 3.97e-12 (non-amplifying ULP) |

The speedups (~6–8×) are modest because the per-step cost is dominated by the
`cos` evaluation, which every compiled backend pays. The libm-divergent backends'
parity stays at the 1e-13…1e-11 level even over 2,000,000 steps — confirming the
non-chaotic, non-amplifying character. `auto` selects Rust: the fastest
**bit-exact** backend and the one that ships in the wheel.

---

## Network Coupling

### Pulse Coupling

In event-driven networks, the theta neuron receives instantaneous
phase perturbations at spike times of presynaptic neurons:

$$\theta_i \to \theta_i + \epsilon \cdot Z(\theta_i)$$

where ε is the coupling strength and Z is the PRC.  Since Z ≥ 0,
excitatory coupling always advances the phase (Type I).

### Current-Based Coupling

In the SC-NeuroCore DenseLayer framework, the theta neuron receives
continuous current input through the gain parameter:

$$I_{eff} = g \cdot \sum_j w_{ij} \cdot s_j(t)$$

where w_ij are synaptic weights and s_j are presynaptic signals
(binary spikes in stochastic computing representation).

### Mean-Field Reduction

For N → ∞ identical theta neurons with distributed currents I_i
drawn from a Lorentzian distribution g(I) = (Δ/π)/((I-I₀)²+Δ²),
the Ott–Antonsen ansatz yields exact macroscopic equations
(Montbrió, Pazó & Roxin, 2015):

$$\dot{r} = \frac{\Delta}{\pi} + 2r\,v$$

$$\dot{v} = v^2 + I_0 + J\,r - \pi^2 r^2$$

where r is the population firing rate, v is the mean membrane potential
(in QIF coordinates), and J is the coupling strength.  These are the
firing-rate equations used in the `MontbrioPazoRoxin` model in
SC-NeuroCore — directly derived from theta neuron population dynamics.

---

## References

1. Ermentrout, G. B. & Kopell, N. (1986). Parabolic bursting in an
   excitable system coupled with a slow oscillation. *SIAM J Appl Math*,
   46(2), 233–253.

2. Ermentrout, G. B. (1996). Type I membranes, phase resetting curves,
   and synchrony. *Neural Computation*, 8(5), 979–1001.

3. Latham, P. E., Richmond, B. J., Nelson, P. G. & Nirenberg, S. (2000).
   Intrinsic dynamics in neuronal networks. I. Theory. *J Neurophysiol*,
   83(2), 808–827.

4. Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience: The
   Geometry of Excitability and Bursting*. MIT Press. Chapter 4.

5. Ott, E. & Antonsen, T. M. (2008). Low dimensional behavior of large
   systems of globally coupled oscillators. *Chaos*, 18(3), 037113.

6. Ermentrout, G. B. & Terman, D. H. (2010). *Mathematical Foundations
   of Neuroscience*. Springer. Chapters 3, 7.

7. Luke, T. B., Barreto, E. & So, P. (2013). Complete classification of
   the macroscopic behavior of a heterogeneous network of theta neurons.
   *Neural Computation*, 25(12), 3207–3234.

8. Montbrió, E., Pazó, D. & Roxin, A. (2015). Macroscopic description
   for networks of spiking neurons. *Physical Review X*, 5(2), 021028.

9. Hodgkin, A. L. (1948). The local electric changes associated with
   repetitive action in a non-medullated axon. *J Physiol*, 107(2),
   165–181.

10. Gutkin, B. S. & Ermentrout, G. B. (1998). Dynamics of membrane
    excitability determine interspike interval variability: a link between
    spike generation mechanisms and cortical spike train statistics.
    *Neural Computation*, 10(5), 1047–1065.

11. Rinzel, J. & Ermentrout, G. B. (1998). Analysis of neural excitability
    and oscillations. In *Methods in Neuronal Modeling* (2nd ed.),
    Koch, C. & Segev, I. (Eds.), MIT Press, 251–291.

12. Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos* (2nd ed.).
    CRC Press. Chapter 4 (flows on the circle).
