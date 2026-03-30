# ResonateAndFireNeuron

**Module:** `sc_neurocore.neurons.models.resonate_and_fire`
**Reference:** Izhikevich 2001
**Family:** Oscillatory (complex-valued subthreshold dynamics)
**State variables:** `x`, `y` (real and imaginary parts of complex state z)

---

## Equations

### Complex form

$$\frac{dz}{dt} = (b + i\omega)\,z + I$$

where $z = x + iy$, $b$ is the damping/growth rate, $\omega$ is the natural
oscillation frequency, and $I$ is real-valued input current.

### Decomposed into real ODEs (as implemented)

$$\frac{dx}{dt} = bx - \omega y + I$$
$$\frac{dy}{dt} = \omega x + by$$

### Spike condition

$$|z| = \sqrt{x^2 + y^2} \geq \theta$$

### Reset

On spike: $x \leftarrow 0,\; y \leftarrow 0$ (reset to origin).

### Implementation (as coded)

```python
def step(self, current: float) -> int:
    dx = (self.b * self.x - self.omega * self.y + current) * self.dt
    dy = (self.omega * self.x + self.b * self.y) * self.dt
    self.x += dx
    self.y += dy
    r = np.sqrt(self.x**2 + self.y**2)
    if r >= self.threshold:
        self.x = 0.0
        self.y = 0.0
        return 1
    return 0
```

Forward Euler, single step per call. No sub-stepping.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `x` | 0.0 | Real part of complex state |
| `y` | 0.0 | Imaginary part of complex state |
| `b` | −0.1 | Damping rate (b<0: damped, b>0: unstable) |
| `omega` | 1.0 | Natural oscillation frequency (rad/time) |
| `threshold` | 1.0 | Spike threshold on |z| |
| `dt` | 0.05 | Integration time step |

---

## Analytical Properties

### Steady-state (constant input, subthreshold)

Setting $dx/dt = 0$, $dy/dt = 0$:

$$x_{ss} = \frac{-bI}{b^2 + \omega^2}, \quad y_{ss} = \frac{I\omega}{b^2 + \omega^2}$$

$$r_{ss} = \sqrt{x_{ss}^2 + y_{ss}^2} = \frac{I}{\sqrt{b^2 + \omega^2}}$$

### Critical current

Spike occurs when $r_{ss} \geq \theta$:

$$I_{crit} = \theta \sqrt{b^2 + \omega^2}$$

With defaults ($b = -0.1$, $\omega = 1.0$, $\theta = 1.0$):

$$I_{crit} = \sqrt{0.01 + 1.0} = \sqrt{1.01} \approx 1.005$$

Verified: I=0.5 produces 0 spikes (r_ss ≈ 0.498 < 1.0). I=1.0 fires
(r_ss ≈ 0.995 but transient overshoot crosses threshold).

### Transient overshoot

The spiral approach to equilibrium can transiently exceed $r_{ss}$.
Measured: I = 0.9 × I_crit still produces spikes (2000 in 50k steps)
due to transient overshoot. A 50% margin below I_crit is needed to
guarantee zero spikes.

### Eigenvalues

The system matrix $A = \begin{pmatrix} b & -\omega \\ \omega & b \end{pmatrix}$
has eigenvalues $\lambda = b \pm i\omega$.

- $b < 0$: stable spiral (damped oscillation)
- $b = 0$: centre (undamped, marginally stable)
- $b > 0$: unstable spiral (amplitude grows exponentially)

### Oscillation period

$$T = \frac{2\pi}{\omega}$$

With $\omega = 1.0$, $T \approx 6.28$ time units, or $T/dt \approx 126$ steps.

---

## Behaviour

### Damped subthreshold oscillation (b < 0)

With b=−0.1, the system is a damped oscillator. Under constant subthreshold
input, x and y spiral inward toward the equilibrium point. The oscillation
is clearly visible in the x trace: > 20 zero-crossings of the mean in
4000 post-transient steps at I=0.5.

### Threshold on radius

Spike detection uses the Euclidean norm $|z| = \sqrt{x^2 + y^2}$, not a
simple voltage threshold. This means the neuron can spike via oscillation
build-up in either x or y — it responds to both amplitude and phase of
the input relative to its internal oscillation.

### Unstable regime (b > 0)

When b > 0, any perturbation from origin grows exponentially. Even with
I=0, a tiny initial displacement (x=0.01) eventually reaches threshold
and triggers spikes. Verified: b=0.1 with x_0=0.01 produces spikes with
zero input.

### omega controls oscillation frequency

Higher omega → faster subthreshold oscillation. Measured via zero-crossings
of x(t) around its mean:
- omega=0.5: fewer crossings
- omega=2.0: more crossings

This confirms the model's resonant property: it preferentially responds
to inputs oscillating near its natural frequency omega.

---

## Measured Dynamics (from test probing)

### Constant current sweep (default parameters)

| Current | Spikes (50k) | Mean ISI | r at end | Regime |
|---------|-------------|----------|----------|--------|
| 0.0 | 0 | — | 0.0000 | Origin rest |
| 0.5 | 0 | — | 0.4975 | Subthreshold spiral |
| 1.0 | 2,272 | 22 | 0.7573 | Spiking (just above I_crit) |
| 2.0 | 4,545 | 11 | 0.4950 | Regular spiking |
| 5.0 | 10,000 | 5 | 0.0000 | Fast spiking |
| 10.0 | 16,666 | 3 | 0.9978 | Rapid spiking |
| 20.0 | 50,000 | 1 | 0.0000 | Every-step spiking |

f–I is monotonic. At I=20, the single-step increment exceeds threshold
on every step → spike rate = 1/dt.

### Damping parameter sweep (I=1.5)

| b | Spikes (50k) | Description |
|---|-------------|-------------|
| −0.05 | many | Weak damping, lower effective I_crit |
| −0.5 | fewer | Strong damping, higher effective I_crit |
| +0.1 | spikes even at I=0 | Unstable spiral |

---

## Comparison with Other Models

| Property | LIF | QIF | Resonate-and-Fire |
|----------|-----|-----|-------------------|
| State variables | 1 (V) | 1 (V) | 2 (x, y) |
| Subthreshold dynamics | Exponential decay | Stable/unstable FP | Damped oscillation |
| Excitability | Type-I (linear onset) | Type-I (sqrt onset) | Type-II (resonance) |
| Spike detection | V ≥ θ | V ≥ V_peak | \|z\| ≥ θ |
| Input selectivity | None (integrator) | None (integrator) | Frequency-selective |
| Reset | V → V_reset | V → V_reset | (x,y) → (0,0) |

The key distinction: R&F is a **resonator**, not an integrator. It responds
preferentially to inputs near its natural frequency ω, making it suitable
for modelling neurons in sensory systems that exhibit band-pass filtering.

---

## Numerical Considerations

- **dt stability:** Tested at dt = 0.02, 0.05, 0.1. All produce finite
  states after 50k steps at I=2.0.
- **Euler integration:** The linear ODE system is unconditionally stable
  for b < 0 at any dt (eigenvalues have negative real part). However,
  for b > 0, the Euler scheme can amplify numerical errors — dt must be
  small relative to 1/b.
- **Radius computation:** Uses `np.sqrt(x² + y²)` each step. This is
  the dominant cost per step.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/resonate_and_fire.py` — 45 lines.
- **Two real state variables:** x and y, representing Re(z) and Im(z).
- **Rust wiring:** Compatible with `step(f64) → i32` dispatch. Two f64
  state variables. Supported via NeuronVariant.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 2-var evolution, finite 50k, reset |
| Steady-state | 3 | r_ss = I/sqrt(b²+ω²) at I=0.5, I=0.3; damping decay at b=−0.5 |
| Threshold | 4 | I_crit analytical (50% below → 0 spikes, 20% above → >10 spikes), radius check, reset to origin, custom threshold |
| f–I curve | 4 | monotonic (4-point), excess current scaling, zero input silent, I=20 fires every step |
| Oscillation | 2 | subthreshold x oscillation (>20 zero-crossings), omega frequency scaling |
| Parameters | 5 | b>0 unstable (spikes at I=0), b more negative → fewer spikes, dt stability (3 values) |
| Determinism | 1 | bit-exact (300 steps) |
| ISI | 1 | constant ISI (CV<0.05) |
| Network | 2 | Population(n=10), Network spikes |
| Analysis | 2 | spike_count ≥ 100, consistency |
| **Total** | **29** | |

---

## Findings

1. **Analytical r_ss confirmed:** At I=0.5, measured r = 0.4975, predicted
   r_ss = 0.498. At I=0.3, convergence also matches within 0.01.
2. **Transient overshoot crosses threshold:** At I = 0.9 × I_crit, the
   transient spiral approach overshoots r_ss and triggers spikes. The
   analytical I_crit is a steady-state prediction, not a transient one.
   A 50% safety margin below I_crit is needed for guaranteed silence.
3. **b > 0 unstable confirmed:** With b=0.1 and x_0=0.01, the expanding
   spiral reaches threshold and fires even with zero input.
4. **Damping controls effective threshold:** More negative b raises the
   effective I_crit (stronger damping attenuates the state more).
   Measured at I=1.5: b=−0.05 fires more than b=−0.5.
5. **Omega sets oscillation frequency:** Higher omega produces more
   zero-crossings in the subthreshold x trace, confirming the
   resonance property.
6. **ISI regular at steady state:** CV(ISI) < 0.05 at I=2.0 after
   skipping the first 10 spikes (transient).
