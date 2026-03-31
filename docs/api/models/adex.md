# AdExNeuron

**Module:** `sc_neurocore.neurons.models.adex`
**Reference:** Brette & Gerstner, J. Neurophysiol. 94(5), 2005
**Family:** Integrate-and-Fire with exponential spike initiation and adaptation
**State variables:** `v` (membrane potential), `w` (adaptation current)

---

## Equations

### Membrane potential

$$C_m \frac{dV}{dt} = -g_L(V - V_{rest}) + g_L \Delta_T \exp\!\left(\frac{V - V_{rh}}{\Delta_T}\right) - w + I$$

Rearranged (as implemented):

$$\frac{dV}{dt} = \frac{-(V - V_{rest}) + \Delta_T \exp\!\left(\frac{V - V_{rh}}{\Delta_T}\right)}{\tau} + \frac{-w + I}{C_m}$$

where $\tau = C_m / g_L$.

### Adaptation current

$$\frac{dw}{dt} = \frac{a(V - V_{rest}) - w}{\tau_w}$$

### Spike and reset

$$V \geq V_{threshold}: \quad V \leftarrow V_{reset}, \quad w \leftarrow w + b$$

### Implementation (as coded)

```python
def step(self, current: float) -> int:
    exp_term = self.delta_t * np.exp(
        np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0)
    )
    dv = (
        (-(self.v - self.v_rest) + exp_term) / self.tau
        + (-self.w + current) / self.c_m
    ) * self.dt
    dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt

    self.v += dv
    self.w += dw

    if self.v >= self.v_threshold:
        self.v = self.v_reset
        self.w += self.b
        return 1
    return 0
```

Forward Euler, single step per call. Exponential argument clipped to
[-20, 20] to prevent overflow.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane potential (initial) |
| `w` | 0.0 | pA | Adaptation current (initial) |
| `v_rest` | −65.0 | mV | Resting potential |
| `v_reset` | −68.0 | mV | Post-spike reset potential |
| `v_threshold` | −50.0 | mV | Spike threshold |
| `v_rh` | −55.0 | mV | Rheobase voltage (exponential midpoint) |
| `delta_t` | 2.0 | mV | Spike sharpness (slope factor) |
| `tau` | 20.0 | ms | Membrane time constant (C_m / g_L) |
| `tau_w` | 100.0 | ms | Adaptation time constant |
| `a` | 0.5 | nS | Subthreshold adaptation coupling |
| `b` | 7.0 | pA | Spike-triggered adaptation increment |
| `c_m` | 200.0 | pF | Membrane capacitance |
| `dt` | 0.1 | ms | Integration timestep |

---

## Analytical Properties

### Subthreshold steady state (I constant, no spikes)

Setting $dV/dt = 0$ and $dw/dt = 0$:

$$w_{ss} = a(V_{ss} - V_{rest})$$

$$0 = -(V_{ss} - V_{rest}) + \Delta_T \exp\!\left(\frac{V_{ss} - V_{rh}}{\Delta_T}\right) + \tau\frac{I - w_{ss}}{C_m}$$

This is a transcendental equation — no closed-form solution for $V_{ss}$.
The exponential term dominates when $V$ approaches $V_{rh}$, creating the
characteristic upstroke.

### Rheobase current

The minimum current to elicit a spike (in the non-adaptive case, a=0, b=0):

$$I_{rh} = g_L(V_{rh} - V_{rest}) - g_L \Delta_T$$

With defaults ($V_{rh} = -55$, $V_{rest} = -65$, $\Delta_T = 2$, $\tau = 20$,
$C_m = 200$, so $g_L = C_m/\tau = 10$):

$$I_{rh} = 10 \times 10 - 10 \times 2 = 80 \text{ pA}$$

In practice, the adaptation current $w$ raises the effective rheobase.
Measured: I=0 produces zero spikes, I=200 produces spikes.

### Spike sharpness

The slope factor $\Delta_T$ controls the transition from subthreshold
integration to spike generation:

- $\Delta_T \rightarrow 0$: approaches the hard-threshold LIF (perfect step)
- $\Delta_T = 1$ mV: sharp spike initiation (cortical pyramidal cells)
- $\Delta_T = 2$ mV: moderate (default, layer 5 pyramidal)
- $\Delta_T = 5$ mV: soft spike onset

The exponential term $\Delta_T \exp((V - V_{rh})/\Delta_T)$ is negligible
when $V \ll V_{rh}$ (subthreshold: pure LIF behaviour) and dominates when
$V \approx V_{rh}$ (suprathreshold: exponential blowup → spike).

### Adaptation dynamics

Between spikes, $w$ decays exponentially toward $a(V - V_{rest})$ with time
constant $\tau_w$:

$$w(t) = w_0 \, e^{-t/\tau_w} + a(V - V_{rest})(1 - e^{-t/\tau_w})$$

At each spike, $w$ jumps by $b$. After $n$ spikes, the accumulated adaptation
is approximately $w \approx n \cdot b \cdot e^{-\text{elapsed}/\tau_w}$,
which lengthens subsequent ISIs — spike-frequency adaptation.

### Firing patterns (Brette & Gerstner 2005, Table 1)

| Pattern | Parameters | Mechanism |
|---------|-----------|-----------|
| Tonic spiking | a=0, b=0 | No adaptation → constant ISI |
| Adaptation | a=0, b>0 | Spike-triggered increment → ISI lengthens |
| Initial burst | a=0, b large | Strong first-spike w jump suppresses later spikes |
| Regular bursting | a>0, b>0 | Subthreshold + spike adaptation interact |
| Delayed spiking | a>0, b=0 | Subthreshold w opposes depolarisation |
| Transient | a=0, b very large | Single spike then silence |

---

## Behaviour

### Spike-frequency adaptation

The signature feature of the AdEx model. With default parameters (b=7.0):
- Early ISIs are short (w is small, low suppression)
- Late ISIs are longer (w accumulates, suppresses firing)
- Verified: ISI lengthens over time at I=500

Setting b=0 eliminates adaptation entirely — the model becomes an
Exponential IF (EIF) with constant ISI (CV < 0.05 measured).

### Adaptation strength controls firing rate

Stronger adaptation (larger b) → fewer spikes for the same input current:
- b=2.0: many spikes in 10,000 steps
- b=20.0: fewer spikes in 10,000 steps
- Verified: s_weak > s_strong

### w decay between spikes

The adaptation current w decays toward $a(V - V_{rest})$ between spikes.
With a=0: $w \rightarrow 0$ exponentially. With a>0: $w \rightarrow
a(V - V_{rest})$, which can be positive (subthreshold adaptation, opposing
depolarisation) or negative (facilitating depolarisation) depending on the
voltage.

Verified: w=50 with I=0 → w decays below 50 after 1000 steps.

### Exponential overflow protection

The exponential term $\exp((V - V_{rh})/\Delta_T)$ is clipped to
$\exp(\pm 20)$ to prevent IEEE 754 overflow. Without clipping, a single
missed spike (V exceeds threshold without detection) would produce
$\exp(50/2) = \exp(25) \approx 7.2 \times 10^{10}$, causing NaN propagation.

Verified: setting v=100 (far above threshold) and stepping produces finite v.

---

## Comparison with Related Models

| Property | LIF | EIF | AdEx | Izhikevich |
|----------|-----|-----|------|------------|
| State variables | 1 (V) | 1 (V) | 2 (V, w) | 2 (V, u) |
| Spike mechanism | Hard threshold | Exponential | Exponential | Quadratic |
| Adaptation | None | None | w += b on spike | u += d on spike |
| Subthreshold | Linear decay | Linear + exp | Linear + exp + w | Quadratic + u |
| Biological match | Poor | Better near threshold | Good (cortical cells) | Good (phenomenological) |
| Firing patterns | 1 (tonic) | 1 (tonic) | 6+ (Table 1) | 20+ (Izhikevich 2004) |

The AdEx is the biophysically-motivated counterpart to the Izhikevich model.
Both capture adaptation via a second variable, but AdEx uses exponential
spike initiation (matching cortical neuron voltage traces more closely),
while Izhikevich uses a quadratic nonlinearity (computationally simpler).

---

## Numerical Considerations

- **dt stability:** Tested at dt = 0.05, 0.1, 0.2. All produce finite states
  after 10,000 steps at I=500. The exponential term makes the model stiff
  near threshold — dt > 0.5 may cause missed spikes or instability.
- **Exponential clipping:** Critical for robustness. The clip range [-20, 20]
  corresponds to exp values in [2.06e-9, 4.85e8], well within float64 range.
- **Single-step Euler:** No sub-stepping. Adequate for default dt=0.1ms
  because the exponential blowup is immediately caught by the threshold
  check. Multi-step methods (RK4) would not improve accuracy significantly
  because the spike discontinuity is the dominant error source.
- **Adaptation stiffness:** With large b (>50), the w jump can cause the
  next dV to overshoot v_reset significantly on the first post-spike step.
  This is physically correct (w is an inhibitory current) but can produce
  V < -200 mV. No clipping is applied to V.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/adex.py` — 56 lines.
- **Two state variables:** v (membrane) and w (adaptation).
- **Dataclass:** Uses `@dataclass` for parameter storage and defaults.
- **No sub-stepping:** Single Euler step per call. Efficient but stiff-sensitive.
- **Rust wiring:** Compatible with `step(f64) → i32` dispatch. Two f64 state
  variables. Supported via NeuronVariant.

---

## Infrastructure Pipeline

```
AdExNeuron
├── step(current) → int {0, 1}
├── 1 Euler step per call (dt=0.1ms)
├── Population, Network, SpikeMonitor: compatible
│   PoissonInput(weight=500, rate=500Hz)
├── Projection: tested src→tgt wiring
├── Analysis: spike_count, isi, firing_rate verified
└── Rust: supported (2 f64 state vars, step dispatch)
```

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~500 Ksteps/s | Not measured |
| Network (10 neurons, 2s) | ~40K neuron-steps/s | — |

Fast model — single Euler step, 1 exp() per call. The np.clip + np.exp
is the dominant cost. No sub-stepping overhead.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | defaults, binary, 2-var evolution, finite 50k, reset, exp clipping |
| Adaptation | 5 | w increments on spike, ISI lengthens, w decays, b=0 no adaptation (CV<0.05), stronger b fewer spikes |
| Exponential | 1 | sharp vs soft delta_T both fire |
| f–I curve | 2 | subthreshold silent, monotonic (4-point) |
| Parameters | 2 | dt stability (3 values), deterministic (200 steps bit-exact) |
| Pipeline | 4 | Population, Network+drive, Projection wiring, analysis (spike_count, isi, firing_rate) |
| **Total** | **20** | |

See `tests/test_model_adex.py`. No bugs found.

---

## Findings

1. **ISI adaptation confirmed:** Early ISIs are shorter than late ISIs at
   I=500, consistent with spike-frequency adaptation from w += b accumulation.

2. **b=0 eliminates adaptation:** CV(ISI) < 0.05 measured — the model
   behaves as a pure EIF with constant firing rate.

3. **Stronger b → fewer spikes:** b=2.0 produces more spikes than b=20.0
   at the same input current, confirming that adaptation strength is
   controlled by the spike-triggered increment.

4. **w decay verified:** Starting from w=50 with zero input, w decays
   below 50 after 1000 steps (τ_w = 100ms = 1000 steps at dt=0.1ms).

5. **Exponential clipping prevents overflow:** v=100 (far above threshold)
   produces finite output — the clip to [-20, 20] in the exp argument
   prevents IEEE 754 overflow.

6. **dt robust in tested range:** dt=0.05, 0.1, 0.2 all produce finite
   states after 10,000 steps. No sub-stepping needed at these timesteps.

7. **Network pipeline functional:** Population(n=10) + PoissonInput(500Hz,
   weight=500) + SpikeMonitor produces spikes after 2s simulation.
   Projection wiring from source to target accepted by Network.

8. **Deterministic:** Two identical runs (200 steps at I=500) produce
   bit-exact identical traces for v, w, and spike pattern. No stochastic
   components in the model.

9. **Analysis pipeline verified:** spike_count, isi, and firing_rate all
   produce correct results from the binary spike train. ISI array has
   length equal to spike_count − 1. firing_rate > 0 for suprathreshold
   input.

10. **Monotonic f–I curve:** Firing rate increases monotonically across
    I = [200, 500, 1000, 2000]. No non-monotonic regions in the tested
    range, consistent with Type-I excitability (saddle-node on invariant
    circle bifurcation at threshold).
