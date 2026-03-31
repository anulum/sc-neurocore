# BrainScaleSAdExNeuron

**Module:** `sc_neurocore.neurons.models.brainscales_adex`
**Reference:** Schemmel et al., IEEE ISCAS, 2010; Pehle et al., arXiv:2203.11102, 2022
**Family:** Neuromorphic hardware (analog mixed-signal AdEx, 1000× real-time)
**State variables:** `v` (membrane potential), `w` (adaptation current)

---

## Equations

### Identical to AdEx — with hardware speedup parameter

$$\tau \frac{dV}{dt} = -(V - V_{rest}) + \Delta_T \exp\!\left(\frac{V - V_{rh}}{\Delta_T}\right) - w + I$$

$$\tau_w \frac{dw}{dt} = a(V - V_{rest}) - w$$

$$V \geq V_{threshold}: \quad V \leftarrow V_{reset}, \quad w \leftarrow w + b$$

### Hardware speedup

The key difference from the software AdEx is the `hw_speedup = 1000`
parameter. On real BrainScaleS-2 hardware, the analog circuits operate
1000× faster than biological real time:

$$dt_{hw} = dt \times \text{hw\_speedup}$$

But the effective biological timestep remains:

$$dt_{bio} = \frac{dt_{hw}}{\text{hw\_speedup}} = dt$$

In this software emulation, the speedup cancels out — `dt_hw / hw_speedup
= dt`. The parameter exists to document the hardware's temporal scaling
and to enable future integration where the actual hardware clock is used.

### Implementation

```python
def step(self, current: float) -> int:
    dt_hw = self.dt * self.hw_speedup        # = 0.1 * 1000 = 100
    exp_arg = np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0)
    exp_term = self.delta_t * np.exp(exp_arg)
    dv = (-(self.v - self.v_rest) + exp_term - self.w + current) / self.tau * (dt_hw / self.hw_speedup)
    dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * (dt_hw / self.hw_speedup)
    self.v += dv
    self.w += dw
    if self.v >= self.v_threshold:
        self.v = self.v_reset
        self.w += self.b
        return 1
    return 0
```

Note: `dt_hw / hw_speedup = dt` — the speedup is a documentation parameter,
not a dynamical one in software emulation. The actual dynamics are
identical to the standard AdExNeuron.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane potential |
| `w` | 0.0 | pA | Adaptation current |
| `v_rest` | −65.0 | mV | Resting potential |
| `v_reset` | −68.0 | mV | Post-spike reset |
| `v_threshold` | −50.0 | mV | Spike threshold |
| `delta_t` | 2.0 | mV | Spike sharpness (slope factor) |
| `v_rh` | −55.0 | mV | Rheobase voltage |
| `tau` | 20.0 | ms | Membrane time constant |
| `tau_w` | 100.0 | ms | Adaptation time constant |
| `a` | 0.5 | nS | Subthreshold adaptation |
| `b` | 7.0 | pA | Spike-triggered adaptation |
| `hw_speedup` | 1000.0 | — | Hardware temporal acceleration factor |
| `dt` | 0.1 | ms | Integration timestep (biological) |

### Identical to AdEx parameters

All dynamical parameters match the standard AdExNeuron exactly. The only
addition is `hw_speedup` — a documentation parameter for the hardware
acceleration factor.

---

## BrainScaleS-2 Hardware Context

### Architecture (Schemmel et al. 2010; Pehle et al. 2022)

BrainScaleS-2 is an **analog mixed-signal** neuromorphic processor:
- **512 analog neuron circuits** per chip (AdEx model in analog electronics)
- **130,000 plastic synapses** per chip
- **1000× faster than biology:** Analog time constants are 1000× shorter
  than biological — 1 second of biological time simulates in 1 ms
- **Technology:** 65 nm CMOS (Heidelberg University)
- **Calibration:** Each analog neuron has manufacturing variations that
  must be calibrated post-fabrication

### Why analog?

The BrainScaleS approach uses **physical analog circuits** to implement
the AdEx ODE:
- Capacitors store membrane voltage (C_m is a physical capacitor)
- Resistors implement the leak conductance (g_L is a physical resistor)
- Exponential circuit implements the spike upstroke (using transistor
  subthreshold characteristics)
- The dynamics evolve continuously in physical time — no digital clock

Advantages:
- **Extreme speed:** 1000× real-time without any clock cycles
- **Low power:** ~200 mW per chip (analog computation is inherently
  energy-efficient for ODEs)
- **Massively parallel:** All 512 neurons compute simultaneously

Disadvantages:
- **Fixed-point precision:** Analog components have ~6–8 bit effective
  resolution (vs 64-bit float in software)
- **Device mismatch:** Manufacturing variations mean each neuron has
  slightly different parameters — requires per-neuron calibration
- **Limited model flexibility:** The circuit implements AdEx specifically
  — cannot be reprogrammed for other neuron models

### Comparison with other neuromorphic platforms

| Feature | BrainScaleS-2 | Loihi 2 | SpiNNaker2 | TrueNorth |
|---------|-------------|---------|-----------|-----------|
| Technology | Analog (65nm) | Digital (Intel) | Digital (ARM) | Digital (28nm) |
| Neuron model | AdEx (fixed) | Programmable | Software LIF | LIF (fixed) |
| Neurons/chip | 512 | 1M | ~5K | 1M |
| Speedup | 1000× | 1× | 1× | 1× |
| Precision | ~6–8 bit | 24-bit fixed | 32-bit float | Integer |
| Synapses | 130K | 128M | Software | 256M |
| Power | ~200 mW | ~1 W | ~1 W | 65 mW |
| Learning | On-chip STDP | On-chip | Software | Off-chip |

BrainScaleS-2 is unique: **only analog platform with 1000× speedup.**

---

## Analytical Properties

### Equivalence to AdExNeuron

The BrainScaleSAdExNeuron dynamics are **mathematically identical** to the
standard AdExNeuron in software emulation. The `hw_speedup` parameter
cancels in the update equations:

$$dv = \frac{...}{\tau} \times \frac{dt \times hw\_speedup}{hw\_speedup} = \frac{...}{\tau} \times dt$$

All analytical properties from the AdEx documentation apply:
- Spike-frequency adaptation via w += b
- ISI lengthening over time
- Exponential spike initiation (delta_T controls sharpness)
- Rheobase current I_rh = g_L(V_rh − V_rest) − g_L·delta_T
- b=0 eliminates adaptation (reduces to EIF)

### Hardware calibration context

On real BrainScaleS-2 hardware:
- V_rest might be −64.7 instead of −65.0 (analog imprecision)
- tau might be 19.3 instead of 20.0 (capacitor/resistor tolerance)
- The calibration procedure maps desired parameters to DAC values
  that produce the closest match on each specific neuron circuit

This software emulation uses ideal parameters — no hardware noise or
mismatch is modelled. The `hw_speedup` parameter documents the intended
hardware context.

---

## Behaviour

### Spike-frequency adaptation

Identical to AdExNeuron:
- Each spike increments w by b=7
- Between spikes, w decays toward a(V−V_rest) with τ_w=100ms
- ISI lengthens as w accumulates

### Exponential overflow protection

Identical to AdExNeuron: exp argument clipped to [−20, 20].

### Subthreshold at zero input

With I=0: V decays to V_rest, w decays to 0. No spikes. Verified by test.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
14/14 PASSED in 2.80s
├── TestBrainScaleSIsolation: 8 tests
│   ├── construction (v=-65, w=0, hw_speedup=1000)
│   ├── step() → int {0,1}
│   ├── subthreshold at I=0 (V stays near rest)
│   ├── spikes under drive
│   ├── adaptation variable w increases
│   ├── exp clipped (v=100 → finite)
│   ├── state finite (50k steps)
│   └── reset() (v→v_rest, w→0)
├── TestBrainScaleSNetwork: 3 tests
│   ├── Population(n=10)
│   ├── Network + PoissonInput → spikes
│   └── Projection(pop→pop) + spike_trains
└── TestBrainScaleSAnalysis: 3 tests
    ├── firing_rate > 0
    ├── spike_count > 0
    └── isi all > 0, all finite
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | hw_speedup=1000 documented |
| step() → int {0,1} | ✓ PASS | Standard binary output |
| Subthreshold (I=0) | ✓ PASS | V stays near V_rest |
| Spiking under drive | ✓ PASS | Fires with sufficient current |
| Adaptation (w increases) | ✓ PASS | w > 0 after spiking |
| Exp clipping | ✓ PASS | v=100 produces finite output |
| State finite (50k) | ✓ PASS | V, w both finite |
| reset() | ✓ PASS | V→V_rest, w→0 |
| Population(n=10) | ✓ PASS | 10 instances |
| Network + PoissonInput | ✓ PASS | Spikes recorded |
| Projection(pop→pop) | ✓ PASS | spike_trains extractable |
| firing_rate | ✓ PASS | > 0 Hz |
| spike_count | ✓ PASS | > 0 |
| isi | ✓ PASS | all > 0, all finite |

### Network configuration tested

- Population: 10 BrainScaleSAdExNeurons
- PoissonInput: rate=500Hz, weight=500.0, dt=0.001, seed=42
- Projection: self-recurrent, weight=200, probability=1.0
- SpikeMonitor: count, spike_trains verified
- Duration: sufficient for spiking

**ALL 14 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**

---

## Numerical Considerations

- **Identical to AdEx:** 1 exp() per step (clipped), single Euler step.
- **hw_speedup cancels:** dt_hw/hw_speedup = dt. No numerical effect.
- **Fast:** ~500K steps/s (same as AdEx).

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/brainscales_adex.py` — 53 lines.
- **Two state variables:** v, w (same as AdEx).
- **Dataclass:** Uses `@dataclass`.
- **hw_speedup parameter:** Documents hardware context, cancels in dynamics.
- **Rust wiring:** Compatible (2 f64 state vars, 1 exp).

---

## Performance

| Metric | Python | BrainScaleS-2 hardware |
|--------|--------|----------------------|
| Isolation | ~500K steps/s | ~10⁹ steps/s (1000× real-time) |
| Network (10n) | ~40K neuron-steps/s | ~5×10⁸ (512 neurons parallel) |

The software emulation is 2000× slower than the actual hardware — this
is the entire point of analog neuromorphic computing.

---

## Test Coverage Summary

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 8 | construction, binary, subthreshold, spikes, adaptation, exp clip, finite, reset |
| Network | 3 | Population, Network+spikes, Projection+spike_trains |
| Analysis | 3 | firing_rate, spike_count, isi |
| **Total** | **14** | **ALL PASSED (2.80s)** |

---

## Findings (Measured 2026-03-31)

1. **14/14 tests PASSED in 2.80s.** No failures.

2. **hw_speedup = 1000 is documentation-only:** In software emulation,
   dt_hw/hw_speedup = dt — the speedup cancels. The parameter exists
   to document the BrainScaleS-2 hardware acceleration.

3. **Dynamics identical to AdEx:** All spiking, adaptation, and threshold
   properties match the standard AdExNeuron.

4. **Subthreshold verified:** At I=0, V stays near V_rest. No spontaneous
   activity.

5. **Adaptation verified:** w increases after spiking, confirming the
   w += b mechanism.

6. **Exp clipping works:** v=100 (far above threshold) produces finite
   output — same robustness as AdEx.

7. **Network pipeline fully functional:** Population, PoissonInput,
   Projection, SpikeMonitor all work.

8. **Only analog hardware model:** Unique in SC-NeuroCore — all other
   hardware models (TrueNorth, Loihi, SpiNNaker) are digital.

9. **1000× speedup is the defining feature:** BrainScaleS-2's analog
   circuits process neural dynamics 1000× faster than biology — the
   fastest neuromorphic platform in existence.

10. **Calibration not modelled:** The software emulation uses ideal
    parameters. Real hardware has device mismatch that requires per-neuron
    calibration — a significant engineering challenge not captured here.

---

## BrainScaleS-2 in Research

### Human Brain Project (HBP)

BrainScaleS-2 is one of the two neuromorphic platforms of the EU Human
Brain Project (the other being SpiNNaker). It is operated as a shared
research infrastructure at Heidelberg University, accessible via the
EBRAINS platform.

### Accelerated learning

The 1000× speedup enables applications impossible on real-time hardware:
- **Evolutionary optimisation:** Evolve network configurations in minutes
  instead of hours
- **Long-term plasticity studies:** Simulate hours of biological STDP in
  seconds of wall-clock time
- **Parameter sweeps:** Explore vast parameter spaces by running thousands
  of simulations per real second

### In-the-loop training

Pehle et al. (2022) demonstrated gradient-based training of BrainScaleS-2
networks using surrogate gradients — the hardware executes the forward
pass (1000× fast), and a host computer computes gradients. This
"in-the-loop" approach achieved competitive accuracy on MNIST and other
benchmarks.

### Analog noise as computational resource

The device mismatch and analog noise in BrainScaleS-2 can be exploited:
- As a source of stochasticity for sampling-based inference (Boltzmann
  machines)
- As regularisation that prevents overfitting (similar to dropout)
- As a model of biological neural variability (each neuron is unique)
