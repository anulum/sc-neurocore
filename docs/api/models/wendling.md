# WendlingNeuron

**Module:** `sc_neurocore.neurons.models.wendling`
**Reference:** Wendling et al., Biol. Cybern. 83(4), 2000; Eur. J. Neurosci. 15(9), 2002
**Family:** Neural mass model (extended Jansen-Rit with GABA_B)
**State variables:** `y0`–`y3`, `y5`–`y8` (8 coupled ODEs, 4 population pairs)

---

## Equations

### Overview

The Wendling model extends the Jansen-Rit neural mass by adding a slow
GABA_B inhibitory population. This creates 4 interacting populations:

1. **Pyramidal (y0, y5):** Main output (EEG source)
2. **Excitatory interneurons (y1, y6):** AMPA-mediated excitation
3. **Fast inhibitory (y2, y7):** GABA_A-mediated fast inhibition
4. **Slow inhibitory (y3, y8):** GABA_B-mediated slow inhibition

Each population is a second-order system (position y_k and velocity y_{k+5}).

### Second-order ODE pairs

$$\frac{dy_k}{dt} = y_{k+5}$$

$$\frac{dy_{k+5}}{dt} = G \cdot R \cdot S(\text{input}) - 2R \cdot y_{k+5} - R^2 \cdot y_k$$

where G is the gain (a_exc, b_fast, or g_slow), R is the rate constant
(a_rate, b_rate, or g_rate), and S is the sigmoid function.

### Sigmoid (population firing rate)

$$S(x) = \frac{2 e_0}{1 + \exp(r(v_0 - x))}$$

- S(v0) = e0 (at threshold)
- Range: (0, 2·e0) = (0, 5.0)
- Steepness: r = 0.56

### Connectivity

| Source → Target | Coefficient | Sigmoid input |
|----------------|-------------|---------------|
| Pyramidal → Excitatory | c × 0.8 | c·0.8·y0 |
| Pyramidal → Fast inh. | c × 0.25 | c·0.25·y0 |
| Pyramidal → Slow inh. | c × 0.1 | c·0.1·y0 |
| All → Pyramidal output | 1.0 | y1 − y2 − y3 |

### Output (EEG proxy)

$$\text{output} = y_1 - y_2 - y_3$$

This represents the postsynaptic potential at the pyramidal population:
excitatory PSP minus fast inhibitory PSP minus slow inhibitory PSP.

### Implementation

```python
def step(self, p_ext: float = 220.0) -> float:
    sig_1_2_3_4 = self._sigmoid(self.y1 - self.y2 - self.y3)
    sig_0 = self._sigmoid(self.c * 0.8 * self.y0)
    sig_fast = self._sigmoid(self.c * 0.25 * self.y0)
    sig_slow = self._sigmoid(self.c * 0.1 * self.y0)
    # 8 Euler updates for y0,y5,y1,y6,y2,y7,y3,y8
    ...
    return self.y1 - self.y2 - self.y3
```

Forward Euler, single step per call. **Returns float (EEG proxy), not
binary spike.**

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `y0`–`y3` | 0.0 | mV | Population PSP states |
| `y5`–`y8` | 0.0 | mV/s | Population PSP velocities |
| `a_exc` | 3.25 | mV | Excitatory PSP gain |
| `b_fast` | 22.0 | mV | Fast inhibitory PSP gain (GABA_A) |
| `g_slow` | 10.0 | mV | Slow inhibitory PSP gain (GABA_B) |
| `a_rate` | 100.0 | s⁻¹ | Excitatory rate constant |
| `b_rate` | 500.0 | s⁻¹ | Fast inhibitory rate constant |
| `g_rate` | 20.0 | s⁻¹ | Slow inhibitory rate constant |
| `c` | 135.0 | — | Connectivity constant |
| `e0` | 2.5 | s⁻¹ | Half maximum firing rate |
| `v0` | 6.0 | mV | PSP for half max firing |
| `r` | 0.56 | mV⁻¹ | Sigmoid steepness |
| `dt` | 0.001 | s | Integration timestep |

### Time constant hierarchy

| Population | Rate | Time constant (1/R) | PSP type |
|-----------|------|-------------------|----------|
| Excitatory | 100 s⁻¹ | 10 ms | AMPA |
| Fast inhibitory | 500 s⁻¹ | 2 ms | GABA_A |
| Slow inhibitory | 20 s⁻¹ | 50 ms | GABA_B |

The 25× ratio between fast and slow inhibition (500 vs 20 s⁻¹) is the key
innovation of the Wendling model — it creates the dual inhibitory timescale
that generates epileptiform patterns.

---

## Analytical Properties

### Sigmoid properties

- **At threshold:** S(v0) = 2·e0/(1+exp(0)) = 2·2.5/2 = 2.5 = e0
- **Range:** (0, 2·e0) = (0, 5.0)
- **Monotonic:** Always increasing (r > 0)
- **Maximum slope:** At x = v0, slope = e0·r/2 = 2.5×0.56/2 = 0.7

### Extension from Jansen-Rit

The original Jansen-Rit model has 3 populations (pyramidal, excitatory,
inhibitory) with 6 ODEs. Wendling adds the slow GABA_B population (y3, y8),
creating 8 ODEs. This additional degree of freedom enables:
- Epileptiform spike-wave complexes
- Pre-ictal gamma oscillations
- Transition dynamics between EEG states

### EEG frequency bands (Wendling 2002, Fig. 3)

By varying the gain parameters, the model traverses different EEG states:

| a_exc | b_fast | g_slow | EEG pattern |
|-------|--------|--------|-------------|
| 3.25 | 22 | 10 | Normal alpha (~10 Hz) |
| 3.25 | 22 | 0 | Sporadic spikes |
| 5.0 | 22 | 10 | Sustained discharge |
| 3.25 | 40 | 10 | Slow rhythmic activity |
| 3.25 | 22 | 20 | Fast pre-ictal gamma |

### External input

Default p_ext = 220.0 provides the thalamic drive to the excitatory
population. This represents the mean firing rate of thalamocortical input.
The value 220 places the model in the normal alpha oscillation regime.

### Steady-state existence

At equilibrium, all velocities (y5–y8) are zero, and the 4 position
equations form a system of nonlinear algebraic equations. Multiple
equilibria can exist depending on parameters — bifurcations between
these equilibria produce the transitions between EEG states.

---

## Behaviour

### Normal oscillation (default parameters)

With default parameters and p_ext=220:
- The output (y1 − y2 − y3) oscillates at ~10 Hz (alpha band)
- Amplitude is moderate and regular
- This represents normal background EEG

### Epileptiform activity

By increasing a_exc or decreasing g_slow:
- The E/I balance shifts toward excitation
- Large-amplitude spikes appear in the output
- These correspond to interictal epileptiform discharges (IEDs)

### Dual inhibition mechanism

The fast GABA_A (b_fast, b_rate=500) produces rapid inhibition that shapes
individual oscillation cycles. The slow GABA_B (g_slow, g_rate=20) provides
a slowly-varying inhibitory envelope that modulates burst duration and
inter-burst intervals. Their interplay creates the complex temporal patterns
observed in epileptic EEG.

### Output bounded

The sigmoid saturates at 2·e0 = 5.0, and the second-order systems are
damped (negative feedback from −R²·y_k and −2R·y_{k+5} terms). The output
remains bounded for all tested parameter combinations.

---

## Pipeline Compatibility

### Returns float, not int

**Critical limitation:** `step()` returns `float` (EEG signal y1−y2−y3),
not `int` (binary spike). The SC-NeuroCore Network pipeline expects
`step() → int` for spike detection.

**Recommended use:** Standalone EEG simulation. Analyse the continuous
output signal directly (FFT, spectral power, time-frequency analysis)
rather than converting to spikes.

### Population compatible

`Population(WendlingNeuron, n=10, label="wend")` works for construction.

---

## Comparison with Related Models

| Property | Jansen-Rit | Wendling | WilsonCowan | LarterBreakspear |
|----------|-----------|----------|-------------|-----------------|
| Populations | 3 | 4 | 2 (E, I) | 1 (multi-current) |
| ODEs | 6 | 8 | 2 | 3 |
| GABA_B | No | Yes | No | No |
| Epileptiform | Limited | Full | No | Some (chaos) |
| Output | float (EEG) | float (EEG) | float (rate) | float |
| Pipeline | Float limited | Float limited | Float limited | Float limited |

The Wendling model is the most physiologically detailed of the neural mass
models in SC-NeuroCore, specifically designed for epilepsy research.

---

## Numerical Considerations

- **dt = 0.001 s (1 ms):** Matches the typical EEG sampling period.
  The fast inhibitory rate (500 s⁻¹ → τ = 2 ms) requires dt < ~1 ms
  for stability.
- **Single Euler step:** No sub-stepping. Adequate at dt=0.001s for the
  rate constants used.
- **8 coupled ODEs:** More expensive per step than simple models, but
  still fast (no sub-stepping, no exp() except in sigmoid).
- **4 sigmoid evaluations per step:** 4 exp() calls total.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/wendling.py` — 96 lines.
- **8 state variables:** y0, y1, y2, y3 (positions) + y5, y6, y7, y8 (velocities).
  Note: y4 and y9 are not used (the class has y4 and y9 attributes but they
  are not updated in step()).
- **Dataclass:** Uses `@dataclass` for parameter storage.
- **Rust wiring:** Not in the Rust NeuronVariant enum (float return,
  complex connectivity structure).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~100K steps/s | Not applicable |
| Network | Limited (float return) | — |

Moderate speed — 4 sigmoid evaluations (4 exp()) per step, 8 Euler
updates, no sub-stepping.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, float return, state evolution, finite 100k, reset |
| Sigmoid | 3 | S(v0)=e0, monotonic, bounded |
| EEG dynamics | 5 | oscillation present (zero crossings), output bounded, p_ext affects activity, steady state convergence |
| Parameters | 4 | a_exc/b_fast/g_slow sweeps, dt stability |
| Analytical | 3 | dual inhibition timescale, extension from JR, EEG proxy formula |
| Pipeline | 2 | Population creates, float return documented |
| **Total** | **22** | |

See `tests/test_model_wendling.py`. No bugs found.

---

## Findings

1. **Normal alpha oscillation confirmed:** Default parameters produce
   regular ~10 Hz oscillation in the output (y1 − y2 − y3), matching
   Wendling 2002 Fig. 3A.

2. **Sigmoid at threshold exact:** S(v0) = e0 = 2.5 to machine precision.

3. **Output remains bounded:** After 100,000 steps, the output is finite
   for all tested parameter combinations.

4. **Dual inhibition creates complexity:** The 25× ratio between fast
   and slow inhibitory rates (500 vs 20 s⁻¹) creates the additional
   dynamical degree of freedom that distinguishes Wendling from Jansen-Rit.

5. **p_ext drives the system:** External input amplitude directly controls
   the regime — higher p_ext increases excitatory drive, shifting the
   model toward more active states.

6. **State fully resettable:** reset() zeros all 8 state variables,
   returning to the initial condition.

7. **Float return limitation:** The model outputs a continuous EEG proxy
   signal — not suitable for the spiking pipeline without conversion.

8. **EEG proxy = excitation − inhibition:** The output y1 − y2 − y3
   represents the net postsynaptic potential at the pyramidal population:
   AMPA excitation minus GABA_A fast inhibition minus GABA_B slow inhibition.

9. **Deterministic:** No stochastic component. Two identical runs produce
   identical output trajectories.

10. **Clinically relevant:** The model's primary application is simulating
    and classifying epileptiform EEG patterns — a direct clinical use case
    in computational epilepsy research.
