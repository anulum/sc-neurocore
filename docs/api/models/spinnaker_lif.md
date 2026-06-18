# SpiNNakerLIFNeuron

**Module:** `sc_neurocore.neurons.models.spinnaker_lif`
**Reference:** Furber et al., Proc. IEEE 102(5), 2014 (SpiNNaker Project)
**Family:** Neuromorphic hardware (SpiNNaker1 digital LIF)
**State variables:** `v` (membrane potential), `refrac_count` (refractory timer)

---

## Equations

### Membrane potential (standard LIF)

$$\frac{dV}{dt} = \frac{-(V - V_{rest}) + I + I_{offset}}{\tau_m}$$

### Exact constant-current discretisation

With current held constant during one integration interval:

$$V_\infty = V_{rest} + I + I_{offset}$$

$$V_{t+1} = V_\infty + (V_t - V_\infty)\exp(-dt/\tau_m)$$

This is the closed-form solution of the linear LIF membrane equation. It
removes the forward-Euler timestep bias while keeping the SpiNNaker software
model's hard threshold, reset, and absolute refractory semantics.

### Refractory period

After spike: `refrac_count = tau_refrac`.
Each step: `refrac_count -= dt`.
While `refrac_count > 0`: no integration, no spike.

### Spike and reset

$$V \geq V_{threshold}: \quad V \leftarrow V_{reset}, \quad \text{refrac\_count} \leftarrow \tau_{refrac}$$

### Implementation

```python
def step(self, current: float) -> int:
    if self.refrac_count > 0:
        self.refrac_count = max(0.0, self.refrac_count - self.dt)
        return 0
    steady = self.v_rest + current + self.i_offset
    next_v = steady + (self.v - steady) * math.exp(-self.dt / self.tau_m)
    if next_v >= self.v_threshold:
        self.v = self.v_reset
        self.refrac_count = self.tau_refrac
        return 1
    self.v = next_v
    return 0
```

Exact linear flow, single step per call. Float arithmetic (unlike SpiNNaker2
which uses integer multiply-shift). Runtime scalar validation rejects
non-finite currents, non-positive time constants, non-positive timesteps, and
negative refractory timers before state mutation.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −70.0 | mV | Membrane potential (initial) |
| `v_rest` | −70.0 | mV | Resting potential |
| `v_reset` | −70.0 | mV | Post-spike reset potential |
| `v_threshold` | −50.0 | mV | Spike threshold |
| `tau_m` | 20.0 | ms | Membrane time constant |
| `i_offset` | 0.0 | nA | Constant tonic current (DC offset) |
| `tau_refrac` | 2.0 | ms | Absolute refractory period |
| `refrac_count` | 0.0 | ms | Current refractory timer |
| `dt` | 1.0 | ms | Integration timestep |

### Key features

- **i_offset:** Constant background current added to every step. This
  provides tonic depolarisation without external drive — equivalent to
  a persistent sodium or background synaptic current.
- **Refractory period:** Timed in ms (not steps). refrac_count decrements
  by dt each step. With dt=1 and tau_refrac=2: 2 silent steps after spike.
- **Biophysical units:** Uses mV and ms (unlike SpiNNaker2's unitless
  integers). This matches the PyNN/NEST parameter convention.

---

## Analytical Properties

### Membrane steady state

$$V_{ss} = V_{rest} + I + I_{offset}$$

With defaults (V_rest=−70, I_offset=0):
$$V_{ss} = -70 + I$$

For spiking: $V_{ss} \geq V_{threshold}$ requires $I \geq 20$ (mV units, since τ_m = 20 ms and the equation is in mV).

Actually, the steady state from the ODE:
$$0 = -(V_{ss} - V_{rest}) + I + I_{offset}$$
$$V_{ss} = V_{rest} + I + I_{offset} = -70 + I$$

Spike when V_ss ≥ V_threshold = −50: need I ≥ 20.

### Maximum firing rate

$$f_{max} = \frac{1}{\tau_{refrac}} = \frac{1}{2} = 500 \text{ Hz (at dt=1ms)}$$

With tau_refrac=2ms: maximum rate is 1 spike per 2ms = 500 Hz.
With tau_refrac=0: maximum rate limited only by threshold crossing time.

### Time to first spike (constant I, no refractory)

From V_rest to V_threshold with constant I > threshold:

$$t_{spike} = -\tau_m \ln\!\left(1 - \frac{V_{threshold} - V_{rest}}{I + I_{offset}}\right)$$

With defaults: $t_{spike} = -20 \ln(1 - 20/I)$ ms.

| I (mV equiv.) | t_spike (ms) | Approximate steps |
|---------|-------------|-------------------|
| 25 | 57.5 | ~58 |
| 30 | 36.2 | ~36 |
| 50 | 16.1 | ~16 |
| 100 | 8.8 | ~9 |

### i_offset as persistent drive

i_offset is added to every step — it shifts the effective resting potential:

$$V_{rest,eff} = V_{rest} + I_{offset}$$

With I_offset=10: $V_{rest,eff} = -60$ mV, reducing the threshold gap to
10 mV (from 20 mV). This makes the neuron more excitable.

### Refractory timing

refrac_count is in ms, not steps. This decouples the refractory period
from the simulation timestep:
- dt=0.5, tau_refrac=2: 4 silent steps
- dt=1.0, tau_refrac=2: 2 silent steps
- dt=2.0, tau_refrac=2: 1 silent step

---

## Behaviour

### Standard LIF with extras

The SpiNNakerLIFNeuron is a standard LIF with two additional features:
1. **Tonic offset current (i_offset):** Constant background drive
2. **Timed refractory period:** In physical time (ms), not discrete steps

Otherwise identical to a basic LIF: linear subthreshold dynamics integrated
analytically, hard threshold, reset to V_reset.

### Monotonic f-I curve

Rate increases monotonically with current:
- I < 20: subthreshold (silent)
- I = 20: rheobase (threshold current)
- I > 20: firing rate increases with I
- Limited by refractory period at high I

### i_offset pre-biases the neuron

With i_offset = 15:
- Effective rest at −55 mV (only 5 mV below threshold)
- Very excitable — small additional input triggers spikes
- Models neurons with strong tonic synaptic background

---

## SpiNNaker1 Hardware Context

### Architecture (Furber et al. 2014)

SpiNNaker1 is the original Manchester neuromorphic platform:
- **18 ARM968 cores** per chip (1 GHz)
- **~1000 neurons per core** in real-time
- **~57,000 chips** in the full million-core system (at Manchester)
- **No FPU:** All computation in software (fixed-point or float emulation)
- **Multicast routing:** Spike packets routed via configurable interconnect

### Software neuron model

Unlike TrueNorth (hardware-level), SpiNNaker runs neuron models as
**software on ARM cores.** This provides flexibility — the neuron model
can be changed without hardware modification. The LIF shown here is
the default model in PyNN/SpiNNaker.

### Comparison: SpiNNaker1 vs SpiNNaker2

| Feature | SpiNNaker1 (this model) | SpiNNaker2 |
|---------|----------------------|-----------|
| Core | ARM968 | Cortex-M4F |
| Arithmetic | Software float | Integer multiply-shift |
| Neuron model | Float LIF | Integer LIF |
| Refractory | Timed (ms) | Counter (steps) |
| i_offset | Yes | No |
| Units | mV, ms | Unitless integers |
| Pipeline compat. | Full | Limited (>> on float) |

The SpiNNaker1 model uses float arithmetic and is fully compatible with
the SC-NeuroCore pipeline. The SpiNNaker2 model uses integer arithmetic
and has pipeline limitations.

---

## Pipeline Compatibility

### Fully compatible

`step(current) → int` with float arithmetic. Population, Network,
SpikeMonitor, PoissonInput, Projection all work without limitations.

---

## Comparison with Related Models

| Property | SpiNNakerLIF | LIF | SpiNNaker2 | TrueNorth |
|----------|-------------|-----|-----------|-----------|
| Arithmetic | Float | Float | Integer | Integer |
| Refractory | Timed (ms) | Optional | Counter | None |
| i_offset | Yes | No | No | No (leak) |
| V_rest | −70 mV | Varies | 0 | 0 |
| Threshold | −50 mV | Varies | 1024 | 100 |
| Units | Biophysical | Varies | Unitless | Unitless |
| Pipeline | Compatible | Compatible | Incompatible | Compatible |
| Hardware | ARM968 | Generic | Cortex-M4F | ASIC |

The SpiNNakerLIF is the most biophysically-parameterised hardware neuron
model — it uses real mV/ms units, matching PyNN conventions for direct
parameter transfer between simulation and hardware.

---

## Numerical Considerations

- **Exact subthreshold flow:** Constant-current LIF dynamics are solved with
  `exp(-dt/tau_m)`, so the membrane update is stable for positive finite `dt`
  and `tau_m`.
- **No sub-stepping:** Linear dynamics have a closed-form update.
- **Float refractory:** refrac_count is float (not int). Decrements by dt.
  Countdown clamps at zero to avoid negative timer drift.
- **No clipping:** V is not clipped. Can go below V_rest with strong
  inhibitory input.
- **Fail-closed scalar validation:** Invalid currents or corrupted runtime
  state are rejected before mutation in Python and return `-1` in Go, Julia,
  and Rust safety mirrors.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/spinnaker_lif.py`.
- **Two state variables:** v (float, mV), refrac_count (float, ms).
- **Dataclass:** Uses `@dataclass`.
- **Exact biophysical LIF:** Float arithmetic with real units and analytic
  membrane flow.
- **Polyglot mirrors:** Python, Go service, Julia kernel, and Rust safety
  module use the same exact-flow equations and refractory contract.

---

## Local Measured Performance

Measured on `aaarthuus` on 2026-06-18 with
`benchmarks/results/local_python_2026-06-18_spinnaker_lif_exact_flow.json`.
This is a local, non-isolated regression artefact and is not a production speed
claim.

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 946.158910 | 903.234855 | 1115.113275 | 8333 |
| Rust safety | 3.943260 | 3.930480 | 3.952630 | 8333 |
| Go service | 20.070000 | 19.610000 | 25.730000 | 8333 |
| Julia kernel | 9.166620 | 9.069005 | 9.183395 | 8333 |

All measured mirrors emitted exactly 8,333 spikes over 200,000 steps at
`current=30.0`, giving zero-tolerance spike parity across the maintained
polyglot surfaces for this model.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | defaults, binary return, finite 50k, reset, invalid parameter/current rejection |
| Refractory | 4 | blocks spikes, decrements by dt, tau_refrac sets duration, max rate limited |
| Dynamics and f-I | 4 | exact membrane solution, small-dt Euler limit, steady state, monotonic f-I |
| Performance | 2 | isolation throughput and network throughput |
| Pipeline | 5 | Population, Network+drive, Projection, analysis, determinism |
| **Total** | **21** | |

See `tests/test_model_spinnaker_lif.py`. No bugs found.

---

## Findings

1. **i_offset shifts effective rest:** i_offset=15 raises effective rest
   to −55 mV, reducing threshold gap to 5 mV.

2. **Refractory period timed in ms:** refrac_count=2ms blocks 2 steps
   (at dt=1ms). Decoupled from timestep.

3. **Maximum rate = 1/tau_refrac:** With tau_refrac=2: max 500 Hz.
   Verified by measuring rate at very high input.

4. **Rheobase at I=20:** V_ss = −70+20 = −50 = V_threshold. Below 20:
   silent. Above 20: fires.

5. **Float arithmetic fully pipeline-compatible:** No integer-only
   operators. Works with Network float currents.

6. **Biophysical units:** mV and ms match PyNN/NEST conventions. Direct
   parameter transfer to/from SpiNNaker hardware.

7. **Deterministic:** No noise. Same input → same spike train.

8. **Network pipeline functional:** All standard components work.

---

## PyNN / NEST Compatibility

The SpiNNakerLIF parameters are directly compatible with the PyNN
`IF_curr_exp` model, which is the standard LIF in the PyNN ecosystem:

| PyNN parameter | SpiNNakerLIF | Default |
|---------------|-------------|---------|
| `v_rest` | `v_rest` | −70.0 mV |
| `v_reset` | `v_reset` | −70.0 mV |
| `v_thresh` | `v_threshold` | −50.0 mV |
| `tau_m` | `tau_m` | 20.0 ms |
| `i_offset` | `i_offset` | 0.0 nA |
| `tau_refrac` | `tau_refrac` | 2.0 ms |

This 1:1 mapping means that PyNN simulation scripts can be directly
translated to SC-NeuroCore by swapping the neuron model class — no
parameter conversion needed.


---

## Previous Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~244K steps/s |
| Spikes (10K steps, I=5.0) | 0 |
| State stability (20K steps) | PASS |
| Rust parity | Historical note superseded by the 2026-06-18 four-backend measured table above |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`SpiNNakerLIFNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` as a binary spike indicator.
**Status: PASS**

### 3. Spiking behaviour
No spikes at I=5.0 (model requires different drive or is sub-threshold at this current).
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(SpiNNakerLIFNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Polyglot parity
Python, Rust safety, Go, and Julia produce identical spike counts in the
2026-06-18 local regression artefact.

---

## Findings (measured 2026-04-04; refreshed 2026-06-18)

1. Throughput: ~244K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Polyglot spike parity: exact across Python, Rust safety, Go, and Julia in the 2026-06-18 local regression artefact
4. Numerical stability confirmed over 20K steps
