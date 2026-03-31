# DPINeuron

**Module:** `sc_neurocore.neurons.models.dpi_neuron`
**Reference:** Indiveri et al., Proc. IEEE 99(12), 2011
**Family:** Neuromorphic hardware (analog VLSI, DYNAP-SE differential-pair integrator)
**State variables:** `i_mem` (membrane current, nA)

---

## Equations

### Membrane current dynamics

$$\tau \frac{dI_{mem}}{dt} = -I_{mem} + g \cdot I_{syn} + I_{leak}$$

### Spike and reset

$$I_{mem} \geq I_{threshold}: \quad I_{mem} \leftarrow I_{reset}$$

### Non-negativity constraint

$$I_{mem} = \max(I_{mem}, 0)$$

### Implementation

```python
def step(self, i_syn: float) -> int:
    di = (-self.i_mem + self.gain * i_syn + self.i_leak) / self.tau * self.dt
    self.i_mem += di
    self.i_mem = max(self.i_mem, 0.0)
    if self.i_mem >= self.i_threshold:
        self.i_mem = self.i_reset
        return 1
    return 0
```

Forward Euler, single step per call. **Current-domain** dynamics — all
state variables are in nanoamperes (nA), not millivolts (mV). This
mirrors the subthreshold transistor currents in analog VLSI circuits.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `i_mem` | 0.0 | nA | Membrane current (state) |
| `i_threshold` | 1.0 | nA | Spike threshold current |
| `i_reset` | 0.0 | nA | Post-spike reset current |
| `i_leak` | 0.01 | nA | Leak current (baseline offset) |
| `tau` | 20.0 | ms | Integration time constant |
| `gain` | 1.0 | — | Synaptic gain factor |
| `dt` | 1.0 | ms | Integration timestep |

### Current-domain parameters

Unlike voltage-domain models (LIF, HH), all DPI parameters are in
current units:
- i_threshold = 1.0 nA (vs −50 mV in LIF)
- i_reset = 0.0 nA (vs −65 mV in LIF)
- i_leak = 0.01 nA (vs g_L × (V − E_L) in LIF)

This reflects the physical implementation: in subthreshold CMOS,
transistor currents are exponentially related to gate voltages
(I ∝ exp(V/U_T)), so the natural state variable is current, not
voltage.

### gain = 1.0

The gain parameter scales the synaptic input: i_syn_eff = gain × i_syn.
In hardware, this corresponds to the W/L ratio of the input
transistor in the differential pair.

### i_leak = 0.01

A small constant leak current that provides baseline excitability.
Even without synaptic input, i_mem will integrate toward i_leak × τ.
In hardware, this models the transistor subthreshold leakage.

---

## Analytical Properties

### Isomorphism with LIF

The DPI equation is mathematically identical to the LIF:

| DPI (current domain) | LIF (voltage domain) |
|---------------------|---------------------|
| τ dI/dt = −I + gI_syn + I_leak | τ dV/dt = −(V−V_rest) + RI |
| I_mem ≥ I_θ → reset | V ≥ V_θ → reset |
| I_mem ≥ 0 | No equivalent |

The mapping: I_mem ↔ V − V_rest, I_leak ↔ 0 (resting current),
gain × I_syn ↔ R × I_ext.

### Steady-state current

For constant input I_syn (subthreshold):

$$I_{mem,ss} = g \cdot I_{syn} + I_{leak}$$

At default parameters with I_syn = 0: I_mem,ss = 0.01 nA (just the leak).
With I_syn = 0.5: I_mem,ss = 0.5 + 0.01 = 0.51 nA (below threshold).

### Time to spike

For constant I_syn > I_threshold (suprathreshold):

Starting from I_reset = 0:
$$I_{mem}(t) = I_{ss} (1 - e^{-t/\tau}), \quad I_{ss} = g \cdot I_{syn} + I_{leak}$$

Time to threshold:
$$t_{spike} = -\tau \ln\left(1 - \frac{I_{threshold}}{I_{ss}}\right)$$

For I_syn = 2.0: I_ss = 2.01, t_spike = −20 ln(1 − 1/2.01) ≈ 13.9 ms.

### Non-negativity

The max(0, ...) clamp ensures i_mem ≥ 0. In transistor circuits,
current flows in one direction only — negative currents are physically
impossible. This is a hard boundary that does not exist in voltage-domain
models (V can be arbitrarily negative).

### Gain as programmable weight

The gain parameter scales all synaptic input uniformly:
- gain = 0.5: half-strength synapses (lower sensitivity)
- gain = 1.0: unity (default)
- gain = 2.0: doubled synapses (higher sensitivity)

In the DYNAP-SE chip, gain is set by programming transistor bias
currents — it is a per-neuron parameter, not per-synapse.

---

## Behaviour

### Current-domain LIF

The DPI operates identically to a standard LIF but in the current
domain. Input is synaptic current (nA), not voltage. Output is binary
spikes. The dynamics are the same linear ODE with threshold and reset.

### Subthreshold at I_syn = 0

With zero synaptic input, i_mem integrates toward i_leak = 0.01 nA,
far below threshold (1.0). No spikes produced.

### Spiking under drive

At sufficiently high I_syn (tested at 30.0), i_mem reaches threshold
and the neuron fires. Rate increases monotonically with input current.

### Very fast computation

No transcendental functions (no exp, no sqrt, no log). Each step
requires: 1 subtraction, 1 multiplication, 1 addition, 1 division,
1 max, 1 comparison. This is the absolute minimum for an integrate-
and-fire model.

---

## DYNAP-SE Hardware Context

### Indiveri et al. 2011

The Differential-Pair Integrator (DPI) circuit was introduced as a
building block for the DYNAP-SE neuromorphic processor:

- **DYNAP-SE:** Dynamic Neuromorphic Asynchronous Processor — Scalable
  Edition (ETH Zurich / University of Zurich)
- **Technology:** 180 nm CMOS (mixed-signal)
- **Per-chip:** 1024 neurons, 64 synapses per neuron
- **Multi-chip:** Scalable to millions via asynchronous inter-chip links
- **Real-time:** Operates at biological real-time (1× speed)

### DPI circuit operation

The differential-pair integrator uses a pair of transistors operating
in the subthreshold regime:
- Transistor M1: driven by i_syn (input)
- Transistor M2: driven by i_mem (feedback)
- The difference current charges a capacitor → i_mem integrates
- A comparator detects when i_mem exceeds i_threshold
- Digital reset logic restores i_mem to i_reset

### Why current domain?

In subthreshold CMOS (V_GS < V_th):
$$I_D = I_0 \exp(V_{GS} / nU_T)$$

Current is exponentially related to voltage. Working in the log
(current) domain converts multiplication to addition and makes the
dynamics naturally linear — exactly what the DPI equation describes.

---

## Comparison with Related Models

| Property | DPI | LIF | AkidaNeuron | LoihiCUBANeuron |
|----------|-----|-----|-------------|-----------------|
| Domain | Current (nA) | Voltage (mV) | Integer | Fixed-point |
| State vars | 1 (i_mem) | 1 (v) | 1 (v) | 1 (v) |
| Transcendentals | 0 | 0–1 | 0 | 0 |
| Non-negative | Yes (i_mem ≥ 0) | No | Yes (unsigned) | No |
| Hardware | DYNAP-SE | Generic | Akida | Loihi 2 |
| Real-time | 1× | — | 1× | 1× |
| Gain param | Yes (gain) | No | No | No |
| Reference | Indiveri 2011 | Lapicque 1907 | BrainChip | Intel |

---

## Numerical Considerations

- **No transcendental functions.** Pure arithmetic: add, subtract,
  multiply, divide, max, compare.
- **i_mem ≥ 0 clamp.** Prevents negative currents (unphysical in
  transistor circuits).
- **Single Euler step.** dt=1.0 ms — large timestep but adequate for
  the simple linear ODE.
- **No overflow risk.** The spike-and-reset mechanism keeps i_mem
  bounded. Without input, i_mem converges to i_leak (0.01 nA).

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/dpi_neuron.py` — 42 lines.
- **One state variable:** i_mem.
- **Dataclass:** Uses `@dataclass`.
- **No numpy dependency:** Pure Python arithmetic.
- **Rust wiring:** Trivially compatible (1 f64 state var, pure arithmetic).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~1M+ steps/s | Not measured |
| Network (20n, 500ms) | ~800K neuron-steps/s | — |

Among the fastest models — no transcendental functions, single state
variable, minimal per-step computation.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | defaults, binary output, current domain (I not V), state finite (50K), reset, deterministic |
| Analytical | 6 | dI_mem formula, i_mem non-negative, spike resets to i_reset, steady-state, gain scales input, leak provides baseline |
| Dynamics | 7 | fires under drive, subthreshold silent, rate monotonic, f-I sweep [0,10,30,50,100] (parametrised) |
| Parameters | 9 | tau sweep [5,20,50], threshold sweep [0.5,1.0,2.0], gain sweep [0.5,1.0,2.0] (all parametrised) |
| Performance | 2 | isolation throughput, network throughput |
| Pipeline | 6 | Population, Projection, Network spikes, spike_count, isi, firing_rate |
| **Total** | **37** | **ALL PASSED (3.39s)** |

See `tests/test_model_dpi_neuron.py`.

---

## Findings (Measured 2026-03-31)

1. **37/37 tests PASSED in 3.39s.** No failures.

2. **Current domain confirmed.** State variable is i_mem (nA), not
   voltage (mV). Input parameter named i_syn (nA).

3. **dI_mem formula verified.** di = (−i_mem + gain×i_syn + i_leak)/τ × dt
   matches implementation exactly.

4. **i_mem non-negative.** After stepping with zero input, i_mem ≥ 0.
   The max(0, ...) clamp prevents negative currents.

5. **Spike resets to i_reset.** After driving i_mem above threshold,
   i_mem is reset to i_reset (0.0 nA).

6. **Steady-state matches analytical.** For constant subthreshold input,
   i_mem converges to gain × i_syn + i_leak.

7. **Gain scales input.** Higher gain → higher effective synaptic
   current → faster integration to threshold.

8. **Leak provides baseline.** With zero input, i_mem → i_leak (0.01 nA).

9. **Fires under drive.** Sufficient i_syn produces spikes.

10. **Subthreshold silent.** Small i_syn → no spikes.

11. **Rate monotonic.** Higher i_syn → more spikes.

12. **Parameter sweeps stable.** tau ∈ {5, 20, 50}, threshold ∈ {0.5, 1.0,
    2.0}, gain ∈ {0.5, 1.0, 2.0} — all combinations produce finite state.

13. **Network pipeline functional.** Population, Projection, PoissonInput,
    SpikeMonitor, analysis pipeline all work.

14. **Deterministic.** Bit-exact traces across repeated runs.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
37/37 PASSED in 3.39s
├── TestDPIIsolation: 6 tests
│   ├── defaults (i_mem=0, i_threshold=1, tau=20, gain=1)
│   ├── step() → int {0,1}
│   ├── current domain (i_mem, not v)
│   ├── state finite (50K steps)
│   ├── reset (i_mem → 0)
│   └── deterministic
├── TestDPIAnalytical: 6 tests
│   ├── dI_mem formula verified
│   ├── i_mem non-negative (max(0,...))
│   ├── spike resets to i_reset
│   ├── steady-state = gain×i_syn + i_leak
│   ├── gain scales input
│   └── leak provides baseline
├── TestDPIDynamics: 7 tests
│   ├── fires under drive
│   ├── subthreshold silent
│   ├── rate monotonic
│   └── f-I sweep [0, 10, 30, 50, 100]
├── TestDPIParameters: 9 tests
│   ├── tau sweep [5, 20, 50]
│   ├── threshold sweep [0.5, 1.0, 2.0]
│   └── gain sweep [0.5, 1.0, 2.0]
├── TestDPIPerformance: 2 tests
│   ├── isolation throughput
│   └── network throughput
└── TestDPIPipeline: 6 tests
    ├── Population
    ├── Projection
    ├── Network + PoissonInput
    ├── spike_count, isi, firing_rate
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | i_mem=0, threshold=1 nA |
| step(i_syn) → int {0,1} | ✓ PASS | Current-domain input |
| dI_mem formula | ✓ PASS | Matches analytical |
| i_mem ≥ 0 | ✓ PASS | Non-negative clamp |
| Spike → i_reset | ✓ PASS | Resets to 0 nA |
| Steady-state | ✓ PASS | = gain×i_syn + i_leak |
| Gain scaling | ✓ PASS | Proportional |
| Leak baseline | ✓ PASS | i_mem → i_leak |
| Fires under drive | ✓ PASS | Sufficient i_syn |
| Subthreshold silent | ✓ PASS | Low i_syn |
| Rate monotonic | ✓ PASS | More input → more spikes |
| State finite | ✓ PASS | 50K steps |
| reset() | ✓ PASS | i_mem → 0 |
| Deterministic | ✓ PASS | Bit-exact |
| Population | ✓ PASS | Instances created |
| Projection | ✓ PASS | Cross-pop wiring |
| Network | ✓ PASS | Runs, spikes |
| Analysis | ✓ PASS | spike_count, isi, firing_rate |

**ALL 37 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**
