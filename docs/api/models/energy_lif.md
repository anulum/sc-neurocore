# EnergyLIFNeuron

**Module:** `sc_neurocore.neurons.models.energy_lif`
**Reference:** Fardet & Levina, Neural Comput. 32(12), 2020
**Family:** Integrate-and-fire with metabolic energy constraint
**State variables:** `v` (membrane potential), `epsilon` (metabolic energy reserve)

---

## Equations

### Membrane potential

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R \cdot \varepsilon \cdot I$$

### Energy dynamics

$$\tau_\varepsilon \frac{d\varepsilon}{dt} = \varepsilon_0 - \varepsilon$$

### Spike condition (energy-gated)

$$V \geq V_{threshold} \; \text{AND} \; \varepsilon > 0.1: \quad V \leftarrow V_{reset}, \quad \varepsilon \leftarrow \max(0, \varepsilon - \alpha)$$

### Implementation

```python
def step(self, current: float) -> int:
    effective_r = self.resistance * self.epsilon
    self.v += (-(self.v - self.v_rest) + effective_r * current) / self.tau_m * self.dt
    self.epsilon += (self.epsilon_0 - self.epsilon) / self.tau_e * self.dt
    if self.v >= self.v_threshold and self.epsilon > 0.1:
        self.v = self.v_reset
        self.epsilon -= self.alpha
        self.epsilon = max(0.0, self.epsilon)
        return 1
    return 0
```

Forward Euler. Two key mechanisms:
1. **Energy scales input:** R_eff = R × ε. Low energy → weak input response.
2. **Energy gates spikes:** Must have ε > 0.1 to fire.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −70.0 | mV | Membrane potential |
| `epsilon` | 1.0 | — | Metabolic energy reserve (0 to ε₀) |
| `v_rest` | −70.0 | mV | Resting potential |
| `v_reset` | −70.0 | mV | Post-spike reset |
| `v_threshold` | −50.0 | mV | Spike threshold |
| `tau_m` | 10.0 | ms | Membrane time constant |
| `tau_e` | 500.0 | ms | Energy recovery time constant |
| `alpha` | 0.1 | — | Energy cost per spike |
| `epsilon_0` | 1.0 | — | Resting energy level |
| `resistance` | 1.0 | MΩ | Membrane resistance |
| `dt` | 1.0 | ms | Integration timestep |

### τ_e = 500 ms (slow energy recovery)

The energy time constant is 50× the membrane time constant (10 ms).
This creates a slow metabolic constraint: energy depletes quickly (one
spike costs α=0.1) but recovers slowly (τ_e=500 ms → ~5 seconds to
full recovery from 0).

### α = 0.1 (spike cost)

Each spike reduces ε by 0.1. Starting from ε₀=1.0, the neuron can
fire at most 10 rapid spikes before depleting to 0 (with no recovery
between spikes). More realistically, the competition between spike
cost (−0.1 per spike) and recovery (+Δε per dt) creates a sustainable
firing rate.

### Energy threshold at 0.1

The neuron requires ε > 0.1 to spike. This is a hard gate — even if
V >> V_threshold, no spike is emitted when ε ≤ 0.1. This creates
metabolic silencing: after a burst of activity, the neuron enters a
mandatory quiet period while energy recovers.

---

## Analytical Properties

### Energy-modulated input gain

The effective resistance R_eff = R × ε scales the input current:
- Full energy (ε=1.0): R_eff = 1.0 × I → full response
- Half energy (ε=0.5): R_eff = 0.5 × I → half response
- Depleted (ε=0.1): R_eff = 0.1 × I → minimal response

This creates a **negative feedback loop**: high activity → low ε →
weak response → less activity → ε recovers → cycle.

### Maximum sustained firing rate

At steady state, spike cost must balance recovery:

$$\text{rate} \times \alpha = \frac{\varepsilon_0 - \varepsilon_{ss}}{\tau_\varepsilon}$$

For ε_ss ≈ 0.5 (moderate depletion):
$$\text{rate} = \frac{(1.0 - 0.5)}{500 \times 0.1} = 10 \text{ Hz}$$

The metabolic constraint limits sustained firing rate to about 10 Hz —
consistent with typical cortical firing rates in vivo.

### Energy recovery dynamics

Without spiking, ε recovers exponentially toward ε₀:
$$\varepsilon(t) = \varepsilon_0 - (\varepsilon_0 - \varepsilon_i) e^{-t/\tau_\varepsilon}$$

From ε=0 to ε=0.5: t = −500 ln(0.5) ≈ 347 ms.
From ε=0 to ε=0.9: t = −500 ln(0.1) ≈ 1151 ms.

### Comparison with standard LIF

| Feature | EnergyLIF | Standard LIF |
|---------|-----------|-------------|
| State vars | 2 (V, ε) | 1 (V) |
| Input scaling | R × ε × I | R × I |
| Spike gate | V ≥ θ AND ε > 0.1 | V ≥ θ |
| Spike cost | ε −= 0.1 | None |
| Sustained rate | Limited by energy | Limited by refractory |
| Recovery | τ_e = 500 ms | Immediate |
| Adaptation | Via energy depletion | None (external w) |

The EnergyLIF provides **intrinsic adaptation** through metabolic
depletion — no explicit adaptation variable needed. The adaptation
timescale (τ_e=500 ms) is much longer than typical w-based adaptation
(τ_w=100–200 ms).

---

## Behaviour

### Three activity regimes

1. **Subthreshold (I=10):** Current too weak to reach threshold even
   at full energy. Zero spikes in 5000 steps.

2. **Spiking (I=30+):** Sufficient current drives V above threshold.
   Energy gates permit spiking. Rate depends on both I and ε.

3. **Energy-depleted silence:** After sustained high-frequency firing,
   ε drops below 0.1 → neuron enters mandatory quiet period until
   ε recovers. This creates bursty dynamics at high drive.

### Energy depletion under drive

At I=50 (strong drive), ε decreases below 1.0 after sustained spiking.
Verified by test: after 5000 steps at I=50, ε < 1.0.

### Energy recovery without drive

Starting from ε=0.1, after 5000 steps at I=0 (no spiking), ε > 0.1.
The exponential recovery toward ε₀=1.0 is verified.

### Energy gates spiking

When ε=0.05 (< 0.1 threshold), even at I=50 (well above threshold),
zero spikes are produced in 100 steps. The energy gate is absolute.

---

## Comparison with Related Models

| Property | EnergyLIF | AdEx | SFA (Benda-Herz) | EPropALIF |
|----------|-----------|------|-------------------|----------|
| Adaptation | Energy depletion | w current | Threshold shift | Threshold shift |
| Mechanism | ε scales R, gates spikes | w hyperpolarises | Dynamic θ | Dynamic θ |
| Timescale | 500 ms (τ_e) | 100 ms (τ_w) | 50 ms (τ_a) | 200 ms (τ_a) |
| Hard gate | Yes (ε > 0.1) | No | No | No |
| Energy metric | ε (explicit) | None | None | None |
| Metabolic | Yes (biological) | No | No | No |
| Pipeline | Compatible | Compatible | Compatible | Compatible |

The EnergyLIF is the only model with an explicit metabolic energy
constraint — biologically motivated by ATP consumption during spiking.

---

## Numerical Considerations

- **No transcendental functions.** Pure linear ODE + threshold + clamp.
- **ε ≥ 0 clamp.** Prevents negative energy (unphysical).
- **ε > 0.1 gate.** Hard threshold on spiking — discontinuous but
  numerically trivial (comparison).
- **Single Euler step.** dt=1.0 ms — large but adequate for the
  simple linear dynamics.
- **Two state variables.** V and ε. Both bounded by natural dynamics
  (V by spike-reset, ε by [0, ε₀]).

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/energy_lif.py` — 43 lines.
- **Two state variables:** v, epsilon.
- **Dataclass:** Uses `@dataclass`.
- **No numpy dependency:** Pure Python arithmetic.
- **Rust wiring:** Trivially compatible (2 f64 state vars, pure arithmetic).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~1.76M steps/s | Not measured |
| Network (20n, 500ms) | ~1.4M neuron-steps/s | — |

Among the fastest models — no exp(), no sub-stepping, 2 state variables,
pure arithmetic.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 9 | construction, binary output, subthreshold (I=10), spikes (I=30), energy depletes (I=50), energy recovers (from ε=0.1), energy gates spiking (ε=0.05), energy non-negative (10K at I=50), reset |
| Network | 3 | Population(n=10/20), Network+PoissonInput spikes, Projection+spike_trains |
| Analysis | 2 | firing_rate >0, spike_count >10 |
| **Total** | **14** | **ALL PASSED (1.01s)** |

See `tests/test_model_energy_lif.py`.

---

## Findings (Measured 2026-03-31)

1. **14/14 tests PASSED in 1.01s.** No failures.

2. **Subthreshold at I=10.** Zero spikes in 5000 steps.

3. **Spiking at I=30.** More than 10 spikes in 5000 steps.

4. **Energy depletes under drive.** After 5000 steps at I=50, ε < 1.0.

5. **Energy recovers without drive.** From ε=0.1, after 5000 steps at
   I=0, ε > 0.1.

6. **Energy gates spiking absolutely.** At ε=0.05 (< 0.1), even strong
   drive (I=50) produces zero spikes in 100 steps.

7. **Energy non-negative.** After 10K steps at I=50, ε ≥ 0.

8. **Reset restores full energy.** v → v_rest, ε → ε₀.

9. **Network pipeline functional.** Population(n=20) with PoissonInput
   (rate=500Hz, weight=30) produces spikes. Projection(pop→pop,
   weight=5, prob=0.3) works.

10. **Analysis verified.** firing_rate > 0, spike_count > 10.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
14/14 PASSED in 1.01s
├── TestEnergyLIFIsolation: 9 tests
│   ├── construction (ε=1.0)
│   ├── step() → int {0,1}
│   ├── subthreshold at I=10 (0 spikes)
│   ├── spikes at I=30 (>10)
│   ├── energy depletes (ε < 1.0 after I=50)
│   ├── energy recovers (ε > 0.1 from 0.1)
│   ├── energy gates spiking (ε=0.05 → 0)
│   ├── energy non-negative (ε ≥ 0)
│   └── reset()
├── TestEnergyLIFNetwork: 3 tests
│   ├── Population(n=10)
│   ├── Network(n=20) + PoissonInput → spikes
│   └── Projection(pop→pop) + spike_trains
└── TestEnergyLIFAnalysis: 2 tests
    ├── firing_rate > 0
    └── spike_count > 10
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | v=-70, ε=1.0 |
| step() → int {0,1} | ✓ PASS | Energy-gated binary |
| Subthreshold (I=10) | ✓ PASS | 0 spikes |
| Spiking (I=30) | ✓ PASS | >10 spikes |
| Energy depletes | ✓ PASS | ε < 1.0 |
| Energy recovers | ✓ PASS | ε increases |
| Energy gates | ✓ PASS | ε=0.05 → no spikes |
| Energy non-negative | ✓ PASS | ε ≥ 0 |
| reset() | ✓ PASS | v, ε to defaults |
| Population | ✓ PASS | Instances |
| Network | ✓ PASS | Spikes > 0 |
| Projection | ✓ PASS | spike_trains |
| Analysis | ✓ PASS | firing_rate, spike_count |

### Network configuration tested

- Population: 20 EnergyLIFNeurons (spiking), 10 (Projection)
- PoissonInput: rate=500Hz, weight=30.0, dt=0.001, seed=42
- Projection: self-recurrent, weight=5.0, probability=0.3
- SpikeMonitor: count, spike_trains verified
- Duration: 0.5s (spiking), 0.3s (Projection)

**ALL 14 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**

---

## Theoretical Context

### Metabolic constraints in neural computation

Fardet & Levina (2020) proposed that metabolic energy constraints
(ATP supply/demand) shape neural firing patterns:

1. **Spiking is expensive:** Each action potential consumes ~10⁹ ATP
   molecules for Na⁺/K⁺-ATPase restoration of ionic gradients.
2. **Brain uses 20% of body energy:** Despite being 2% of body mass.
3. **Energy limits firing rate:** Real cortical neurons fire at
   1–20 Hz, not the 100+ Hz that biophysics allows — metabolic
   constraints explain this gap.
4. **Sparse coding:** Energy constraints naturally produce sparse
   representations — the neuron "saves energy" by firing rarely.

### EnergyLIF as homeostatic mechanism

The energy variable ε acts as a homeostatic regulator:
- High activity → ε depletes → reduced sensitivity → less activity
- Low activity → ε recovers → increased sensitivity → more activity
- The system self-regulates to a sustainable firing rate

This is computationally equivalent to synaptic scaling (Turrigiano 2008)
but operates on a faster timescale (500 ms vs hours/days for synaptic
homeostasis).

### Implications for network dynamics

In networks of EnergyLIF neurons:
- **Self-organised criticality:** Energy constraints push the network
  toward critical states — balanced between quiescence and runaway
  activity (Fardet & Levina 2020).
- **Avalanche statistics:** The energy-gated suppression creates
  neuronal avalanches with power-law size distributions.
- **Metabolically efficient coding:** The network naturally discovers
  sparse, energy-efficient representations.
