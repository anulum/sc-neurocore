# SRM0Neuron

**Module:** `sc_neurocore.neurons.models.srm0`
**Reference:** Gerstner & Kistler, Spiking Neuron Models, Cambridge University Press, 2002, Ch. 4
**Family:** Spike Response Model (kernel-based, zeroth order)
**State variables:** `v` (membrane potential), `_eta` (refractory kernel), `_t` (internal clock)

---

## Equations

### General SRM formulation (continuous)

$$V(t) = \eta(t - \hat{t}) + \int_0^\infty \kappa(s) \, I(t - s) \, ds$$

where $\hat{t}$ is the time of the last spike, $\eta$ is the refractory
(afterhyperpolarisation) kernel, and $\kappa$ is the postsynaptic
integration kernel.

### SRM0 simplification

The zeroth-order approximation replaces the full convolution with
exponential kernels and a single-step update rule:

**Refractory kernel decay:**
$$\eta_{t+1} = \eta_t \cdot \exp(-dt / \tau_\eta)$$

**Membrane integration:**
$$dV = \frac{R \cdot I - (V - (V_{rest} + \eta))}{\tau_m} \cdot dt$$
$$V_{t+1} = V_t + dV$$

**Spike and reset:**
$$V \geq V_{threshold}: \quad V \leftarrow V_{rest}, \quad \eta \leftarrow -\eta_{reset}$$

### Key insight

The refractory kernel η enters as an offset to the effective resting
potential: $V_{rest,eff} = V_{rest} + \eta$. After a spike, η is set to
$-\eta_{reset}$ (negative), which pulls the effective rest downward.
As η decays toward 0 with time constant $\tau_\eta$, the effective rest
returns to $V_{rest}$, gradually lifting the refractory suppression.

### Implementation

```python
def step(self, current: float) -> int:
    self._eta *= np.exp(-self.dt / self.tau_eta)
    effective_rest = self.v_rest + self._eta
    dv = (self.resistance * current - (self.v - effective_rest)) * self.dt / self.tau_m
    self.v += dv
    self._t += self.dt
    if self.v >= self.v_threshold:
        self.v = self.v_rest
        self._eta = -self.eta_reset
        self._last_spike_time = self._t
        return 1
    return 0
```

Forward Euler with exponential refractory decay. Single step per call.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | 0.0 | a.u. | Membrane potential (initial) |
| `v_rest` | 0.0 | a.u. | Resting potential |
| `v_threshold` | 1.0 | a.u. | Spike threshold |
| `tau_m` | 20.0 | ms | Membrane time constant |
| `tau_eta` | 50.0 | ms | Refractory kernel time constant |
| `eta_reset` | 5.0 | a.u. | Refractory kernel amplitude (negative on spike) |
| `resistance` | 1.0 | — | Input resistance (gain on current) |
| `dt` | 1.0 | ms | Integration timestep |

### Time constant hierarchy

$$\tau_m (20) < \tau_\eta (50)$$

The refractory kernel decays 2.5× slower than the membrane. This means
that after a spike, the neuron recovers its membrane potential (τ_m = 20 ms)
much faster than the refractory suppression lifts (τ_eta = 50 ms). The
refractory period is therefore long-lasting but graded — not a hard
all-or-nothing block.

---

## Analytical Properties

### Refractory kernel dynamics

After spike at t=0:
$$\eta(t) = -\eta_{reset} \cdot \exp(-t / \tau_\eta) = -5.0 \cdot \exp(-t/50)$$

The effective rest potential after spike:
$$V_{rest,eff}(t) = V_{rest} + \eta(t) = 0.0 - 5.0 \cdot \exp(-t/50)$$

| Time after spike | η | V_rest,eff | Recovery % |
|-----------------|-----|-----------|-----------|
| 0 ms | −5.0 | −5.0 | 0% |
| 10 ms | −4.09 | −4.09 | 18% |
| 25 ms | −3.03 | −3.03 | 39% |
| 50 ms (1 τ_η) | −1.84 | −1.84 | 63% |
| 100 ms (2 τ_η) | −0.68 | −0.68 | 86% |
| 150 ms (3 τ_η) | −0.25 | −0.25 | 95% |

Full recovery (~95%) requires ~3 τ_eta = 150 ms.

### Effective threshold elevation

The refractory kernel effectively raises the threshold. For the neuron
to fire again, V must reach V_threshold = 1.0 starting from
V_rest,eff = V_rest + η < V_rest. The effective gap:

$$V_{threshold} - V_{rest,eff} = V_{threshold} - V_{rest} - \eta = 1.0 - \eta$$

Immediately after spike: gap = 1.0 − (−5.0) = 6.0 (6× the resting gap).
This creates spike-frequency adaptation: each spike makes the next one
harder to produce.

### Membrane steady state (no spike, eta = 0)

$$V_{ss} = V_{rest} + R \cdot I = I \quad \text{(with defaults)}$$

For spiking: V_ss ≥ V_threshold requires I ≥ 1.0.

### ISI adaptation

Each spike resets η to −5.0, which decays with τ_eta = 50 ms. The first
ISI is determined by the time to reach threshold from V_rest with η ≈ 0.
Subsequent ISIs are longer because η is still negative from the previous
spike — the effective threshold is elevated.

This produces spike-frequency adaptation similar to the AdEx model's w
variable, but through a kernel mechanism rather than a differential equation
for the adaptation variable.

### Comparison: SRM0 vs LIF

The SRM0 is equivalent to a LIF with:
- An adaptive threshold that decays exponentially after each spike
- No separate adaptation ODE (η is integrated into the effective rest)

Formally, the SRM0 with V_rest,eff = V_rest + η is identical to a LIF
with dynamic threshold V_threshold,eff = V_threshold − η. The two
formulations are mathematically equivalent.

---

## Behaviour

### Graded refractoriness

Unlike hard refractory periods (TrueNorth: refrac_steps = 2), the SRM0
has a **graded refractory period:**
- Immediately after spike: very difficult to fire (gap = 6.0)
- 50 ms later: moderately difficult (gap ≈ 2.84)
- 150 ms later: nearly normal (gap ≈ 1.25)

Strong enough input can overcome the refractory suppression at any time
— there is no absolute refractory period, only a relative one.

### Adaptation produces ISI lengthening

With constant input:
- First ISI: short (η ≈ 0)
- Second ISI: longer (η still negative from first spike)
- nth ISI: converges to steady state where eta decays just enough
  between spikes to allow the next one

This is spike-frequency adaptation — the hallmark of the SRM0 model.

### Strong drive overrides refractoriness

With very high input (I ≫ V_threshold):
- Even with η = −5.0, V can reach threshold quickly
- ISI becomes short (approaching the τ_m-limited minimum)
- Adaptation effect is proportionally weaker

### Subthreshold decay

Without input (I = 0), V decays to V_rest,eff = V_rest + η with time
constant τ_m. If η < 0, V decays below V_rest temporarily.

---

## Historical Context

### Spike Response Model family

The SRM was introduced by Gerstner (1995) as an alternative formulation
to the integrate-and-fire framework. Key advantages:

1. **Kernel-based:** The entire neural response is described by kernels
   (η for refractory, κ for integration) — no differential equations needed.

2. **Multiple timescales:** The separate τ_m and τ_eta naturally create
   adaptation without adding an explicit adaptation variable.

3. **Connection to renewal theory:** The SRM framework connects to
   point-process theory, enabling analytical calculation of ISI
   distributions, hazard rates, and population firing rates.

### SRM hierarchy

| Model | Description | Equivalent to |
|-------|-------------|---------------|
| SRM0 | Last spike only | LIF + adaptive threshold |
| SRM1 | Last spike + full convolution | LIF + full synaptic filtering |
| SRM∞ | All past spikes | No simple equivalent |

The SRM0 (this model) uses only the last spike time — the zeroth-order
approximation. Higher-order SRMs track multiple past spikes for more
accurate refractoriness modelling.

---

## Pipeline Compatibility

### Fully compatible

`step(current) → int` — standard spiking interface. Population, Network,
SpikeMonitor, PoissonInput, Projection all work without limitations.

### Internal state access

`get_state() → dict` provides {"v", "eta", "t"} for debugging and
analysis. This is not part of the standard interface but is useful for
verifying refractory dynamics.

---

## Comparison with Related Models

| Property | SRM0 | LIF | AdEx | EscapeRate |
|----------|------|-----|------|------------|
| State vars | 1 (V) + η kernel | 1 (V) | 2 (V, w) | 1 (V) |
| Refractoriness | Graded (η kernel) | None | Adaptation (w) | Stochastic |
| Adaptation | Via η | None | Via w += b | None |
| ISI lengthening | Yes (η) | No | Yes (w) | No |
| Stochastic | No | No | No | Yes |
| Framework | Kernel-based | ODE-based | ODE-based | Point-process |
| Pipeline | Compatible | Compatible | Compatible | Compatible |

The SRM0 achieves similar adaptation dynamics to AdEx but through a
single kernel variable (η) rather than a separate ODE (dw/dt).

---

## Numerical Considerations

- **Single Euler step:** dt=1.0ms. Adequate for τ_m=20ms and τ_eta=50ms.
- **One exp() per step:** η decay uses np.exp(-dt/tau_eta). This is the
  only transcendental function — the rest is linear.
- **Eta decay stability:** The exponential multiplier exp(-dt/τ_eta) is
  always in (0, 1) for positive dt and τ_eta → η always decays toward 0.
  No instability possible.
- **Internal clock:** _t tracks absolute time for spike timing records.
  Uses float addition — may accumulate rounding error over very long runs.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/srm0.py` — 71 lines.
- **State:** v (membrane), _eta (refractory kernel), _t (clock),
  _last_spike_time (for diagnostics).
- **__post_init__:** Initialises private state after dataclass construction.
- **get_state():** Debug accessor for {v, eta, t}.
- **Rust wiring:** Compatible with `step(f64) → i32`. Two f64 state vars
  (v, eta) + clock.

---

## Infrastructure Pipeline

```
SRM0Neuron
├── step(current) → int {0, 1}
├── 1 Euler step + 1 exp() per call (dt=1.0ms)
├── Population, Network, SpikeMonitor: compatible
│   PoissonInput(weight=2, rate=500Hz)
├── Projection: tested src→tgt wiring
├── Analysis: spike_count, isi, firing_rate verified
└── Rust: compatible (2 f64 state vars)
```

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~389K steps/s | Not measured |
| Network (10 neurons, 1s) | ~35K neuron-steps/s | — |

Fast model — single Euler step with 1 exp() call. The exp() for η decay
is the dominant cost per step.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary return, state evolution, finite 50k, reset |
| Refractory kernel | 5 | η set to -eta_reset on spike, η decays exponentially, effective rest lowered, recovery timeline, graded refractoriness |
| Adaptation | 3 | ISI lengthens, first ISI shorter than later, constant input convergence |
| f–I curve | 3 | subthreshold silent, monotonic, rate increases with current |
| Parameters | 3 | dt stability, τ_eta sweep, eta_reset sweep |
| Pipeline | 4 | Population, Network+drive, Projection wiring, analysis (spike_count, isi, firing_rate) |
| **Total** | **23** | |

See `tests/test_model_srm0.py`. No bugs found.

---

## Findings

1. **Refractory kernel η set correctly:** On spike, η = −eta_reset = −5.0.
   After 50 ms (1 τ_eta), η ≈ −1.84 (63% recovery).

2. **ISI adaptation confirmed:** First ISI shorter than later ISIs at
   constant input. The η accumulation from previous spikes creates
   spike-frequency adaptation.

3. **Graded refractoriness:** No absolute refractory period — strong
   enough input can trigger a spike immediately after the previous one,
   but with higher effective threshold.

4. **Subthreshold silent:** I < V_threshold/R produces zero spikes (with
   η = 0). The threshold is sharp.

5. **Monotonic f-I:** Higher current → more spikes across the tested range.

6. **η decay verified analytically:** Measured η values match
   -eta_reset × exp(-t/tau_eta) within 1e-10.

7. **get_state() provides debug access:** Returns {v, eta, t} dictionary
   for kernel inspection.

8. **reset() clears all state:** V, η, t, _last_spike_time all return to
   initial values.

9. **Network pipeline fully functional:** Population(n=10) + PoissonInput
   + SpikeMonitor produces spikes. Projection wiring works.

10. **Kernel-based equivalent to adaptive LIF:** The SRM0 produces the
    same adaptation dynamics as an AdEx with b=eta_reset and τ_w=τ_eta,
    but through a simpler single-kernel mechanism.

---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~86K steps/s |
| Spikes (10K steps, I=5.0) | 371 |
| State stability (20K steps) | PASS |
| Rust parity | N/A (no Rust binding for SRM0Neuron) |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`SRM0Neuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → int {0, 1}
Returns native Python int.
**Status: PASS**

### 3. Spiking behaviour
371 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(SRM0Neuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**N/A** — no Rust binding exists for SRM0Neuron.

---

## Findings (measured 2026-04-04)

1. Throughput: ~86K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. No Rust binding — candidate for future Rust port
4. Numerical stability confirmed over 20K steps

---

## Theoretical Context

### The Spike Response Model family

The Spike Response Model (SRM) was introduced by Wulfram Gerstner as a
mathematically tractable alternative to Hodgkin-Huxley-type conductance
models. Instead of modelling ionic currents through differential
equations, the SRM expresses membrane potential as a superposition of
response kernels triggered by input spikes and the neuron's own output
spikes.

The general SRM expresses voltage as:

$$V_i(t) = \eta(t - \hat{t}_i) + \sum_j w_{ij} \sum_f \kappa(t - t_j^f) + \int \kappa_0(s)\, I_{ext}(t-s)\, ds$$

where the first term captures the refractory response to the neuron's
own last spike, the second sums postsynaptic responses from all
presynaptic spikes, and the third integrates external current.

### SRM0: the zeroth-order simplification

SRM0 is the simplest member of the SRM hierarchy. It assumes that the
kernels depend only on the time since the last spike of the postsynaptic
neuron (hence "zeroth order" — no dependence on earlier spikes). This
makes the model memoryless apart from the single most recent spike time.

The SRM0 is mathematically equivalent to a leaky integrate-and-fire
neuron with spike-frequency adaptation, but it makes the kernel structure
explicit. This is useful for:

- Analytical tractability (threshold crossing is a 1D problem)
- Network-level mean-field analysis
- Population density methods (Gerstner & Kistler, 2002, Ch. 6)
- Escape-rate noise models (soft threshold → GIFPopulation)

### Relation to other SC-NeuroCore models

| Model | Relation to SRM0 |
|-------|-----------------|
| LIF (Lapicque) | SRM0 without refractory kernel (η=0) |
| AdEx | SRM0 + exponential spike initiation + w adaptation |
| EscapeRate | SRM0 + stochastic threshold (exp hazard) |
| GIFPopulation | SRM0 + escape rate at population level |
| SpikeResponseNeuron | Pure kernel-based SRM variant (Rust, see below) |

### Excitability type

SRM0 with the implemented parameters exhibits **Type-I excitability**:
the f–I curve rises continuously from zero at rheobase, with no minimum
firing frequency. This is because the effective threshold returns
smoothly to V_threshold as η decays, allowing arbitrarily slow firing.

---

## Variant Implementations

SC-NeuroCore provides **two distinct SRM variants**, each implementing a
valid but mathematically different formulation of the spike response
model:

### Python: SRM0Neuron (ODE-based with η offset)

- **Module:** `sc_neurocore.neurons.models.srm0`
- **Approach:** Leaky integrator with refractory kernel as resting-potential offset
- **Voltage:** Accumulated via forward Euler (has memory between steps)
- **Key equation:** $dV/dt = (R \cdot I - (V - V_{rest} - \eta)) / \tau_m$
- **Parameters:** τ_m = 20 ms, τ_η = 50 ms, η_reset = 5.0, R = 1.0

### Rust: SpikeResponseNeuron (pure kernel superposition)

- **Module:** `sc_neurocore_engine::neurons::SpikeResponseNeuron`
- **Approach:** Voltage recomputed each step from kernels (memoryless)
- **Voltage:** $V = \eta(t_{since}) + \kappa(I)$ — fresh each step
- **Key equations:**
  - $\eta(t) = \eta_{reset} \cdot \exp(-t / \tau_\eta)$
  - $\kappa(I) = I \cdot (1 - \exp(-dt / \tau_\kappa))$
- **Parameters:** τ_η = 10 ms, τ_κ = 5 ms, η_reset = −5.0

### Why two variants?

Both are valid SRM formulations from the literature:

1. The **Python SRM0** follows Gerstner & Kistler (2002, Ch. 4.1) — an
   ODE approximation where the refractory kernel modifies the effective
   resting potential within a standard leaky integrator. This is closer
   to how experimentalists think about adaptation.

2. The **Rust SpikeResponse** follows Gerstner (1995) — the pure
   kernel-based formulation where voltage is a superposition of
   exponential response functions. This is closer to the original
   mathematical formalism and avoids numerical drift.

### Key differences

| Property | Python SRM0 | Rust SpikeResponse |
|----------|------------|-------------------|
| Voltage dynamics | ODE accumulation | Kernel superposition |
| Memory in V | Yes (leaky decay) | No (recomputed) |
| τ_m (membrane) | 20.0 ms | N/A (replaced by τ_κ) |
| τ_η (refractory) | 50.0 ms | 10.0 ms |
| τ_κ (input) | N/A | 5.0 ms |
| η_reset | 5.0 (negated in code) | −5.0 (pre-negated) |
| Resistance R | 1.0 | N/A |
| Sub-steps | 1 | 1 |
| PyO3 binding | No | Yes |
| NetworkRunner | No | Yes |
| Parity | N/A — different models | N/A — different models |

---

## Usage Examples

### Example 1: Basic Python — SRM0 with constant current

```python
from sc_neurocore.neurons.models.srm0 import SRM0Neuron

neuron = SRM0Neuron()

# Simulate 1000 ms at I = 5.0
spike_times = []
for t in range(1000):
    spike = neuron.step(5.0)
    if spike:
        spike_times.append(t)

print(f"Fired {len(spike_times)} spikes in 1000 ms")
print(f"Mean ISI: {sum(b-a for a,b in zip(spike_times, spike_times[1:]))/(len(spike_times)-1):.1f} ms")
```

### Example 2: Advanced Python — refractory adaptation dynamics

```python
from sc_neurocore.neurons.models.srm0 import SRM0Neuron
import numpy as np

neuron = SRM0Neuron(tau_eta=100.0, eta_reset=10.0)

# Inject brief pulse, observe η recovery
voltages, etas = [], []
for t in range(500):
    current = 20.0 if t < 5 else 0.0
    neuron.step(current)
    state = neuron.get_state()
    voltages.append(state["v"])
    etas.append(state["eta"])

# After spike, η decays exponentially with τ_η = 100 ms
# Effective threshold recovery: V_rest + η → V_rest as η → 0
print(f"η at t=10: {etas[10]:.3f}")
print(f"η at t=100: {etas[100]:.3f}")
print(f"η at t=300: {etas[300]:.3f}")
```

### Example 3: PyO3 Rust — SpikeResponseNeuron (kernel variant)

```rust
use sc_neurocore_engine::neurons::SpikeResponseNeuron;

let mut neuron = SpikeResponseNeuron::new();

// 10,000 steps at I = 2.0
let mut spikes = 0;
for _ in 0..10_000 {
    spikes += neuron.step(2.0);
}
println!("Kernel SRM: {spikes} spikes in 10 s");

// Voltage is recomputed each step — no accumulation
println!("V = {:.4} (superposition of η + κ)", neuron.v);

// Reset clears time_since_spike
neuron.reset();
assert!(neuron.time_since_spike > 100.0);  // far in the past
```

---

## Technical Reference

### Methods — Python SRM0Neuron

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `step(current: float) → int` | 0 or 1 | Advance 1 ms, return spike |
| `reset` | `reset() → None` | — | Restore v, η, t to initial values |
| `get_state` | `get_state() → dict` | dict | Returns {v, eta, t} |

### Methods — Rust SpikeResponseNeuron

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `step(f64) → i32` | 0 or 1 | Advance 1 ms, return spike |
| `reset` | `reset()` | — | Restore v, time_since_spike |

### Supported operations

| Operation | Python SRM0 | Rust SpikeResponse |
|-----------|-------------|-------------------|
| Population | Yes | Yes |
| Projection | Yes | Yes |
| NetworkRunner | No | Yes |
| SpikeMonitor | Yes | Yes |
| PoissonInput | Yes | Yes |
| PyO3 bridge | No | Yes |
| get_state() | Yes | Yes (via PyO3) |

---

## Performance Benchmarks

### Python SRM0Neuron

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-04.

| Metric | Value |
|--------|------:|
| Isolation throughput | ~86 000 steps/s |
| Per step | ~11.6 µs |
| 10K steps, I=5.0 | 371 spikes |
| State stability (20K) | PASS |

### Rust SpikeResponseNeuron

| Metric | Value |
|--------|------:|
| Python throughput (via PyO3) | ~160 000 steps/s |
| Per step (Python) | ~6.3 µs |
| Criterion benchmark | Not yet measured |

### Computational cost

- Python SRM0: 1 exp() per step (η decay) + 1 division + arithmetic = minimal
- Rust SpikeResponse: 2 exp() per step (η kernel + κ kernel) + arithmetic = minimal
- Both are among the fastest spiking models in the library

---

## Citations

1. Gerstner, W. & Kistler, W. M. (2002). *Spiking Neuron Models: Single
   Neurons, Populations, Plasticity*. Cambridge University Press, Ch. 4.
   DOI: [10.1017/CBO9780511815706](https://doi.org/10.1017/CBO9780511815706)

2. Gerstner, W. (1995). Time structure of the activity in neural network
   models. *Physical Review E*, 51(1), 738–758.
   DOI: [10.1103/PhysRevE.51.738](https://doi.org/10.1103/PhysRevE.51.738)

3. Gerstner, W., Kistler, W. M., Naud, R. & Paninski, L. (2014).
   *Neuronal Dynamics: From Single Neurons to Networks and Models of
   Cognition*. Cambridge University Press, Ch. 6.
   DOI: [10.1017/CBO9781107447615](https://doi.org/10.1017/CBO9781107447615)

4. Jolivet, R., Rauch, A., Lüscher, H.-R. & Gerstner, W. (2006).
   Predicting spike timing of neocortical pyramidal neurons by simple
   threshold models. *Journal of Computational Neuroscience*, 21(1),
   35–49.
   DOI: [10.1007/s10827-006-7074-5](https://doi.org/10.1007/s10827-006-7074-5)

5. Gerstner, W. & van Hemmen, J. L. (1992). Associative memory in a
   network of 'spiking' neurons. *Network: Computation in Neural
   Systems*, 3(2), 139–164.
   DOI: [10.1088/0954-898X/3/2/004](https://doi.org/10.1088/0954-898X/3/2/004)

6. Pillow, J. W., Shlens, J., Paninski, L., Sher, A., Litke, A. M.,
   Chichilnisky, E. J. & Simoncelli, E. P. (2008). Spatio-temporal
   correlations and visual signalling in a complete neuronal population.
   *Nature*, 454(7207), 995–999.
   DOI: [10.1038/nature07140](https://doi.org/10.1038/nature07140)
