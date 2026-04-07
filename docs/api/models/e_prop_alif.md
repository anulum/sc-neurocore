# EPropALIFNeuron

**Module:** `sc_neurocore.neurons.models.e_prop_alif`
**Reference:** Bellec et al., Nat. Commun. 11(1), 2020 (e-prop)
**Family:** Adaptive LIF with eligibility traces (biologically plausible learning)
**State variables:** `v` (membrane potential), `a` (adaptive threshold), `e_trace` (eligibility trace)

---

## Equations

### Membrane potential (discrete exponential decay)

$$V_{t+1} = \alpha_m \cdot V_t + I_t$$

where $\alpha_m = \exp(-dt / \tau_m)$.

No leak subtraction — the exponential decay factor α_m directly scales
the previous voltage. This is the discrete-time equivalent of the
continuous LIF: $\tau_m \, dV/dt = -V + R \cdot I$.

### Adaptive threshold

$$\theta_t = \theta_{base} + \beta \cdot a_t$$

On spike: $a_{t+1} = \alpha_a \cdot a_t + 1$
No spike: $a_{t+1} = \alpha_a \cdot a_t$

where $\alpha_a = \exp(-dt / \tau_a)$.

Each spike increments $a$ by 1. Between spikes, $a$ decays exponentially
with $\tau_a = 200$ ms. The effective threshold $\theta_t$ rises after
each spike, producing spike-frequency adaptation.

### Eligibility trace (Bellec 2020, Eq. 4)

$$\psi_t = 0.3 \cdot \max(0,\; 1 - |V_t - \theta_t|)$$

$$e_{t+1} = \alpha_a \cdot e_t + \psi_t$$

The pseudo-derivative ψ is non-zero only when V is within 1 unit of the
threshold — a triangular kernel centred on the threshold. The eligibility
trace e accumulates these near-threshold events with slow decay (τ_a).

### Spike and reset

$$V_t \geq \theta_t: \quad V \leftarrow V_{reset}, \quad a \leftarrow \alpha_a \cdot a + 1$$

### Three-factor learning rule (not implemented in step, but enabled by e_trace)

$$\Delta w = \eta \cdot L_t \cdot e_t$$

where $L_t$ is the learning signal (error/reward), $e_t$ is the eligibility
trace, and $\eta$ is the learning rate. This is the e-prop rule: synaptic
weight changes are proportional to the product of a global error signal and
a local eligibility trace — no backpropagation through time (BPTT) needed.

### Implementation

```python
def step(self, current: float) -> int:
    self.v = self.alpha_m * self.v + current
    threshold = self.v_threshold_base + self.beta * self.a
    psi = max(0.0, 1.0 - abs(self.v - threshold)) * 0.3
    self.e_trace = self.alpha_a * self.e_trace + psi
    if self.v >= threshold:
        self.v = self.v_reset
        self.a = self.alpha_a * self.a + 1.0
        return 1
    self.a *= self.alpha_a
    return 0
```

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | 0.0 | a.u. | Membrane potential |
| `a` | 0.0 | — | Adaptive threshold component |
| `e_trace` | 0.0 | — | Eligibility trace |
| `tau_m` | 20.0 | ms | Membrane time constant |
| `tau_a` | 200.0 | ms | Adaptation time constant |
| `v_threshold_base` | 1.0 | a.u. | Base spike threshold |
| `beta` | 0.07 | — | Threshold adaptation coupling |
| `v_reset` | 0.0 | a.u. | Post-spike reset potential |
| `dt` | 1.0 | ms | Integration timestep |
| `alpha_m` | computed | — | exp(−dt/τ_m) = exp(−1/20) ≈ 0.9512 |
| `alpha_a` | computed | — | exp(−dt/τ_a) = exp(−1/200) ≈ 0.9950 |

### Precomputed decay constants

α_m and α_a are computed in `__post_init__` to avoid repeated exp() calls
during step(). This is an optimisation — each step() call uses only
multiplication (no exp()).

### Time constant ratio

$$\tau_a / \tau_m = 200 / 20 = 10$$

The adaptation time constant is 10× the membrane time constant. This means
adaptation persists across many spikes (10 membrane time constants), creating
sustained spike-frequency adaptation.

---

## Analytical Properties

### Adaptive threshold mechanism

After n spikes in quick succession (no decay between):
$$a \approx n, \quad \theta \approx \theta_{base} + \beta \cdot n = 1.0 + 0.07n$$

| Spikes | θ | Increase |
|--------|---|----------|
| 0 | 1.00 | — |
| 5 | 1.35 | 35% |
| 10 | 1.70 | 70% |
| 20 | 2.40 | 140% |

The threshold nearly triples after 20 rapid spikes — strong adaptation.

### Pseudo-derivative ψ

The pseudo-derivative is a **surrogate gradient** for the non-differentiable
spike function:
- ψ = 0 when |V − θ| > 1 (far from threshold: no learning signal)
- ψ = 0.3 when V = θ (at threshold: maximum learning signal)
- Linear interpolation between

This triangular approximation replaces the Dirac delta of the true spike
derivative. The width (1 unit on each side of θ) and amplitude (0.3) are
hyperparameters from Bellec 2020.

### Eligibility trace as filtered pseudo-derivative

$$e_t = \sum_{s=0}^{t} \alpha_a^{t-s} \psi_s$$

The eligibility trace is an exponentially-weighted running sum of
near-threshold events. It tells the learning rule: "this synapse
contributed to recent near-threshold activity." The slow decay (τ_a = 200 ms)
creates a long temporal window for credit assignment.

### Connection to BPTT

E-prop is an online approximation to truncated BPTT:
- BPTT: exact gradients, O(T) memory, biologically implausible
- E-prop: approximate gradients via eligibility traces, O(1) memory,
  biologically plausible (only local information + global error)

The approximation error is bounded by the mixing time of the recurrent
network — for networks with fast dynamics (relative to τ_a), e-prop
converges to near-BPTT performance.

---

## Behaviour

### Spike-frequency adaptation

With constant input, the firing rate decreases over time:
- Early: θ ≈ θ_base = 1.0 → easy to spike
- Later: a accumulates → θ increases → harder to spike
- Steady state: a converges where decay (α_a) balances spike increments

### Eligibility trace dynamics

The eligibility trace tracks near-threshold voltage excursions:
- When V is far from θ: e_trace decays (ψ = 0)
- When V approaches θ: e_trace accumulates (ψ > 0)
- After spike: e_trace continues to decay slowly (τ_a = 200 ms)

This provides the learning rule with a temporally-extended memory of
which synapses recently contributed to near-threshold activity.

### No learning in step()

The step() function computes e_trace but does not apply weight updates.
Learning requires a separate outer loop:
1. Call step() to get spike and update e_trace
2. Compute learning signal L_t (from error/reward)
3. Update weights: Δw = η × L_t × e_trace

This separation allows flexible learning rule implementations.

---

## E-Prop Framework Context

### Bellec et al. 2020 results

The e-prop paper demonstrated that ALIF neurons with eligibility traces
can learn:
- TIMIT speech recognition (word error rate competitive with LSTM)
- Delayed match-to-sample tasks
- Evidence accumulation (similar to Wong-Wang decision tasks)

All without BPTT — using only local eligibility traces and a global
error signal.

### Three types of e-prop

| Type | Learning signal | Biological analogue |
|------|----------------|-------------------|
| e-prop 1 | Broadcasting error | Neuromodulatory signal |
| e-prop 2 | Random feedback | Random feedback alignment |
| e-prop 3 | Symmetric error | Error backpropagation |

The EPropALIFNeuron supports all three types — the eligibility trace
is the same; only the external learning signal differs.

### Why ALIF (not plain LIF)?

Adaptation (via the dynamic threshold) is critical for e-prop performance:
- Creates slow dynamics that encode temporal context
- The slow τ_a provides a long credit assignment window
- Matches the slow timescales of many cognitive tasks (~100 ms)

Without adaptation, e-prop degrades significantly on temporal tasks.

---

## Pipeline Compatibility

### Fully compatible

`step(current) → int` — standard spiking interface. Population, Network,
SpikeMonitor, PoissonInput, Projection all work.

---

## Comparison with Related Models

| Property | EPropALIF | AdEx | SRM0 | LIF |
|----------|----------|------|------|-----|
| State vars | 3 (V, a, e) | 2 (V, w) | 1+η | 1 (V) |
| Adaptation | Dynamic threshold | w current | η kernel | None |
| Eligibility | Yes (e_trace) | No | No | No |
| Learning support | E-prop (3-factor) | None built-in | None | None |
| τ_adaptation | 200 ms | 100 ms | 50 ms (η) | — |
| Discrete decay | α_m, α_a (no exp/step) | Euler (exp clip) | Euler + exp | Various |
| Pipeline | Compatible | Compatible | Compatible | Compatible |

The EPropALIF is the only model in SC-NeuroCore with built-in eligibility
trace support for biologically plausible learning.

---

## Numerical Considerations

- **No exp() per step:** α_m and α_a precomputed in __post_init__.
  step() uses only multiply and compare — no transcendental functions.
- **Stable decay:** α_m, α_a ∈ (0, 1) → V and a always decay. No
  numerical instability possible.
- **ψ bounded:** max(0, 1 − |V − θ|) ∈ [0, 1]. Multiplied by 0.3 →
  ψ ∈ [0, 0.3]. e_trace cannot blow up.
- **Integer-friendly:** The discrete decay formulation (multiply by α)
  is easily quantised for neuromorphic hardware.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/e_prop_alif.py` — 54 lines.
- **Three state variables:** v, a, e_trace.
- **Dataclass + field(init=False):** α_m and α_a are derived parameters.
- **No private methods:** All logic in step().
- **Rust wiring:** Compatible (3 f64 state vars, precomputed α).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~500K steps/s | Not measured |
| Network (10 neurons, 1s) | ~40K neuron-steps/s | — |

Fast model — no exp() per step, no sub-stepping. Only multiplications
and comparisons. Comparable to standard LIF speed.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, alpha precomputed, binary output, state finite, reset |
| Adaptive threshold | 5 | a increments on spike, a decays between spikes, threshold increases with a, ISI lengthens, β=0 no adaptation |
| Eligibility trace | 3 | e_trace accumulates, e_trace decays, pseudo-derivative peaks near threshold |
| F-I curve | 2 | zero input silent, monotonic f-I |
| Parameters | 5 | tau_a controls speed, dt stability [0.5,1.0,2.0] (parametrised), deterministic |
| Performance | 2 | isolation throughput, network throughput |
| Pipeline | 4 | Population, Network spikes, Projection wiring, analysis |
| **Total** | **26** | **ALL PASSED (2.27s)** |

See `tests/test_model_e_prop_alif.py`.

---

## Findings (Measured 2026-03-31)

1. **26/26 tests PASSED in 2.27s.** No failures.

2. **Adaptive threshold rises on spikes:** a variable increments when
   the neuron fires, directly increasing effective threshold θ + β·a.

3. **a decays between spikes.** Without spiking, a decays toward 0
   with time constant τ_a (via α_a multiplication).

4. **Threshold increases with a.** Higher a → higher effective threshold
   → harder to fire. Verified.

5. **ISI adaptation confirmed:** ISI lengthens as a accumulates. First
   ISI shorter than later ISIs at constant input.

6. **β=0 eliminates adaptation:** With β=0, threshold is constant.
   Model reduces to simple exponential-decay LIF.

7. **Eligibility trace accumulates near threshold.** When V is near θ,
   pseudo-derivative ψ > 0 → e_trace increases.

8. **Eligibility trace decays far from threshold.** When |V − θ| > 1,
   ψ = 0 → e_trace decays via α_e.

9. **Pseudo-derivative peaks near threshold.** ψ = 0.3 × max(0, 1 − |V−θ|)
   is maximal at V = θ.

10. **Zero input → silent.** No spikes without external drive.

11. **Monotonic f-I curve.** More input → more spikes.

12. **Precomputed α eliminates exp().** No transcendental functions
    during step() — only multiply and compare.

13. **Network pipeline fully functional.** Population, PoissonInput,
    Projection, SpikeMonitor, analysis all work.

14. **Deterministic.** Bit-exact traces across repeated runs.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
26/26 PASSED in 2.27s
├── TestEPropALIFIsolation: 5 tests
│   ├── defaults (v=0, a=0, e=0, tau_m=20, tau_a=200)
│   ├── alpha precomputed (α_m, α_a = exp(-dt/τ))
│   ├── step() → int {0,1}
│   ├── state finite
│   └── reset() (v→0, a→0, e→0)
├── TestEPropALIFAdaptiveThreshold: 5 tests
│   ├── a increments on spike
│   ├── a decays between spikes
│   ├── threshold increases with a
│   ├── ISI lengthens
│   └── β=0: no adaptation
├── TestEPropALIFEligibilityTrace: 3 tests
│   ├── e_trace accumulates near threshold
│   ├── e_trace decays far from threshold
│   └── pseudo-derivative peaks at V=θ
├── TestEPropALIFFI: 2 tests
│   ├── zero input silent
│   └── monotonic f-I
├── TestEPropALIFParameters: 5 tests
│   ├── tau_a controls adaptation speed
│   ├── dt stability [0.5, 1.0, 2.0]
│   └── deterministic
├── TestEPropALIFPerformance: 2 tests
│   ├── isolation throughput
│   └── network throughput
└── TestEPropALIFPipeline: 4 tests
    ├── Population
    ├── Network + PoissonInput → spikes
    ├── Projection wiring
    └── spike_count + firing_rate analysis
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | v=0, a=0, e_trace=0 |
| step() → int {0,1} | ✓ PASS | Standard binary output |
| Alpha precomputed | ✓ PASS | α_m, α_a in __post_init__ |
| Adaptive threshold | ✓ PASS | a increments, θ rises |
| a decays | ✓ PASS | Between spikes |
| ISI lengthens | ✓ PASS | Adaptation effect |
| β=0 baseline | ✓ PASS | No adaptation |
| Eligibility trace | ✓ PASS | Accumulates and decays |
| ψ peaks at threshold | ✓ PASS | Pseudo-derivative |
| Zero → silent | ✓ PASS | No spikes |
| Monotonic f-I | ✓ PASS | More I → more spikes |
| State finite | ✓ PASS | All 3 vars |
| reset() | ✓ PASS | All to 0 |
| Deterministic | ✓ PASS | Bit-exact |
| Population | ✓ PASS | Instances |
| Network | ✓ PASS | Spikes produced |
| Projection | ✓ PASS | Wiring |
| Analysis | ✓ PASS | spike_count, firing_rate |

**ALL 26 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**

10. **E-prop approximates BPTT:** The eligibility trace mechanism
    provides an online, biologically plausible alternative to
    backpropagation through time for recurrent SNN training.

---

## Theoretical Context

### The learning problem in recurrent SNNs

Training recurrent spiking neural networks is difficult because spikes
are binary, non-differentiable events. Backpropagation through time
(BPTT) applied to SNNs requires unrolling the entire temporal dynamics
and back-propagating gradients through every timestep — this is
computationally expensive, memory-intensive, and biologically
implausible.

Bellec et al. (2020) introduced **e-prop** (eligibility propagation)
as a solution: an online learning rule that combines local eligibility
traces with a global learning signal, approximating BPTT without
requiring backward passes through time.

### Why ALIF for e-prop

The adaptive leaky integrate-and-fire (ALIF) neuron is central to
e-prop because:

1. **Adaptive threshold provides slow memory:** The adaptation variable
   `a` decays with τ_a = 200 ms (10× slower than the membrane τ_m =
   20 ms). This creates a slowly varying state that carries temporal
   information forward — critical for tasks requiring memory over
   hundreds of milliseconds.

2. **Eligibility trace tracks credit assignment:** The pseudo-derivative
   ψ measures how close the neuron is to threshold. Filtering ψ through
   the adaptation time constant produces an eligibility trace that
   records which synapses contributed to recent spikes. When a global
   learning signal (reward, error) arrives, synapses with high
   eligibility are updated — a form of three-factor plasticity.

3. **Bellec et al. demonstrated:** ALIF networks trained with e-prop
   match LSTM performance on TIMIT speech recognition (phoneme error
   rate), delayed match-to-sample, and evidence accumulation tasks —
   all without BPTT.

### Three types of e-prop

The neuron model in SC-NeuroCore computes the eligibility trace but does
not apply weight updates (these are handled by the learning module). The
three e-prop variants differ only in the learning signal:

| Type | Learning signal | Application |
|------|----------------|-------------|
| e-prop symmetric | Random feedback (broadcast) | Unsupervised |
| e-prop random | Random B matrix | Supervised |
| e-prop adaptive | BPTT-derived feedback | Highest accuracy |

All three use the same ALIF neuron with the same eligibility trace
computation. The difference is external to the neuron.

### Relation to other SC-NeuroCore models

| Model | Relation to EPropALIF |
|-------|----------------------|
| LIF (Lapicque) | EPropALIF without adaptation (beta=0, no e_trace) |
| AdEx | Similar adaptation but with exponential spike initiation |
| SFA (Spike Frequency Adaptation) | Simpler adaptation (no eligibility trace) |
| SRM0 | Kernel-based adaptation (different formalism, same effect) |
| SuperSpike | Different surrogate gradient (sigmoid, not triangular) |

---

## Usage Examples

### Example 1: Basic Python — ALIF with constant current

```python
from sc_neurocore.neurons.models.e_prop_alif import EPropALIFNeuron

neuron = EPropALIFNeuron()

# Simulate 1000 ms at I = 0.5
spike_times = []
for t in range(1000):
    spike = neuron.step(0.5)
    if spike:
        spike_times.append(t)

print(f"Fired {len(spike_times)} spikes in 1000 ms")
# ISI should lengthen due to adaptation
if len(spike_times) > 2:
    isis = [b - a for a, b in zip(spike_times, spike_times[1:])]
    print(f"First ISI: {isis[0]} ms, Last ISI: {isis[-1]} ms")
```

### Example 2: Advanced Python — eligibility trace dynamics

```python
from sc_neurocore.neurons.models.e_prop_alif import EPropALIFNeuron
import numpy as np

neuron = EPropALIFNeuron(tau_a=100.0, beta=0.1)

# Track eligibility trace over time
e_traces, voltages, thresholds = [], [], []
for t in range(500):
    current = 0.3 if t < 200 else 0.0
    neuron.step(current)
    e_traces.append(neuron.e_trace)
    voltages.append(neuron.v)
    thresholds.append(neuron.v_threshold_base + neuron.beta * neuron.a)

# e_trace peaks near spikes (where ψ is large)
# and decays with τ_a when neuron is far from threshold
peak_idx = np.argmax(e_traces)
print(f"Peak e_trace = {e_traces[peak_idx]:.4f} at t = {peak_idx} ms")
print(f"Final e_trace = {e_traces[-1]:.6f} (decayed)")
```

### Example 3: PyO3 Rust — high-performance stepping

```rust
use sc_neurocore_engine::neurons::EPropALIFNeuron;

let mut neuron = EPropALIFNeuron::new(20.0, 200.0, 1.0);

// 10,000 steps at I = 0.5
let mut spikes = 0;
for _ in 0..10_000 {
    spikes += neuron.step(0.5);
}
println!("ALIF: {spikes} spikes in 10 s, a = {:.4}", neuron.a);

// Access eligibility trace
println!("e_trace = {:.6}", neuron.e_trace);

// Reset
neuron.reset();
assert!((neuron.v).abs() < 1e-12);
assert!((neuron.a).abs() < 1e-12);
```

---

## Technical Reference

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `step(current: float) → int` | 0 or 1 | Advance 1 ms, return spike |
| `reset` | `reset() → None` | — | Restore v, a, e_trace to 0 |

### Python/Rust Parity

| Property | Python | Rust | Match |
|----------|--------|------|-------|
| Membrane decay | `alpha_m * v + I` | `alpha_m * v + I` | EXACT |
| Adaptive threshold | `θ_base + β·a` | `θ_base + β·a` | EXACT |
| Pseudo-derivative | `max(0, 1-|v-θ|)·0.3` | `((1-(v-θ).abs())·0.3).max(0)` | EXACT |
| Eligibility trace | `α_a·e + ψ` | `α_a·e + ψ` | EXACT |
| Spike condition | `v ≥ threshold` | `v ≥ threshold` | EXACT |
| Reset (spike) | `v=0, a=α_a·a+1` | `v=0, a=α_a·a+1` | EXACT |
| Reset (no spike) | `a *= α_a` | `a *= α_a` | EXACT |
| Parameters | tau_m=20, tau_a=200, β=0.07 | tau_m=20, tau_a=200, β=0.07 | EXACT |

### Supported operations

| Operation | Supported | Notes |
|-----------|-----------|-------|
| Population | Yes | Standard interface |
| Projection | Yes | Standard wiring |
| NetworkRunner | Yes | `EPropALIF` variant |
| SpikeMonitor | Yes | Binary spike output |
| PoissonInput | Yes | Tested |
| PyO3 bridge | Yes | `PyEPropALIFNeuron` with v, a, e_trace accessors |
| Model Zoo | No | Not in pre-built configs |

---

## Performance Benchmarks

### Criterion 0.8 (Rust engine)

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-05.

| Benchmark | Steps | Median | Per step |
|-----------|------:|-------:|---------:|
| `eprop_alif_10k_steps` | 10 000 | 22.6 µs | **2.3 ns** |

This makes EPropALIF one of the **fastest** models in SC-NeuroCore —
no exp() calls in step() (precomputed alphas), no sub-stepping, minimal
arithmetic (3 multiplies, 2 additions, 1 comparison, 1 abs).

### Python throughput

| Metric | Value |
|--------|------:|
| Isolation | ~500 000 steps/s |
| Network (10 neurons, 1 s) | ~40 000 neuron-steps/s |

### Rust speedup

| Metric | Python | Rust | Speedup |
|--------|-------:|-----:|--------:|
| Per step | ~2 µs | 2.3 ns | **~870×** |

---

## Citations

1. Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D.,
   Legenstein, R. & Maass, W. (2020). A solution to the learning
   dilemma for recurrent networks of spiking neurons. *Nature
   Communications*, 11(1), 3625.
   DOI: [10.1038/s41467-020-17236-y](https://doi.org/10.1038/s41467-020-17236-y)

2. Bellec, G., Salaj, D., Subramoney, A., Legenstein, R. & Maass, W.
   (2018). Long short-term memory and learning-to-learn in networks of
   spiking neurons. *Advances in Neural Information Processing Systems*
   (NeurIPS), 31, 787–797.
   arXiv: [1803.09574](https://arxiv.org/abs/1803.09574)

3. Zenke, F. & Ganguli, S. (2018). SuperSpike: Supervised learning in
   multilayer spiking neural networks. *Neural Computation*, 30(6),
   1514–1541.
   DOI: [10.1162/neco_a_01086](https://doi.org/10.1162/neco_a_01086)

4. Neftci, E. O., Mostafa, H. & Zenke, F. (2019). Surrogate gradient
   learning in spiking neural networks. *IEEE Signal Processing
   Magazine*, 36(6), 51–63.
   DOI: [10.1109/MSP.2019.2931595](https://doi.org/10.1109/MSP.2019.2931595)

5. Gerstner, W., Kistler, W. M., Naud, R. & Paninski, L. (2014).
   *Neuronal Dynamics: From Single Neurons to Networks and Models of
   Cognition*. Cambridge University Press, Ch. 9 (Adaptation and firing
   patterns).
   DOI: [10.1017/CBO9781107447615](https://doi.org/10.1017/CBO9781107447615)

6. Brea, J., Senn, W. & Pfister, J.-P. (2013). Matching recall and
   storage in sequence learning with spiking neural networks. *Journal
   of Neuroscience*, 33(23), 9565–9575.
   DOI: [10.1523/JNEUROSCI.4098-12.2013](https://doi.org/10.1523/JNEUROSCI.4098-12.2013)
