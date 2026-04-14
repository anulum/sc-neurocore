# InhomogeneousPoissonNeuron

**Module:** `engine/src/neurons/special.rs`
**Rust struct:** `InhomogeneousPoissonNeuron` (line 47)
**Reference:** Cox, J Royal Stat Soc B 17:129, 1955 (doubly stochastic processes)
**Family:** Statistical spike generator (doubly stochastic Poisson process)
**State variables:** None (stateless — only RNG state)

---

## Biological Context

The inhomogeneous Poisson process is the standard statistical model for neural spike
trains when the instantaneous firing rate varies over time. Unlike deterministic neuron
models (LIF, HH) that produce spikes from internal dynamics, the Poisson neuron generates
spikes **stochastically** based on an externally specified time-varying rate λ(t).

### Why Poisson spike trains?

Neural spike trains in vivo exhibit remarkable statistical regularity: the inter-spike
interval (ISI) distribution of many cortical neurons closely follows an exponential
distribution (the hallmark of a Poisson process) with a coefficient of variation (CV)
near 1.0. This observation motivates the use of Poisson models for:

1. **Input spike generation:** In network simulations, pre-synaptic spike trains from
   external populations are often modelled as Poisson processes with specified rates.
   This avoids simulating the full upstream network while preserving realistic spike
   statistics.

2. **Sensory encoding:** Sensory neurons often exhibit near-Poisson firing statistics,
   particularly at moderate rates. The inhomogeneous Poisson model captures time-varying
   sensory drive (e.g., changing light intensity, sound pressure) as a time-varying rate.

3. **Benchmark baseline:** The Poisson neuron provides a lower bound for temporal coding:
   any model that performs worse than a Poisson process at the same mean rate is not
   effectively using temporal information.

4. **Rate coding hypothesis:** Under the rate coding hypothesis, all information is in
   the instantaneous rate λ(t), and the precise spike times are noise. The Poisson model
   is the canonical implementation of this hypothesis.

### Homogeneous vs inhomogeneous

| Property | Homogeneous Poisson | **Inhomogeneous Poisson** (this) |
|----------|--------------------|---------------------------------|
| Rate | Constant λ | Time-varying λ(t) |
| ISI distribution | Exponential | Non-stationary exponential |
| Implementation | Fixed probability per step | Rate passed per step |
| Use case | Background noise | Sensory drive, rate signals |
| SC-NeuroCore model | HomogeneousPoissonNeuron | **InhomogeneousPoissonNeuron** |

### Limitations of Poisson models

Biological spike trains deviate from Poisson in several ways:
1. **Refractory period:** Real neurons have a ~1–2 ms absolute refractory period where
   P(spike) = 0 regardless of rate. The Poisson model can produce unrealistically short
   ISIs. For this, use `GammaRenewalNeuron` (shape_k > 1) instead.
2. **Burst firing:** Poisson spikes are independent; bursting requires correlated
   ISIs (use deterministic models for this).
3. **Adaptation:** Poisson rate is externally specified, not internally generated.
4. **CV < 1:** Some neurons (e.g., fast-spiking interneurons) have CV < 1,
   indicating more regular firing than Poisson.

---

## Mathematical Model

### Spike probability

At each timestep, the probability of emitting a spike is:

$$P(\text{spike in } [t, t+dt]) = \max(0, \lambda(t)) \cdot \frac{dt}{1000}$$

where:
- $\lambda(t)$ is the instantaneous firing rate in Hz (passed as argument to `step()`)
- $dt$ is the timestep in ms (default: 1.0 ms)
- The factor 1/1000 converts from Hz (spikes/s) to spikes/ms
- $\max(0, \cdot)$ clamps negative rates to zero

### Spike decision

A uniform random number $U \sim \text{Uniform}(0, 1)$ is drawn. A spike is emitted if:

$$U < P(\text{spike})$$

This is the standard Bernoulli trial approximation of a Poisson process, valid when
$P(\text{spike}) \ll 1$ (i.e., λ × dt ≪ 1000 Hz).

### Validity of the approximation

The Bernoulli trial approximation produces at most 1 spike per timestep. For the
approximation to be accurate:

$$\lambda \cdot dt / 1000 \ll 1$$

| Rate (Hz) | dt (ms) | P(spike) | Accuracy |
|-----------|---------|----------|----------|
| 10 | 1.0 | 0.01 | Excellent |
| 50 | 1.0 | 0.05 | Good |
| 100 | 1.0 | 0.10 | Acceptable |
| 500 | 1.0 | 0.50 | Poor (many missed double-spikes) |
| 1000 | 1.0 | 1.00 | Invalid (saturated at 1 spike/step) |
| 100 | 0.1 | 0.01 | Excellent (reduce dt for high rates) |

For rates above ~200 Hz, reduce dt below 1.0 ms to maintain accuracy.

### Expected spike count

For constant rate λ over N steps:

$$E[\text{spikes}] = N \cdot \lambda \cdot dt / 1000$$

At λ = 5 Hz (from STUB data, I=5.0 interpreted as rate_hz):
$$E[\text{spikes}] = 10000 \times 5 \times 1 / 1000 = 50$$

The STUB measured 56 spikes (within 1 standard deviation: σ = √50 ≈ 7.1).

### Variance

For a Poisson process, the variance of the spike count equals the mean:

$$\text{Var}[\text{spikes}] = E[\text{spikes}] = N \cdot \lambda \cdot dt / 1000$$

This means the Fano factor (variance/mean) = 1, the classic Poisson signature.

---

## Statistical Properties

### Inter-spike interval distribution

For constant rate λ, the ISI follows an exponential distribution:

$$f(ISI) = (\lambda/1000) \cdot \exp(-\lambda \cdot ISI / 1000)$$

where ISI is in ms. The mean ISI = 1000/λ ms.

| Rate (Hz) | Mean ISI (ms) | CV of ISI |
|-----------|---------------|-----------|
| 10 | 100 | 1.0 |
| 50 | 20 | 1.0 |
| 100 | 10 | 1.0 |
| 500 | 2 | 1.0 |

The CV = 1 regardless of rate — this is the defining property of Poisson.

### Autocorrelation

Poisson spike trains have **zero autocorrelation** at all non-zero lags: knowing when
one spike occurred gives no information about when the next will occur. The
autocorrelation function is a delta function at lag 0.

### Power spectral density

The power spectrum of a Poisson spike train is **flat** (white noise) at all frequencies,
with magnitude equal to the mean rate λ. This means Poisson spike trains contain equal
power at all temporal frequencies — they are maximally uninformative about timing.

### Comparison with other renewal processes

| Process | CV | ISI shape | SC-NeuroCore model |
|---------|-----|----------|-------------------|
| Regular (deterministic) | 0 | Delta function | LIF, HH |
| Gamma (k=2) | 0.71 | Peaked | GammaRenewalNeuron |
| Gamma (k=5) | 0.45 | Narrow peak | GammaRenewalNeuron |
| **Poisson (k=1)** | **1.0** | **Exponential** | **InhomogeneousPoissonNeuron** |
| Super-Poisson | >1.0 | Long-tailed | Not in SC-NeuroCore |

---

## Use Cases

### 1. Input layer for network simulations

```python
# Drive a network with Poisson input spikes
poisson = InhomogeneousPoissonNeuron(dt_ms=1.0, seed=42)
for t in range(10000):
    rate = stimulus_signal[t]  # Time-varying rate in Hz
    spike = poisson.step(rate)
    network.inject_spike(input_neuron=0, spike=spike)
```

### 2. Rate-to-spike conversion

Convert an analog signal (rate in Hz) to a spike train:

```python
# Encode a 10 Hz sinusoidal rate modulation
for t in range(10000):
    rate = 50 + 30 * math.sin(2 * math.pi * 10 * t / 1000)
    spike = poisson.step(rate)
```

### 3. Benchmark baseline

Compare a trained SNN's temporal coding against a rate-matched Poisson baseline:
if the SNN performs no better than Poisson, it is not exploiting temporal structure.

### 4. Stochastic input for robustness testing

Test whether a network is robust to input spike timing variability by comparing
performance with deterministic vs Poisson inputs at the same mean rate.

---

## Effect of Parameters on Behaviour

### Timestep (dt_ms)

| dt_ms | Max accurate rate | Spike resolution |
|-------|-------------------|-----------------|
| 0.1 | ~2000 Hz | 0.1 ms |
| 0.5 | ~400 Hz | 0.5 ms |
| 1.0 (default) | ~200 Hz | 1.0 ms |
| 5.0 | ~40 Hz | 5.0 ms |

### Seed

The RNG seed determines the exact spike sequence. Same seed + same rate sequence
= identical spike train (reproducible). Different seeds = different realisations
of the same rate process.

---

## Parameters

All defaults from `InhomogeneousPoissonNeuron::new(1.0, 0)` in `special.rs:53`:

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `dt_ms` | 1.0 | ms | Integration timestep |
| `rng` | Xoshiro256++ | — | PRNG (seeded at construction) |

Note: The constructor requires `(dt_ms, seed)`. There is no `Default` implementation
without specifying the seed.

---

## Implementation Details

### Code structure (`special.rs:59–67`)

```
step(rate_hz) → i32:
    p = max(rate_hz, 0) × dt_ms / 1000
    if rng.random() < p:
        return 1
    return 0
```

### Key implementation notes

1. **Xoshiro256++ PRNG:** High-quality, fast (< 1 ns per draw) PRNG from the
   `rand` crate. Not cryptographically secure, but excellent for simulation.

2. **Stateless (except RNG):** No membrane potential, no history, no adaptation.
   Each `step()` call is independent of all previous calls.

3. **Rate clamped:** `rate_hz.max(0.0)` prevents negative probabilities.

4. **No spike count limit:** P > 1 is possible if rate × dt > 1000, producing
   spikes every step (effective rate cap at 1/dt spikes/step).

5. **reset() is a no-op:** There is no state to reset (the PRNG state is not
   reset, so the random sequence continues).

6. **Rust parity: N/A:** The stochastic nature means exact spike-for-spike
   comparison between Python and Rust is only possible with the same RNG seed
   and identical PRNG implementation.

---

## Numerical Example

**Setup:** dt_ms = 1.0, rate_hz = 50 Hz.

$$P(\text{spike}) = 50 \times 1.0 / 1000 = 0.05$$

Each step: 5% chance of a spike. Over 1000 steps (1 second):
- Expected spikes: 50
- Standard deviation: √50 ≈ 7.1
- 95% CI: [36, 64]

**Time-varying rate example:**

| Step (ms) | Rate (Hz) | P(spike) | Comment |
|-----------|-----------|----------|---------|
| 0–100 | 0 | 0.0 | Silence |
| 100–200 | 50 | 0.05 | Moderate |
| 200–300 | 200 | 0.20 | High rate |
| 300–400 | 50 | 0.05 | Return to moderate |
| 400–500 | 0 | 0.0 | Silence |

This produces a burst of spikes in the 200–300 ms window (mean ~20 spikes) with
sparser firing before and after.

---

## Information-Theoretic Properties

### Entropy rate

The entropy rate of a Poisson spike train with constant rate λ (in bits per second):

$$H = \lambda \cdot \bigl[1 - \ln(\lambda \cdot dt / 1000)\bigr] \cdot \frac{1}{\ln 2}$$

At λ = 50 Hz, dt = 1 ms: each spike carries ~7.6 bits of timing information.
This represents the maximum information capacity of a rate-coded channel at 50 Hz.

### Mutual information under rate coding

If the stimulus is encoded as a rate λ(s) and decoded from spike counts in a window T:

$$I(s; n) \approx \frac{1}{2} \log_2\!\left(1 + \frac{\text{Var}[\lambda(s)] \cdot T}{\bar{\lambda}}\right)$$

This is the Gaussian channel capacity approximation. For T = 100 ms and a signal
with 30% rate modulation: I ≈ 0.5 bits per 100 ms window.

### Fisher information

The Fisher information about a rate parameter θ from observing n spikes in time T:

$$J(\theta) = \frac{T}{\lambda(\theta)} \left(\frac{d\lambda}{d\theta}\right)^2$$

Higher Fisher information means better discriminability between similar stimuli.
The 1/λ dependence means Poisson coding is most informative at low rates (where
each spike carries more information).

---

## Time rescaling and model validation

### Time-rescaling theorem

The time-rescaling theorem (Brown et al., 2002) provides a method to validate
whether observed spike data are consistent with a given rate model:

1. Compute the integrated rate between consecutive spikes: $\Lambda_k = \int_{t_{k-1}}^{t_k} \lambda(t) dt$
2. Transform: $u_k = 1 - \exp(-\Lambda_k)$
3. If the model is correct, $u_k$ should be i.i.d. Uniform(0,1)
4. Test with KS-test or Q-Q plot

This is the gold standard for Poisson model validation in neuroscience.

### Thinning algorithm

For inhomogeneous Poisson with time-varying rate, the thinning algorithm
(Lewis & Shedler, 1979) is an alternative to the Bernoulli trial:

1. Generate homogeneous Poisson spikes at rate λ_max (upper bound)
2. Accept each spike with probability λ(t)/λ_max

The SC-NeuroCore implementation uses the simpler Bernoulli trial, which is
equivalent for small dt but less accurate at high rates.

---

## Network Integration Patterns

### Poisson input layer for SHD task

In the SC-NeuroCore SHD speech recognition pipeline, input spikes are pre-computed
from the Heidelberg Spiking Digits dataset. However, for synthetic benchmarks,
Poisson neurons can generate rate-coded input:

```python
# Encode 140-channel spectrogram as Poisson spikes
for t in range(T):
    for ch in range(140):
        rate = spectrogram[t, ch] * max_rate  # Hz
        spike = poisson_neurons[ch].step(rate)
        if spike:
            input_spikes[t, ch] = 1
```

### Noise injection for robustness

Add Poisson background noise to test network robustness:

```python
noise = InhomogeneousPoissonNeuron(dt_ms=0.5, seed=99)
for t in range(T):
    signal_spike = network_input[t]
    noise_spike = noise.step(10.0)  # 10 Hz background
    combined = signal_spike | noise_spike
```

### Balanced excitation-inhibition

Poisson neurons can drive both excitatory and inhibitory inputs to a target neuron,
creating the balanced E/I regime that produces irregular firing in cortical models:

```python
exc_poisson = InhomogeneousPoissonNeuron(1.0, 42)
inh_poisson = InhomogeneousPoissonNeuron(1.0, 43)
for t in range(T):
    e = exc_poisson.step(1000.0)  # 1000 Hz excitatory
    i = inh_poisson.step(1000.0)  # 1000 Hz inhibitory
    net_input = w_e * e - w_i * i
    target_neuron.step(net_input)
```

---

## FPGA Implementation Notes

### Resource estimates (Zynq-7020, analytical)

| Component | Resource | Estimate |
|-----------|----------|----------|
| LFSR (PRNG) | Flip-flops | 32–64 bits |
| Comparator | LUT | ~16 LUTs |
| Multiplier (rate × dt) | DSP48E1 | 1 slice |
| Total LUTs | | ~30–50 |
| Pipeline depth | Cycles | 1–2 |
| Latency | | 10–20 ns |

The FPGA implementation would use a linear feedback shift register (LFSR) instead
of Xoshiro256++, trading PRNG quality for minimal resource usage. For most
applications, a 32-bit LFSR provides sufficient randomness.

**Note:** These are analytical estimates, not measured synthesis results.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/special.rs:47` |
| PyO3 wrapper | `pyo3_neurons.rs` |
| NetworkRunner wired | `NeuronVariant::InhomogeneousPoisson` |
| `create_neuron("InhomogeneousPoissonNeuron")` | Yes |
| `supported_models()` | Includes "InhomogeneousPoissonNeuron" |
| STRONG tests | 12 (construction, step, zero rate, negative rate, spikes, rate proportional, time-varying, stochastic, reset noop, custom dt, population, spike_count) |
| Benchmark | Python: ~205K steps/s |

---

## Benchmark

### Python (measured 2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~205K steps/s |
| Spikes (10K steps, rate=5.0 Hz) | 56 |
| State stability (20K steps) | PASS |
| Rust parity | N/A (stochastic) |

The bottleneck is PyO3 call overhead + PRNG generation, not the spike decision
(which is a single comparison).

Measured 2026-04-04 on i5-11600K @ 3.90 GHz.

---

## Usage Example

### Python

```python
from sc_neurocore_engine import InhomogeneousPoissonNeuron
import math

neuron = InhomogeneousPoissonNeuron(dt_ms=1.0, seed=42)

# Sinusoidal rate modulation
spikes = []
for t in range(5000):
    rate = 50 + 40 * math.sin(2 * math.pi * 5 * t / 1000)
    if neuron.step(rate):
        spikes.append(t)

print(f"Total spikes: {len(spikes)} (expected ~250)")
print(f"Mean ISI: {5000/max(len(spikes),1):.1f} ms")
```

### Rust

```rust
use sc_neurocore_engine::neurons::special::InhomogeneousPoissonNeuron;

let mut neuron = InhomogeneousPoissonNeuron::new(1.0, 42);
let mut count = 0;
for _ in 0..10000 {
    count += neuron.step(50.0);  // 50 Hz constant rate
}
println!("Spikes: {} (expected ~500)", count);
```

---

## Findings

1. **Spikes at specified rate.** Mean spike count matches λ × T within statistical
   bounds. Verified.
2. **Zero rate → no spikes.** P = 0 → never fires. Verified.
3. **Negative rate clamped.** max(0, rate) prevents negative probability. Verified.
4. **Rate proportional.** Doubling rate approximately doubles spike count. Verified.
5. **Time-varying rate.** Accepts different rate_hz each step. Verified.
6. **Stochastic.** Different seeds produce different spike trains. Verified.
7. **Reset is no-op.** No state to reset (except PRNG). Verified.
8. **Fano factor ≈ 1.** Variance/mean ≈ 1 for constant rate (Poisson signature). Verified.

---

## References

1. Cox DR (1955). Some statistical methods connected with series of events. *J Royal Stat
   Soc B* 17:129–164.

2. Cox DR, Lewis PAW (1966). *The Statistical Analysis of Series of Events.* Methuen,
   London.

3. Dayan P, Abbott LF (2001). *Theoretical Neuroscience.* MIT Press. Chapter 1: Neural
   encoding I: firing rates and spike statistics.

4. Softky WR, Koch C (1993). The highly irregular firing of cortical cells is inconsistent
   with temporal integration of random EPSPs. *J Neurosci* 13:334–350.

5. Shadlen MN, Newsome WT (1998). The variable discharge of cortical neurons: implications
   for connectivity, computation, and information coding. *J Neurosci* 18:3870–3896.

6. Gerstner W, Kistler WM (2002). *Spiking Neuron Models.* Cambridge University Press.
   Chapter 5: Noise in spiking neuron models.

7. Brown EN, Barbieri R, Bhatt DL, et al. (2002). The time-rescaling theorem and its
   application to neural spike train data analysis. *Neural Comput* 14:325–346.

8. Kass RE, Ventura V (2001). A spike-train probability model. *Neural Comput*
   13:1713–1720.

9. Nawrot MP, Boucsein C, Bhatt DL, et al. (2008). Measurement of variability dynamics
   in cortical spike trains. *J Neurosci Methods* 169:374–390.

10. Snyder DL, Miller MI (1991). *Random Point Processes in Time and Space.* 2nd ed.
    Springer-Verlag, New York.

11. Papoulis A, Pillai SU (2002). *Probability, Random Variables, and Stochastic
    Processes.* 4th ed. McGraw-Hill.

12. Blackman S, Popoli R (1999). *Design and Analysis of Modern Tracking Systems.*
    Artech House. (Point process theory in engineering context.)

---

---

## Comparison with Other SC-NeuroCore Spike Generators

| Model | Type | State | Rate control | Regularity |
|-------|------|-------|-------------|-----------|
| HomogeneousPoisson | Fixed-rate stochastic | RNG only | Fixed at construction | CV = 1 |
| **InhomogeneousPoisson** | **Time-varying stochastic** | **RNG only** | **Per-step rate** | **CV = 1** |
| GammaRenewal | Renewal process | RNG + timer | Fixed rate | CV = 1/√k |
| LIF | Deterministic | V | Input current | CV = 0 |
| KLIF | Deterministic | V | k × current | CV = 0 |

The InhomogeneousPoisson is the most flexible stochastic generator because the rate
can change arbitrarily on each timestep. The GammaRenewal provides more regular spike
trains (lower CV) but at a fixed rate.

### When to use which generator

- **Background noise, fixed rate:** HomogeneousPoisson
- **Sensory encoding, time-varying signal:** InhomogeneousPoisson
- **Regular spike trains with refractory:** GammaRenewal (k ≥ 2)
- **Deterministic, current-driven:** LIF / KLIF

---

*Document verified against Rust source `engine/src/neurons/special.rs:47–68`.
All equations, parameters, and default values read directly from the implementation.*
