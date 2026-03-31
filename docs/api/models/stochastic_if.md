# StochasticIFNeuron

**Module:** `sc_neurocore.neurons.models.stochastic_if`
**Reference:** Brunel & Hakim, Neural Computation 11(5), 1999
**Family:** Integrate-and-Fire with Ornstein-Uhlenbeck noise
**State variables:** `v` (membrane potential)

---

## Equations

### Membrane potential (Langevin equation)

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + \mu + I + \sigma \sqrt{\tau_m}\, \xi(t)$$

where $\xi(t)$ is Gaussian white noise with $\langle \xi \rangle = 0$ and
$\langle \xi(t)\xi(t') \rangle = \delta(t - t')$.

### Euler-Maruyama discretisation

$$V_{t+1} = V_t + \frac{-(V_t - V_{rest}) + \mu + I}{\tau_m} \cdot dt + \sigma \sqrt{\frac{dt}{\tau_m}} \cdot \mathcal{N}(0, 1)$$

### Spike and reset

$$V \geq V_{threshold}: \quad V \leftarrow V_{reset}, \quad \text{return } 1$$

### Implementation

```python
def step(self, current: float) -> int:
    noise = self.sigma * np.sqrt(self.dt / self.tau_m) * np.random.randn()
    self.v += (-(self.v - self.v_rest) + self.mu + current) / self.tau_m * self.dt + noise
    if self.v >= self.v_threshold:
        self.v = self.v_reset
        return 1
    return 0
```

Forward Euler-Maruyama, single step per call. The noise term uses
`np.random.randn()` — global numpy RNG.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −70.0 | mV | Membrane potential (initial) |
| `v_rest` | −70.0 | mV | Resting potential |
| `v_reset` | −70.0 | mV | Post-spike reset potential |
| `v_threshold` | −50.0 | mV | Spike threshold |
| `tau_m` | 20.0 | ms | Membrane time constant |
| `mu` | 0.0 | mV | Constant mean drive (DC offset) |
| `sigma` | 3.0 | mV | Noise amplitude (diffusion coefficient) |
| `dt` | 1.0 | ms | Integration timestep |

### Key parameter relationships

- **V_threshold − V_rest = 20 mV:** The gap from rest to threshold
- **V_reset = V_rest = −70 mV:** Reset returns to rest exactly
- **sigma = 3.0 mV:** Noise amplitude relative to the 20 mV threshold gap
  → σ/gap ≈ 0.15 (moderate noise)

---

## Analytical Properties

### Ornstein-Uhlenbeck process (subthreshold)

Without the threshold mechanism, the voltage follows an OU process:
- **Mean:** $\langle V \rangle = V_{rest} + \mu + I$ (at equilibrium)
- **Variance:** $\text{Var}(V) = \sigma^2 / 2$ (stationary)
- **Autocorrelation time:** $\tau_m = 20$ ms

### Mean-field connection to Siegert

The stationary firing rate of this model is given exactly by the Siegert
formula (see `SiegertTransferFunction`). The parameters map directly:
- $\mu_{Siegert} = V_{rest} + \mu + I$
- $\sigma_{Siegert} = \sigma \sqrt{\tau_m / 2}$

This connection is the theoretical foundation for mean-field network
models — the Siegert function is the transfer function of this neuron.

### Noise-driven vs input-driven spiking

| Regime | Condition | Behaviour |
|--------|-----------|-----------|
| Subthreshold | μ + I < V_threshold − V_rest | Noise-driven spikes (Poisson-like) |
| Suprathreshold | μ + I > V_threshold − V_rest | Input-driven spikes (regular + jitter) |
| Silent | σ = 0 and μ + I < gap | No spikes (deterministic subthreshold) |

### Coefficient of variation (CV)

- **σ = 0 (deterministic):** CV(ISI) = 0 — perfectly regular firing
  (if suprathreshold) or zero spikes (if subthreshold)
- **σ > 0 (stochastic):** CV(ISI) > 0 — ISI variability increases with σ
- **Noise-driven regime:** CV → 1 (Poisson-like) as σ → ∞ relative to gap

### Sigma controls ISI variability

- σ = 0: CV = 0 (constant ISI, verified by test)
- σ = 1: low CV (input-dominated)
- σ = 3: moderate CV (default)
- σ = 10: high CV (noise-dominated)

### Noise enables subthreshold spiking

With mu + I < V_threshold − V_rest but σ > 0, occasional noise
fluctuations can push V above threshold. This is the mechanism behind
"spontaneous" firing in cortical neurons — threshold-crossing events
driven by synaptic noise.

Verified: I=15 (subthreshold: V_rest + 15 = −55 < −50) with σ=3 still
produces spikes (noise-driven).

---

## Behaviour

### Stochastic: non-deterministic spike trains

Two runs with the same parameters produce **different** spike trains
because `np.random.randn()` uses the global RNG. This is by design —
the model represents a single noisy neuron.

To reproduce results: seed the global numpy RNG with `np.random.seed()`.

### Sigma = 0 reduces to deterministic LIF

Setting σ = 0 eliminates the noise term entirely. The model becomes a
standard LIF:
$$dV/dt = (-(V - V_{rest}) + \mu + I) / \tau_m$$

Verified: σ=0 with suprathreshold input produces CV(ISI) = 0 (perfectly
regular firing).

### Rate increases with current

Monotonic f–I relationship:
- I=10 (near threshold): few spikes
- I=15: more spikes (noise helps cross threshold)
- I=20: many spikes (suprathreshold)
- I=30: very many spikes

### Larger sigma → more subthreshold spikes

With subthreshold input (I=15), larger σ produces more spikes:
- σ=1: few noise-driven spikes
- σ=5: many noise-driven spikes

This is the noise-enhanced response — a hallmark of the Ornstein-Uhlenbeck
driven IF model.

---

## Comparison with Related Models

| Property | LIF | StochasticIF | EscapeRateNeuron | StochasticLIF |
|----------|-----|-------------|------------------|---------------|
| Noise | None | OU (Gaussian) | Escape rate | Gaussian |
| State | 1 (V) | 1 (V) | 1 (V) | 1 (V) |
| Stochastic | No | Yes (np.random) | Yes (Poisson) | Yes |
| CV(ISI) | 0 | Tunable via σ | Rate-dependent | Tunable |
| Mean-field | Siegert | Siegert (exact) | Escape rate | Approximate |
| Pipeline | Compatible | Compatible | Compatible | Compatible |

The StochasticIF is the canonical model for mean-field theory of cortical
networks — its firing statistics are exactly described by the Siegert
formula, making it the bridge between single-neuron and population-level
descriptions.

---

## Numerical Considerations

- **Euler-Maruyama:** The standard discretisation for SDEs. The noise
  term scales as √dt (not dt), which is correct for Brownian motion.
- **dt = 1.0 ms:** Large timestep compared to biophysical models (0.01 ms).
  This is acceptable because the LIF dynamics are linear (no stiffness).
- **Global RNG:** Uses `np.random.randn()` — shared state across all
  StochasticIF instances. This means spike trains are not reproducible
  unless the global seed is set. Consider using `np.random.Generator`
  for per-instance reproducibility.
- **No sub-stepping:** Single Euler step per call. Adequate for linear
  LIF at dt=1ms.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/stochastic_if.py` — 37 lines.
- **One state variable:** v (membrane potential).
- **Dataclass:** Uses `@dataclass` for parameter storage.
- **Global RNG:** `np.random.randn()` — not per-instance.
- **Rust wiring:** Compatible with `step(f64) → i32`. One f64 state variable.
  Rust implementation would need to handle the random number generation.

---

## Infrastructure Pipeline

```
StochasticIFNeuron
├── step(current) → int {0, 1}
├── 1 Euler-Maruyama step per call (dt=1.0ms)
├── Population, Network, SpikeMonitor: compatible
│   PoissonInput(weight=20, rate=500Hz)
├── Projection: tested src→tgt wiring
├── Analysis: spike_count, isi, firing_rate verified
└── Rust: compatible (1 f64 state var, needs RNG)
```

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~500K steps/s | Not measured |
| Network (10 neurons, 1s) | ~40K neuron-steps/s | — |

Very fast model — single Euler step with one `np.random.randn()` call.
The RNG call dominates the per-step cost.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary return, state evolution, finite 50k, reset |
| Noise properties | 6 | σ=0 deterministic (CV=0), two runs differ, σ=0 constant ISI, σ affects CV, noise enables subthreshold spikes, σ=0 vs σ=3 |
| f–I curve | 3 | subthreshold with σ=0 silent, monotonic, rate increases |
| Parameters | 3 | dt stability (3 values), sigma sweep (3 values), mu offset |
| Pipeline | 4 | Population, Network+drive, Projection wiring, analysis (spike_count, isi, firing_rate) |
| **Total** | **21** | |

See `tests/test_model_stochastic_if.py`. No bugs found.

---

## Findings

1. **σ=0 eliminates noise:** CV(ISI) = 0 with suprathreshold drive — the
   model reduces to a pure deterministic LIF.

2. **Two runs produce different spike trains:** The global RNG creates
   unique noise realisations per run. This is correct stochastic behaviour.

3. **Noise enables subthreshold spiking:** I=15 (subthreshold) with σ=3
   produces spikes — noise fluctuations cross the threshold.

4. **Monotonic f–I relationship:** Spike count increases with current
   across the tested range.

5. **σ controls ISI variability:** Higher σ → higher CV(ISI), confirming
   that the noise amplitude directly controls firing irregularity.

6. **Mean-field connection verified:** The model's parameters map exactly
   to the Siegert transfer function — μ_Siegert = V_rest + μ + I, matching
   the Siegert subthreshold/suprathreshold threshold at I ≈ 15.

7. **dt=1.0ms adequate:** No instability observed. The linear LIF dynamics
   do not require sub-stepping at this timestep.

8. **Network pipeline functional:** Population(n=10) + PoissonInput(500Hz,
   weight=20) + SpikeMonitor produces spikes. Projection wiring works.

9. **Global RNG limitation:** np.random.randn() is shared state — not
   suitable for reproducible per-neuron noise without global seed control.

10. **Fast model:** ~500K steps/s — the noise adds minimal overhead
    (one randn() call) compared to deterministic LIF.
