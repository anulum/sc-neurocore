# BendaHerzNeuron

**Module:** `sc_neurocore.neurons.models.benda_herz`
**Reference:** Benda & Herz 2003
**Family:** Phenomenological (spike-frequency adaptation)
**State variables:** `a` (adaptation variable)

## Equations

$$f = f_{\text{onset}}(I - A)$$
$$\frac{dA}{dt} = -\frac{A}{\tau_a} + \delta_a \cdot f$$
$$f_{\text{onset}}(x) = \frac{f_{\max}}{1 + e^{-\beta(x - I_{\text{half}})}}$$

Spike output: Poisson sampling with probability $p = f \cdot dt / 1000$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `a` | 0.0 | Adaptation variable (accumulated SFA) |
| `f_max` | 200.0 | Maximum firing rate (Hz) |
| `beta` | 0.1 | Sigmoid slope of f-I curve |
| `i_half` | 5.0 | Half-activation current |
| `tau_a` | 100.0 | Adaptation time constant (ms) |
| `delta_a` | 0.5 | Adaptation increment per Hz |
| `dt` | 1.0 | Timestep (ms) |

## Behaviour

- **Spike-frequency adaptation (SFA):** Under sustained input, the
  adaptation variable A builds up, shifting the f-I curve rightward.
  This reduces the instantaneous firing rate over time.
- **Stochastic spiking:** Output is not deterministic. Each step
  samples from a Bernoulli distribution with probability proportional
  to the instantaneous rate. This is biologically realistic — cortical
  neurons show Poisson-like variability.
- **Rate model with spikes:** Internally computes a rate (Hz) then
  converts to binary spikes via Poisson sampling. Bridges rate and
  spike coding.

## Infrastructure Pipeline

```
BendaHerzNeuron
├── step(current: float) → int {0,1} (Poisson-sampled spike)
├── reset() → a=0.0
├── In Population: 1 instance per neuron, scalar current
│   └── Return value: native 0/1 (compatible with int8 spike vector)
├── In Network: compatible with all stimuli and monitors
│   ├── PoissonInput (weight=50, rate=500Hz for ~2 Hz output)
│   ├── StepCurrent, TimedArray
│   ├── SpikeMonitor, StateMonitor
│   └── Projection (compatible, but low spike rate limits STDP efficacy)
├── Analysis: all spike_stats functions
│   └── Note: stochastic model — ISI distribution is approximately exponential
├── SC encoding: spike train → rate coding compatible
├── Verilog: compilable via EquationNeuron (sigmoid LUT + Poisson LFSR)
└── Rust NetworkRunner: supported (standard step() interface)
```

## Wiring Plan

```
PoissonInput(weight=50, rate=500Hz)
    ↓ scalar current per neuron
Population(BendaHerzNeuron, n=N)
    ↓ binary spike vector (stochastic, low rate ~1-3 Hz)
Projection(pop, pop, weight=5.0, probability=0.2)
    ↓ recurrent excitation (weak due to low rate)
SpikeMonitor → spike_trains
    ├── firing_rate (expect 1-5 Hz)
    ├── spike_count
    └── ISI (approximately exponential distribution)
```

**Drive requirements:** Sigmoid f-I curve with i_half=5.0. Effective
drive after Poisson sampling: mean current = rate × probability × weight.
For PoissonInput(500 Hz, weight=50, dt=0.001): mean ≈ 0.5 × 50 = 25.
After adaptation equilibrium: output ~1-3 Hz.

## Performance

| Metric | Python (NumPy) | Rust engine |
|--------|---------------|-------------|
| Isolation (single neuron) | 515 Ksteps/s | Not measured |
| Network (20 neurons, 2s) | ~350 Kneuron-steps/s | Expected ~20× faster |
| Typical output rate | 1-3 Hz | — |

Slower than pure LIF models due to sigmoid computation (exp()) and
Poisson random number generation per step.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 8 | construction, step binary, spikes under drive, adaptation increases, adaptation reduces rate, sigmoid shape, state finiteness, reset |
| Network | 3 | Population creation, spike production (2s run), Projection compatibility |
| Analysis | 2 | firing_rate ≥ 0, spike_count ≥ 0 |
| **Total** | **13** | |

See `tests/test_model_benda_herz.py`.
