# BertramPhantomBurster

**Module:** `sc_neurocore.neurons.models.bertram_phantom`
**Reference:** Bertram et al. 2008
**Family:** Bursting (biophysical, pancreatic β-cell)
**State variables:** `v` (voltage), `s1` (fast slow), `s2` (slow slow)

## Equations

$$C_m \frac{dV}{dt} = -(I_{Ca} + I_K + I_{s1} + I_{s2} + I_L) + I_{\text{ext}}$$
$$\frac{ds_1}{dt} = \frac{s_{1,\infty}(V) - s_1}{\tau_{s1}}$$
$$\frac{ds_2}{dt} = \frac{s_{2,\infty}(V) - s_2}{\tau_{s2}}$$

Boltzmann activation: $x_\infty(V) = 1/(1 + e^{(V_x - V)/s_x})$.
Spike detection: upward threshold crossing ($V_{\text{prev}} < \theta$, $V \geq \theta$).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | -50.0 | Membrane voltage (mV) |
| `s1`, `s2` | 0.1 | Slow gating variables |
| `g_ca` | 3.6 | Ca²⁺ conductance |
| `g_k` | 10.0 | K⁺ conductance |
| `g_s1`, `g_s2` | 4.0 | Slow conductances |
| `g_l` | 0.2 | Leak conductance |
| `tau_s1` | 20,000 | Fast slow time constant (ms) = 20 s |
| `tau_s2` | 100,000 | Slow slow time constant (ms) = 100 s |
| `dt` | 0.5 | Timestep (ms) |
| `v_threshold` | -20.0 | Spike detection threshold (mV) |

## Behaviour

- **Phantom bursting:** Two slow variables with timescales separated by 5×
  (20s vs 100s) create a bursting pattern mediated by a "phantom" slow manifold.
  This is the mechanism for pancreatic β-cell electrical activity.
- **High current required:** Default parameters need I ≥ 200 for suprathreshold
  spiking. At I < 100, the model shows sub-threshold oscillations only.
- **Very slow dynamics:** Full burst cycle takes tens of seconds.
  For 50s simulation at dt=0.5ms: 100K steps.

## Infrastructure Pipeline

```
BertramPhantomBurster
├── step(current: float) → int {0,1} (threshold crossing)
├── reset() → v=-50, s1=0.1, s2=0.1
├── In Population: scalar current, standard interface
│   └── Return value: 0/1 (native binary)
├── In Network: compatible with all stimuli and monitors
│   ├── PoissonInput (weight=200, rate=1000Hz for spiking regime)
│   ├── SpikeMonitor, StateMonitor
│   └── Projection (compatible)
├── Analysis: all spike_stats functions
│   └── Burst detection relevant (patterns.burst_detection)
├── SC encoding: spike train → rate coding
├── Verilog: compilable (3 Boltzmann LUTs + 5 currents + Euler)
│   Estimated ~150 LUTs per neuron at Q8.8
└── Rust NetworkRunner: supported (standard interface)
```

## Wiring Plan

```
PoissonInput(weight=200, rate=1000Hz)
    ↓ scalar current (needs ≥200 for spiking)
Population(BertramPhantomBurster, n=N)
    ↓ binary spike vector (burst pattern)
SpikeMonitor → spike_trains
    ├── firing_rate (within burst: ~24 Hz at I=200)
    ├── burst_detection (inter-burst interval ~20s)
    └── ISI bimodal (intra-burst short, inter-burst long)
```

## Performance

| Metric | Python (NumPy) | Rust engine |
|--------|---------------|-------------|
| Isolation (single neuron) | 268 Ksteps/s | Not measured |
| Network (10 neurons, 1s) | ~200 Kneuron-steps/s | Expected ~20× faster |
| Spiking threshold | I ≥ 200 | — |
| Typical burst rate | ~24 Hz within bursts | — |

Slower than LIF due to 3 Boltzmann evaluations (exp()) per step.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 8 | construction, step binary, subthreshold at I=10, spikes at I=200, dual slow variables change, threshold crossing, state finiteness, reset |
| Network | 3 | Population creation, spike production, spike train extraction |
| Analysis | 3 | firing_rate > 0, spike_count > 100, ISI finite |
| **Total** | **14** | |

See `tests/test_model_bertram_phantom.py`.
