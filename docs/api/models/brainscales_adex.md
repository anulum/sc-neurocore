# BrainScaleSAdExNeuron

**Module:** `sc_neurocore.neurons.models.brainscales_adex`
**Reference:** Schemmel et al. 2010 (BrainScaleS-2)
**Family:** Hardware (analog neuromorphic)
**State variables:** `v` (membrane voltage), `w` (adaptation current)

## Equations

$$\tau \frac{dV}{dt} = -(V - V_{\text{rest}}) + \Delta_T e^{(V-V_{rh})/\Delta_T} - w + I$$
$$\tau_w \frac{dw}{dt} = a(V - V_{\text{rest}}) - w$$

Spike: $V \geq V_\theta \Rightarrow V \to V_{\text{reset}}, \; w \to w + b$.
Exponential argument clipped to [-20, 20] for numerical safety.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | -65.0 | Membrane voltage (mV) |
| `w` | 0.0 | Adaptation current |
| `v_rest` | -65.0 | Resting potential (mV) |
| `v_reset` | -68.0 | Post-spike reset (mV) |
| `v_threshold` | -50.0 | Spike threshold (mV) |
| `delta_t` | 2.0 | Sharpness of exp spike initiation (mV) |
| `v_rh` | -55.0 | Rheobase voltage (mV) |
| `tau` | 20.0 | Membrane time constant (ms) |
| `tau_w` | 100.0 | Adaptation time constant (ms) |
| `a` | 0.5 | Subthreshold adaptation conductance |
| `b` | 7.0 | Spike-triggered adaptation increment |
| `hw_speedup` | 1000.0 | BrainScaleS-2 hardware speedup factor |
| `dt` | 0.1 | Timestep (ms) |

## Behaviour

- **BrainScaleS-2 emulation:** Models the analog AdEx circuit running at
  1000× biological real-time on the Heidelberg neuromorphic chip.
- **Strong adaptation:** b=7 causes substantial spike-frequency adaptation.
  With moderate drive, fires only a few spikes then goes silent.
- **Clipped exponential:** `np.clip(arg, -20, 20)` prevents overflow.
  Numerically safe at all voltages.

## Infrastructure Pipeline

```
BrainScaleSAdExNeuron
├── step(current: float) → int {0,1}
├── reset() → v=v_rest, w=0
├── In Population: standard scalar current interface
├── In Network: all stimuli and monitors
│   ├── PoissonInput (weight=40, rate=500Hz for reliable spiking)
│   ├── Projection + STDP compatible
│   └── SpikeMonitor, StateMonitor
├── Analysis: all spike_stats functions
├── SC encoding: spike train → rate coding
├── Verilog: compilable (AdEx with clipped exp LUT, ~80 LUTs)
└── Rust NetworkRunner: supported
```

## Wiring Plan

```
PoissonInput(weight=40, rate=500Hz)
    ↓ mean current ~20 (above spiking threshold)
Population(BrainScaleSAdExNeuron, n=N)
    ↓ binary spike vector (strong SFA — few spikes per neuron)
Projection(pop, pop, weight=2.0, probability=0.3)
SpikeMonitor → spike_trains → analysis
```

## Performance

| Metric | Python (NumPy) | Rust engine |
|--------|---------------|-------------|
| Isolation | 237 Ksteps/s | Not measured |
| Network (20 neurons, 500ms) | ~190 Kneuron-steps/s | Expected ~40× faster |
| Spiking threshold | I ≥ 20 | — |

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 8 | construction, step binary, subthreshold at I=5, spikes at I=20, adaptation, exp clip safety, state finiteness, reset |
| Network | 3 | Population, spike production, Projection |
| Analysis | 3 | firing_rate, spike_count, ISI |
| **Total** | **14** | |

See `tests/test_model_brainscales_adex.py`.
