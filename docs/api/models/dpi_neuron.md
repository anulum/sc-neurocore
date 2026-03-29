# DPINeuron

**Module:** `sc_neurocore.neurons.models.dpi_neuron`
**Reference:** Indiveri et al. 2011
**Family:** Hardware (analog VLSI / DYNAP-SE)
**State variables:** `i_mem` (membrane current, nA)

## Equations

$$\tau \frac{dI_{mem}}{dt} = -I_{mem} + g \cdot I_{syn} + I_{leak}$$

Spike: $I_{mem} \geq I_\theta \Rightarrow I_{mem} \to I_{reset}$.
Current-domain dynamics mirroring subthreshold transistor operation.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `i_mem` | 0.0 | Membrane current (nA) |
| `i_threshold` | 1.0 | Spike threshold (nA) |
| `i_reset` | 0.0 | Post-spike reset (nA) |
| `i_leak` | 0.01 | Leak current (nA) |
| `tau` | 20.0 | Time constant (ms) |
| `gain` | 1.0 | Synaptic gain |
| `dt` | 1.0 | Timestep (ms) |

## Behaviour

- **Current-domain LIF:** All state in current (nA), not voltage (mV).
  Models subthreshold log-domain dynamics of differential-pair integrator
  circuits in neuromorphic VLSI (DYNAP-SE, BrainScaleS).
- **Non-negative:** `i_mem` clamped to ≥ 0 (current cannot be negative
  in the transistor implementation).
- **Simple and fast:** No exp, no transcendentals. 1M+ steps/s.

## Infrastructure Pipeline

```
DPINeuron
├── step(i_syn: float) → int {0,1}
├── reset() → i_mem=0
├── Population: scalar current input (i_syn)
├── Network: PoissonInput(weight=1.5, rate=500Hz)
├── Verilog: trivially compilable (~15 LUTs, adder + comparator)
└── Rust: supported
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | 1.05 Msteps/s | Not measured |
| Network (20, 500ms) | ~800 Kneuron-steps/s | Expected ~50× |
| Spiking threshold | i_syn ≥ 1.0 | — |

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 8 | construction, step binary, subthreshold, spikes, current non-negative, leak, state finite, reset |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **11** | |

No bugs found. Numerically trivial (no transcendentals).
