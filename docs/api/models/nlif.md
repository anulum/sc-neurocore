# NonlinearLIFNeuron

**Module:** `sc_neurocore.neurons.models.nlif`
**Reference:** Touboul & Brette 2008
**Family:** Integrate-and-fire (nonlinear, 2D)
**State variables:** `v` (voltage), `w` (adaptation current)

## Equations

$$C \frac{dV}{dt} = a(V - V_r)(V - V_c) - w + I$$
$$\tau_w \frac{dw}{dt} = b(V - V_r) - w$$

Spike: $V \geq V_\theta$, hard reset $V \to V_{reset}$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `a` | 0.04 | Quadratic nonlinearity coefficient |
| `v_rest` | -65.0 | Resting potential (mV) |
| `v_crit` | -40.0 | Critical voltage — cubic inflection point (mV) |
| `v_threshold` | -20.0 | Spike threshold (mV) |
| `v_reset` | -65.0 | Post-spike reset (mV) |
| `b` | 0.5 | Subthreshold adaptation coupling |
| `tau_w` | 100.0 | Adaptation time constant (ms) |
| `c_m` | 1.0 | Membrane capacitance |
| `dt` | 0.1 | Integration step (ms) |

## Behaviour

- **Cubic nonlinearity:** $a(V-V_r)(V-V_c)$ is negative for $V_r < V < V_c$
  (stable) and positive for $V > V_c$ (runaway → spike). This creates
  a clear excitability threshold at $V_c$.
- **Subthreshold adaptation:** w tracks voltage via b and provides
  negative feedback, producing spike-frequency adaptation.
- **Touboul & Brette 2008:** Generalisation of Izhikevich — with
  specific (a, b, V_c) values, can reproduce AdEx, QIF, EIF behaviour.
- **Hard reset:** V jumps to V_reset on spike.

## Infrastructure Pipeline

```
NonlinearLIFNeuron
├── step(current) → int {0,1}
├── Population: PoissonInput(weight=20, rate=500Hz)
├── Verilog: quadratic + adaptation, ~40 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, cubic above V_crit, w adaptation, rate increase, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **13** | |
