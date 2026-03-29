# AdaptiveThresholdIFNeuron

**Module:** `sc_neurocore.neurons.models.adaptive_threshold_if`
**Reference:** Platkiewicz & Bhatt 2010
**Family:** Integrate-and-Fire (adaptive)
**State variables:** `v` (membrane voltage), `theta` (dynamic threshold)

## Equations

$$C \frac{dV}{dt} = -g_L(V - V_{\text{rest}}) + I$$
$$\frac{d\theta}{dt} = -\frac{\theta - \theta_{\text{rest}}}{\tau_\theta}$$

Spike when $V \geq \theta$. Reset: $V \to V_{\text{reset}}$, $\theta \to \theta + \Delta\theta$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | -65.0 | Initial membrane voltage (mV) |
| `theta` | -50.0 | Initial threshold (mV) |
| `v_rest` | -65.0 | Resting potential (mV) |
| `v_reset` | -65.0 | Post-spike reset voltage (mV) |
| `theta_rest` | -50.0 | Resting threshold (mV) |
| `delta_theta` | 5.0 | Threshold increment per spike (mV) |
| `tau_m` | 10.0 | Membrane time constant (ms) |
| `tau_theta` | 50.0 | Threshold adaptation time constant (ms) |
| `dt` | 0.1 | Timestep (ms) |

## Behaviour

- **Spike frequency adaptation:** Each spike raises the threshold by
  `delta_theta`. The threshold then decays back to `theta_rest` with
  time constant `tau_theta`. This produces decreasing firing rate
  under constant input (adaptation).
- **Refractory effect:** Immediately after a spike, the threshold is
  elevated, making the next spike harder to achieve.

## Network Usage

```python
from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor

pop = Population(AdaptiveThresholdIFNeuron, n=20, label="atif")
drive = PoissonInput(n=20, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
mon = SpikeMonitor(pop)
net = Network(pop, drive, mon)
net.run(duration=0.5, dt=0.001)
print(f"Spikes: {mon.count}")
```

**Drive requirements:** This model operates in mV with `tau_m=10`,
`dt=0.1`. The gap from rest (-65) to threshold (-50) is 15 mV.
At steady state, `I_required ≈ 15 × tau_m / dt = 1500` for one spike
per model step. In a Network with `dt=0.001`, PoissonInput
`weight=100, rate_hz=500` provides sufficient drive.

## Test Coverage

- Isolation: construction, step returns binary, spikes under drive,
  threshold adaptation, state finiteness, reset
- Network: Population creation, spike production, spike train extraction
- Analysis: firing_rate, spike_count, ISI

See `tests/test_model_adaptive_threshold_if.py` (12 tests).
