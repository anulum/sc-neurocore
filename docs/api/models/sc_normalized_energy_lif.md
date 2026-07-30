# SCNormalizedEnergyLIFNeuron

`SCNormalizedEnergyLIFNeuron` preserves the project's former `EnergyLIFNeuron` recurrence under an explicit SC identity. Its energy state relaxes exponentially to `epsilon_0`; the voltage uses the exact constant-current solution for energy-modulated gain; a level event resets voltage and subtracts `alpha` from available energy.

This is a count-neutral compatibility model. It has no Fardet-Levina or Sengupta attribution. The frozen 256-step drive `[30, 0, 50, 10] × 64` records three events, final state `(-52.508269792668216, 0.7868689314467242)`, and SHA-256 `29a07937…d12a`.

All five runtimes preserve the event vector and complete two-state trajectory within `2e-12`. Paired schemas and a pinned signed-Q32.32 RTL/Yosys/bounded-safety lane document the maintained H1 boundary; timing, PPA, device evidence, and universal floating-point equivalence remain outside scope.

```python
from sc_neurocore.neurons.models.sc_normalized_energy_lif import (
    SCNormalizedEnergyLIFNeuron,
)

neuron = SCNormalizedEnergyLIFNeuron()
event = neuron.step(30.0)
```
