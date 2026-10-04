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

The native class accepts all eleven model parameters. Current, resting and
reset voltages stay inside `[-200, 100]`; energy stays in `[0, epsilon_0]`.
All parameters are finite, time constants, resistance and `dt` are positive,
`dt` does not exceed either time constant, and threshold exceeds both resting
and reset voltage. Reset validates its complete candidate before committing
either state, including after edits to the Python dataclass configuration.
Valid reset can recover invalid dynamic state.

The installed native batch reads aligned contiguous float64 input before
allocating owning float64 voltage/energy arrays and an int32 event array.
Allocation failure raises `MemoryError`, releases partial outputs and permits
a subsequent batch. Invalid layouts retain their authored `TypeError`.
Class-global pickle identity remains stable; native instances retain explicit
pickle/copy refusal. Go, Mojo and Julia validate configuration for empty batches
as well as nonempty traces. Mojo uses `expm1` for the coupled exponential term
and `1 + expm1(-dt/tau)` for decay over the validated `0 < dt/tau <= 1` range.
This preserves the existing `2e-12` envelope near equal time constants and at
the maximum permitted timestep.

Dedicated configured tests cover positional/keyword construction, empty finals,
all trace samples, error atomicity, reset continuation, and five real runtimes.
Linux allocation-pressure proof covers each output and layout-before-allocation;
it does not establish qualification for other platform allocator profiles.
