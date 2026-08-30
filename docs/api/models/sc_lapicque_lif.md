# SCLapicqueLIFNeuron

**Module:** `sc_neurocore.neurons.models.lapicque`

This count-neutral project identity preserves SC-NeuroCore's historical
repetitive exact-flow, hard-reset LIF recurrence. It exists so compatibility
behaviour is never deleted and is never mistaken for the complete Lapicque
1907 polarization experiment.

```python
from sc_neurocore.neurons.models import SCLapicqueLIFNeuron

neuron = SCLapicqueLIFNeuron(tau=20.0, resistance=1.0, dt=1.0)
voltage, events = neuron.simulate_complete(1_000, 5.0)
assert events.sum() == 200
```

For constant injected current $I$,

$$
v_{n+1}=v_\infty+(v_n-v_\infty)e^{-\Delta t/\tau},\qquad
v_\infty=v_\mathrm{rest}+RI.
$$

If the candidate reaches `v_threshold`, it emits and hard-resets to `v_reset`.
The paired `sc_lapicque_lif.toml`/JSON schemas, all five runtime lanes, and the
existing Q16.16 `sc_lapicque` co-simulation/formal surfaces retain this exact
contract. This identity adds no literature-model count.

For the counted source identity, source voltage input, strength-duration law,
and no-reset first-attainment semantics, see [LapicqueNeuron](lapicque.md).
