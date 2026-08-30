# SCInclusivePerfectIntegratorNeuron

**Module:** `sc_neurocore.neurons.models.perfect_integrator`

This count-neutral project identity preserves SC-NeuroCore's historical
perfect-integrator recurrence with the inclusive
`candidate >= v_threshold` event boundary. It remains available so existing SC
experiments retain their exact behavior without attributing that comparator to
the Naud-Gerstner source equation.

```python
from sc_neurocore.neurons.models import SCInclusivePerfectIntegratorNeuron

neuron = SCInclusivePerfectIntegratorNeuron()
voltage, events = neuron.simulate_complete(1_000, 5.0, backend="auto")
assert events.sum() == 500
```

For a held current sample $I_n$, the state transition is the exact integral

$$V_{n+1}=V_n+\frac{I_n\,\Delta t}{C}.$$

If the candidate equals or exceeds `v_threshold`, the model emits and resets to
`v_reset`. The paired `sc_perfect_integrator.toml`/JSON schemas, all five native
runtime lanes, and the dedicated Q8.8 `sc_perfect_integrator` RTL/formal lane
retain this contract. The compatibility identity does not add to the literature
model count.

For the counted source identity, strict `candidate > v_threshold` semantics,
DOI-bound receipt, controlled benchmark, and source RTL, see
[PerfectIntegratorNeuron](perfect_integrator.md).
