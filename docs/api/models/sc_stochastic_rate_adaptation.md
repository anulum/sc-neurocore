# SCStochasticRateAdaptationNeuron

**Module:** `sc_neurocore.neurons.models.sc_stochastic_rate_adaptation`

This count-neutral SC project model preserves the former `BendaHerzNeuron`
behavior: a logistic onset curve, candidate-first RK4 adaptation, and
exponential-hazard Bernoulli spike sampling. It intentionally carries no
Benda-Herz paper attribution.

```python
from sc_neurocore.neurons.models.sc_stochastic_rate_adaptation import (
    SCStochasticRateAdaptationNeuron,
)

neuron = SCStochasticRateAdaptationNeuron(seed=42)
event = neuron.step(10.0)
```

`step_with_uniform(current, uniform)` exposes the same transition using a
controlled variate for exact backend and hardware parity checks.
