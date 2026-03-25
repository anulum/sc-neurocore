# Delay Training

Trainable per-synapse delays for temporal coding in SNNs.

- `DelayLinear` — Dense layer with learnable per-synapse delays via differentiable interpolation. Each synapse stores a continuous-valued delay parameter. During forward pass, spike history is interpolated at the learned delay offset.

Temporal coding (spike timing) carries more information per spike than rate coding. Trainable delays let the network learn optimal spike timing relationships.

```python
from sc_neurocore.training.delay_linear import DelayLinear
```

See [Tutorial 39: Learnable Delays](../tutorials/39_learnable_delays.md).

::: sc_neurocore.training.delay_linear
    options:
      show_root_heading: true
