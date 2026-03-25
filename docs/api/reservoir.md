# Reservoir Computing

Liquid State Machine with mean-field auto-criticality tuning.

- `AutoCriticalReservoir` — Self-tuning reservoir that maintains edge-of-chaos dynamics via mean-field theory. Spectral radius and input scaling adjusted automatically to maximize computational capacity.

Reservoir computing requires no training of recurrent weights — only a readout layer is trained. The reservoir provides a high-dimensional nonlinear projection of the input.

```python
from sc_neurocore.reservoir import AutoCriticalReservoir
```

See [Tutorial 74: Reservoir Computing](../tutorials/74_reservoir.md).

::: sc_neurocore.reservoir
    options:
      show_root_heading: true
