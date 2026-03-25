# Optics

Photonic spiking neuron layer for optical neuromorphic computing.

- `PhotonicBitstreamLayer` — Simulates photonic SC layer using laser interference. Phase noise from two coherent beams generates stochastic bitstreams. `laser_power` controls SNR. Supports weighted optical summation and threshold detection.

Photonic SC uses physical randomness (laser phase noise) instead of LFSR pseudo-randomness, giving truly uncorrelated bitstreams.

```python
from sc_neurocore.optics import PhotonicBitstreamLayer

layer = PhotonicBitstreamLayer(n_channels=8, laser_power=1.0)
interference = layer.simulate_interference(length=1024)
```

::: sc_neurocore.optics.photonic_layer
    options:
      show_root_heading: true
