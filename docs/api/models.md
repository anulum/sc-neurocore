# Models

Pre-built network architectures and model registry.

- `SCDigitClassifier` — Pre-configured SC network for MNIST digit classification. Architecture: Conv (28x28, 1ch->4ch, kernel 3, stride 2) + Vectorized dense + output.
- [`SCIzhikevichNeuron`](models/izhikevich.md) — maintained Izhikevich software neuron reference with explicit baseline/RK4 integrator paths, cross-language RK4 parity, and latest benchmark evidence.

10 pre-built configurations in the model zoo:

| Config | Task |
|--------|------|
| Brunel balanced | E/I balance dynamics |
| Cortical column | Layered cortical model |
| CPG | Central pattern generator |
| Decision-making | 2-pool WTA |
| Working memory | Persistent activity |
| Visual cortex V1 | Orientation selectivity |
| MNIST classifier | Digit recognition |
| SHD classifier | Speech (Spiking Heidelberg Digits) |
| DVS gesture | Event camera gestures |
| Auditory | Sound processing |

```python
from sc_neurocore.models import SCDigitClassifier

model = SCDigitClassifier()
output = model.forward(image_28x28)
```

::: sc_neurocore.models.zoo
    options:
      show_root_heading: true
