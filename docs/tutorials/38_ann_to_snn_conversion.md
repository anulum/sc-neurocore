<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 38: ANN-to-SNN Conversion

SC-NeuroCore converts trained PyTorch ANNs to rate-coded spiking neural
networks. Train with standard PyTorch, convert in one function call, deploy
to FPGA via the SC pipeline.

## Why Convert?

Most practitioners have trained ANNs, not SNNs. Conversion bridges the gap:
use PyTorch's mature ecosystem for training, then convert to an SNN that
runs on neuromorphic hardware with the energy efficiency of spike-based
computation.

## 1. Train a PyTorch ANN (Standard)

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
# Train with standard optimizer...
```

## 2. Convert to SNN

```python
from sc_neurocore.conversion import convert

calibration_data = torch.randn(100, 784)  # representative inputs

snn = convert(
    model,
    calibration_data=calibration_data,
    T=32,  # simulation timesteps (higher = more accurate)
)
print(f"Converted: {snn.n_layers} layers, T={snn.T}")
```

## 3. Run the Converted SNN

```python
import numpy as np

x = np.random.rand(784)  # input in [0, 1]
spike_counts = snn.run(x)
prediction = snn.classify(x)
print(f"Prediction: {prediction}")
```

Batch inference:

```python
x_batch = np.random.rand(100, 784)
predictions = snn.classify(x_batch)
```

## 4. How It Works

The conversion pipeline:

1. **Extract weights** from each `nn.Linear` layer
2. **Calibrate thresholds**: run calibration data through the ANN, record
   per-layer activation statistics (99.9th percentile)
3. **Normalize weights**: scale so max activation maps to firing threshold
4. **Build IF neurons**: each ReLU becomes an integrate-and-fire neuron
   with threshold from calibration
5. **Rate coding**: input values become Poisson spike trains over T steps.
   ANN activation a maps to spike count a*T/threshold.

## 5. QCFS Conversion-Aware Fine-Tuning (Near-Lossless)

For higher accuracy, replace ReLU with QCFS so the ANN trains against the same
quantized activation grid the SNN can reproduce. `convert` then detects the
QCFS layers automatically: it uses each layer's *learned* threshold directly
(no calibration pass) and pre-loads each IF neuron to `theta / 2` — the optimal
shift from Bu et al. 2022 that cancels the quantization flooring bias.

Swap the activations of an already-trained ReLU model and fine-tune:

```python
from sc_neurocore.conversion import replace_relu_with_qcfs, convert

# `model` is a trained ReLU network; substitute QCFS in place, then fine-tune.
replace_relu_with_qcfs(model, T=8)
# ... a few fine-tuning epochs so the learnable thresholds settle ...

model.eval()
snn = convert(model)        # QCFS route: learned thresholds, theta/2 shift,
                            # T inferred from the QCFS layers (here 8)
print(f"Converted: {snn.n_layers} layers, T={snn.T}, "
      f"shift={snn.initial_membrane_fraction}")
```

Or build the network with QCFS activations from the start:

```python
from sc_neurocore.conversion import QCFSActivation

model = nn.Sequential(
    nn.Linear(784, 256),
    QCFSActivation(T=16),
    nn.Linear(256, 10),
)
# Train with QCFS — accuracy is slightly lower than ReLU, but conversion to an
# SNN is nearly lossless at the matching timestep budget.
```

## 6. Deploy to FPGA

After conversion, the SNN weights can be compiled to Verilog:

```bash
# Save weights, then deploy
sc-neurocore deploy model_weights.pt --target artix7 -o build/
```

## Accuracy vs Timesteps

| T (timesteps) | Accuracy | Latency |
|:-:|:-:|:-:|
| 4 | ~85% | Very fast |
| 8 | ~90% | Fast |
| 16 | ~94% | Moderate |
| 32 | ~96% | Slow |
| 64 | ~97% | Very slow |

Higher T improves accuracy at the cost of more clock cycles per inference.
The sweet spot depends on your accuracy-latency tradeoff.

## Further Reading

- [Tutorial 03: Surrogate Gradient Training](03_surrogate_gradient_training.md) — direct SNN training
- [Tutorial 33: Equation-to-Verilog](33_equation_to_verilog.md) — compile to hardware
- [API: Conversion](../api/conversion.md) — auto-generated API docs
