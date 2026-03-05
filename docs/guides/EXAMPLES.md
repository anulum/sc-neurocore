# sc-neurocore Examples

This document provides runnable examples for the most common modules. Each example is intentionally small and can be copied into a script. Use these as starting points for experiments or for validating that your environment is working.

All examples assume you have activated a virtual environment and installed numpy. If a snippet uses matplotlib, install it separately.

---

## Example 1: Single stochastic LIF neuron

```python
import numpy as np
from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

neuron = StochasticLIFNeuron(tau_mem=20.0, dt=1.0, v_threshold=1.0, noise_std=0.02)

spikes = []
for _ in range(100):
    spikes.append(neuron.step(0.15))

print("Total spikes:", sum(spikes))
```

What to expect: a small number of spikes. Increase input current or lower threshold to increase spiking.

---

## Example 2: Dense layer with shared current source

```python
import numpy as np
from sc_neurocore.layers.sc_dense_layer import SCDenseLayer

layer = SCDenseLayer(
    n_neurons=4,
    x_inputs=[0.2, 0.8],
    weight_values=[0.6, 0.4],
    x_min=0.0,
    x_max=1.0,
    w_min=0.0,
    w_max=1.0,
    length=128,
    neuron_params={"noise_std": 0.0}
)

layer.run(50)
spikes = layer.get_spike_trains()
print("Spike matrix shape:", spikes.shape)
print("Summary:", layer.summary())
```

---

## Example 3: Learning layer with STDP

```python
import numpy as np
from sc_neurocore.layers.sc_learning_layer import SCLearningLayer

layer = SCLearningLayer(n_inputs=3, n_neurons=2, learning_rate=0.05, length=64)

for epoch in range(5):
    spikes = layer.run_epoch([0.2, 0.5, 0.8])
    print("Epoch", epoch, "spike sum", spikes.sum())

print("Weights:")
print(layer.get_weights())
```

Use a larger length for more stable learning. If you want deterministic behavior, set numpy seed before creating the layer.

---

## Example 4: Vectorized layer for speed

```python
import numpy as np
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

layer = VectorizedSCLayer(n_inputs=8, n_neurons=32, length=256)
output = layer.forward([0.1] * 8)
print("Output shape:", output.shape)
print("Output range:", output.min(), output.max())
```

This layer is much faster than bitwise loops because it uses packed bitstreams and vectorized operations.

---

## Example 5: Convolutional layer

```python
import numpy as np
from sc_neurocore.layers.sc_conv_layer import SCConv2DLayer

layer = SCConv2DLayer(in_channels=1, out_channels=2, kernel_size=3)
image = np.random.random((1, 8, 8))
output = layer.forward(image)
print("Output shape:", output.shape)
```

---

## Example 6: Recurrent layer

```python
import numpy as np
from sc_neurocore.layers.recurrent import SCRecurrentLayer

layer = SCRecurrentLayer(n_inputs=3, n_neurons=5, seed=1)
for t in range(10):
    state = layer.step(np.array([0.2, 0.4, 0.6]))
    print("t", t, "mean state", state.mean())
```

---

## Example 7: DVS input processing

```python
import numpy as np
from sc_neurocore.interfaces.dvs_input import DVSInputLayer

layer = DVSInputLayer(height=4, width=4)
# events: (x, y, time_ms, polarity)
events = [(1, 1, 10.0, 1), (2, 2, 10.5, 1)]
probs = layer.process_events(events)
print("Surface mean:", probs.mean())

bits = layer.generate_bitstream_frame(length=16)
print("Bitstream shape:", bits.shape)
```

---

## Example 8: BCI decoder

```python
import numpy as np
from sc_neurocore.interfaces.bci import BCIDecoder

decoder = BCIDecoder(channels=2)
# two channels, 5 time samples
signal = np.array([[0.1, 0.2, 0.3, 0.2, 0.1],
                   [0.3, 0.4, 0.5, 0.4, 0.3]])
bitstreams = decoder.encode_to_bitstream(signal, length=32)
print(bitstreams.shape)
```

---

## Example 9: ONNX export (JSON schema)

```python
import numpy as np
from sc_neurocore.export.onnx_exporter import SCOnnxExporter
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

layer1 = VectorizedSCLayer(n_inputs=4, n_neurons=8, length=64)
layer2 = VectorizedSCLayer(n_inputs=8, n_neurons=2, length=64)
SCOnnxExporter.export([layer1, layer2], "sc_model.json")
```

This produces `sc_model.json` and sidecar `.npy` weight files.

---

## Example 10: Exotic and meta modules

```python
import numpy as np
from sc_neurocore.meta.time_crystal import TimeCrystalLayer
from sc_neurocore.exotic.chemical import ReactionDiffusionSolver

crystal = TimeCrystalLayer(n_spins=4)
print("Time crystal bitstream:", crystal.get_bitstream(cycles=8))

chem = ReactionDiffusionSolver(width=16, height=16)
for _ in range(10):
    chem.step()
print("Reaction diffusion state mean:", chem.get_state().mean())
```

---

## Notes

- Many examples rely on randomness. Set `np.random.seed(...)` for reproducible runs.
- If you plan to benchmark, enable `SC_NEUROCORE_PERF=1` and run the performance tests.
- For hardware deployment, see HARDWARE_GUIDE.md.
