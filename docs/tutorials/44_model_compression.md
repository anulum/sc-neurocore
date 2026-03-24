<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 44: SNN Model Compression

Reduce FPGA resource usage through pruning, quantization, and automatic
optimization. Make networks fit on smaller, cheaper FPGAs.

## 1. Weight Pruning

```python
import numpy as np
from sc_neurocore.compression import prune_weights

weights = [np.random.randn(16, 8) * 0.3]
pruned, report = prune_weights(weights, threshold=0.1)
print(f"Sparsity: {report.sparsity:.1%}")
```

## 2. Structural Pruning

```python
from sc_neurocore.compression import prune_neurons

pruned, report = prune_neurons(weights, activity_threshold=0.05)
print(f"Removed {report.pruned_neurons} neurons")
```

## 3. Weight Quantization

```python
from sc_neurocore.compression.quantization import quantize_weights
q8 = quantize_weights(weights, bits=8)
```

## 4. Delay Quantization

```python
from sc_neurocore.compression.quantization import quantize_delays
delays = np.array([1.3, 2.7, 4.1])
quantized = quantize_delays(delays, resolution=2)
```

## 5. Auto-Optimize for Target

```python
from sc_neurocore.optimizer import fit_to_target
result = fit_to_target([(32, 16), (16, 8)], weights, target="ice40")
print(result.summary())
```

## Further Reading

- [API: Compression](../api/compression.md)
- [API: Optimizer](../api/optimizer.md)
