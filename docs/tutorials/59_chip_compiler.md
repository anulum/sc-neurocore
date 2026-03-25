# Tutorial 59: Multi-Chip Hardware Compiler

Compile your SNN to any neuromorphic chip.

## Supported Targets

| Chip | Vendor | Cores | Neurons | Weight Bits | On-Chip Learning |
|------|--------|-------|---------|-------------|------------------|
| Loihi 2 | Intel | 128 | 16,384 | 8 | STDP, R-STDP, e-prop |
| Xylo | SynSense | 1 | 1,000 | 8 | No |
| Speck | SynSense | 1 | 32,768 | 4 | No |
| Akida | BrainChip | 80 | 20,480 | 4 | STDP |
| SpiNNaker2 | Manchester | 152 | 155,648 | 16 | STDP, custom |
| BrainScaleS-2 | Heidelberg | 1 | 512 | 6 | STDP (analog) |

## Quick Start

```python
from sc_neurocore.chip_compiler import compile_for_chip
import numpy as np

# Compile a simple network to Loihi 2
result = compile_for_chip(
    layer_sizes=[(784, 256), (256, 10)],
    weights=[np.random.randn(256, 784) * 0.1, np.random.randn(10, 256) * 0.1],
    neuron_types=["LIF", "LIF"],
    target="loihi2",
)

print(result.summary())
# Compilation [loihi2]: SUCCESS
#   Cores: 3
#   Neurons: 266
#   Weight precision: 8-bit
```

## Custom Chip Specs

```python
from sc_neurocore.chip_compiler import ChipSpec, CoreSpec, compile_for_chip

my_chip = ChipSpec(
    name="my_asic",
    vendor="My Lab",
    total_cores=16,
    core=CoreSpec(
        max_neurons=512,
        max_synapses_per_neuron=2048,
        weight_bits=8,
        supported_neuron_types=["LIF", "ALIF"],
        has_on_chip_learning=True,
        learning_rules=["STDP"],
    ),
)

result = compile_for_chip([(128, 64)], target=my_chip)
```

## Constraint Checking

The compiler validates:
- Total neuron count vs chip capacity
- Neuron type compatibility per layer
- Fan-out limits
- Core partitioning (auto-splits layers across cores)
- Weight quantization with precision loss warnings
- Analog noise warnings (BrainScaleS-2)
