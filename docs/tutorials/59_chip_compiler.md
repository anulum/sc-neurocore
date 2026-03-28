# Tutorial 59: Multi-Chip Hardware Compiler

Compile your SNN to any neuromorphic chip — Loihi 2, SpiNNaker2,
Akida, BrainScaleS-2, Xylo, or custom ASICs. The compiler handles
core partitioning, weight quantisation, neuron type mapping, and
constraint checking automatically.

## Supported Targets

| Chip | Vendor | Cores | Neurons/Core | Weight Bits | On-Chip Learning |
|------|--------|:-----:|:------------:|:-----------:|:----------------:|
| Loihi 2 | Intel | 128 | 128 | 8 | STDP, R-STDP, e-prop |
| Xylo | SynSense | 1 | 1,000 | 8 | No |
| Speck | SynSense | 1 | 32,768 | 4 | No |
| Akida | BrainChip | 80 | 256 | 4 | STDP |
| SpiNNaker2 | Manchester | 152 | 1,024 | 16 | STDP, custom |
| BrainScaleS-2 | Heidelberg | 1 | 512 | 6 | STDP (analog) |

## Quick Start

```python
from sc_neurocore.chip_compiler import compile_for_chip
import numpy as np

rng = np.random.default_rng(42)

result = compile_for_chip(
    layer_sizes=[(784, 256), (256, 10)],
    weights=[
        rng.standard_normal((256, 784)).astype(np.float32) * 0.1,
        rng.standard_normal((10, 256)).astype(np.float32) * 0.1,
    ],
    neuron_types=["LIF", "LIF"],
    target="loihi2",
)

print(result.summary())
# Compilation [loihi2]: SUCCESS
#   Cores used: 3 / 128
#   Neurons mapped: 266 / 16,384
#   Synapses: 201,480
#   Weight precision: 8-bit (from float32, max quantisation error: 0.004)
#   Estimated power: 12 mW
#   Estimated latency: 0.5 ms per inference
```

## Core Partitioning

Large layers are automatically split across multiple cores:

```python
# 784→256 layer: each Loihi core holds 128 neurons
# → Layer split across 2 cores (128 + 128 neurons)
# → Each core receives all 784 inputs (fan-in broadcast)

# The compiler optimises for minimum inter-core communication
print(f"Core mapping: {result.core_map}")
# {0: [hidden_0:128], 1: [hidden_128:256], 2: [output_0:10]}
```

## Custom Chip Specs

Define your own neuromorphic chip:

```python
from sc_neurocore.chip_compiler import ChipSpec, CoreSpec

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

result = compile_for_chip(
    [(128, 64), (64, 10)],
    weights=my_weights,
    target=my_chip,
)
```

## Constraint Checking

The compiler validates all hardware constraints:

| Constraint | Check | Error if Violated |
|-----------|-------|-------------------|
| Neuron count | Total vs chip capacity | CRITICAL: too many neurons |
| Fan-in | Synapses per neuron vs core limit | CRITICAL: split layer |
| Fan-out | Outputs per neuron vs routing limit | WARNING: may need repeaters |
| Neuron type | Model vs supported types | CRITICAL: unsupported neuron |
| Weight precision | Float → N-bit quantisation loss | WARNING: >1% accuracy loss |
| Core count | Partitioned layers vs total cores | CRITICAL: out of cores |
| Analog noise | BrainScaleS-2 transistor mismatch | WARNING: ~5% weight variation |

## Cross-Chip Comparison

```python
for target in ["loihi2", "akida", "spinnaker2", "xylo"]:
    result = compile_for_chip([(128, 64)], weights=my_weights, target=target)
    if result.success:
        print(f"{target:12s}: {result.cores_used} cores, "
              f"{result.estimated_power_mw:.1f} mW, "
              f"{result.estimated_latency_ms:.2f} ms")
    else:
        print(f"{target:12s}: FAILED — {result.error}")
```

## Integration with Studio

In the Visual Studio:
1. Design network on Canvas
2. Train in Training Monitor
3. Select chip target from dropdown
4. Click **Compile** to see core mapping, constraint checks, and estimates

## Comparison

| Feature | SC-NeuroCore | Lava | snnTorch |
|---------|:-----------:|:----:|:--------:|
| Multi-chip support | 6+ chips | Loihi only | No |
| Auto core partitioning | Yes | Yes (Loihi) | No |
| Custom chip specs | Yes | No | No |
| Constraint checking | 7 checks | Loihi-specific | No |
| Cross-chip comparison | Yes | No | No |

## References

- Davies et al. (2018). "Loihi: A Neuromorphic Manycore Processor with
  On-Chip Learning." IEEE Micro 38(1):82-99.
- Mayr et al. (2019). "SpiNNaker 2: A 10 Billion Neuron System for
  Brain Simulation." arXiv:1911.02385.
- Schemmel et al. (2022). "BrainScaleS-2: A Full-Wafer Mixed-Signal
  Neuromorphic System." NICE 2022.
