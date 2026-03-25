# Hardware-Aware SNN NAS Guide

Find the best SNN architecture for your FPGA target automatically.

## Quick Start

```python
from sc_neurocore.nas import SearchSpace, nas

# Define search space
space = SearchSpace(
    n_inputs=784,       # MNIST input
    n_outputs=10,       # 10 classes
    min_layers=1,
    max_layers=3,
)

# Run NAS targeting iCE40 FPGA
result = nas(space, target="ice40", population_size=50, generations=20)

# Best accuracy on Pareto front
best = result.best_accuracy()
print(f"Best: {best.layer_widths}, L={best.bitstream_lengths}")
print(f"Accuracy: {best.fitness_accuracy:.3f}, LUTs: {best.fitness_luts}")

# Best energy efficiency
efficient = result.best_efficiency()
print(f"Efficient: {efficient.layer_widths}, E={efficient.fitness_energy_nj:.1f} nJ")

# Full Pareto front
print(result.summary())
```

## Custom Accuracy Function

By default, NAS uses a capacity proxy. For real accuracy, provide a training function:

```python
def train_and_eval(arch):
    """Train an SNN with the given architecture, return test accuracy."""
    # Build network from arch.layer_widths, arch.neuron_types, etc.
    # Train on your dataset
    # Return accuracy in [0, 1]
    return accuracy

result = nas(space, target="artix7", accuracy_fn=train_and_eval)
```

## Search Space Configuration

```python
space = SearchSpace(
    n_inputs=128,
    n_outputs=11,
    min_layers=1,
    max_layers=4,
    width_choices=[16, 32, 64, 128],
    neuron_choices=["StochasticLIFNeuron", "FixedPointLIFNeuron"],
    L_choices=[32, 64, 128, 256],
    delay_choices=[0, 1, 2, 4],
)

# Approximate search space size
print(f"Search space: ~{space.space_size:,} architectures")
```

## FPGA Targets

| Target | LUTs | Use Case |
|--------|------|----------|
| ice40 | 7,680 | Ultra-low-power edge |
| ecp5 | 83,640 | Mid-range |
| artix7 | 63,400 | Research boards |
| zynq | 53,200 | SoC with ARM |

## Formal Equivalence Verification

After NAS selects an architecture, verify that the generated Verilog
matches the Python simulation exactly:

```python
from sc_neurocore.nas.equiv import check_equivalence

# Check without running (generates proof files)
result = check_equivalence(run=False)
print(result.summary())

# Run with SymbiYosys (requires sby + z3 installed)
result = check_equivalence(run=True, depth=30)
print(result.summary())  # "PROVED" or "FAILED"
```

The miter circuit drives both models with ALL possible inputs via
bounded model checking. A depth-30 proof covers 30 clock cycles
with arbitrary inputs — stronger than any test suite.
