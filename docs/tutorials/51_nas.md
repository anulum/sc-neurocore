# Tutorial 51: Hardware-Aware SNN NAS

Find the optimal SNN architecture for your FPGA automatically.

## What You'll Learn

- Define a search space over neuron types, widths, bitstream lengths, and delays
- Run NSGA-II evolutionary search under FPGA resource constraints
- Interpret the Pareto front: accuracy vs hardware cost

## Step 1: Define Search Space

```python
from sc_neurocore.nas import SearchSpace

space = SearchSpace(
    n_inputs=784,
    n_outputs=10,
    min_layers=1,
    max_layers=3,
    width_choices=[16, 32, 64, 128],
    L_choices=[32, 64, 128, 256],
)
print(f"Space size: ~{space.space_size:,} architectures")
```

## Step 2: Run NAS

```python
from sc_neurocore.nas import nas

result = nas(
    space,
    target="ice40",          # 7,680 LUTs
    population_size=50,
    generations=20,
    seed=42,
)
print(result.summary())
```

## Step 3: Inspect Results

```python
best = result.best_accuracy()
print(f"Layers: {best.layer_widths}")
print(f"Neurons: {best.neuron_types}")
print(f"Bitstream lengths: {best.bitstream_lengths}")
print(f"LUTs: {best.fitness_luts}")
print(f"Energy: {best.fitness_energy_nj:.1f} nJ")
```

## Step 4: Deploy the Winner

```python
from sc_neurocore.energy import estimate

report = estimate(best.layer_sizes, target="ice40",
                  bitstream_length=best.bitstream_lengths[0])
print(report.summary())
```

## Custom Accuracy Function

Replace the built-in proxy with real training:

```python
def my_accuracy(arch):
    # Build and train an SNN with arch.layer_widths, arch.neuron_types
    # Return test accuracy in [0, 1]
    ...

result = nas(space, target="artix7", accuracy_fn=my_accuracy)
```

## Search Dimensions

| Dimension | Options | Effect |
|-----------|---------|--------|
| width | 8–256 neurons | Capacity vs LUTs |
| neuron_type | LIF, Izhikevich, Homeostatic, FixedPoint | Biological fidelity vs resources |
| bitstream_length | 32–512 | Precision vs latency |
| delay_range | 0–8 cycles | Temporal processing vs BRAM |
