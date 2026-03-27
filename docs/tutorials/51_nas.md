# Tutorial 51: Hardware-Aware SNN NAS

Neural Architecture Search (NAS) finds the optimal SNN architecture
for your FPGA automatically. SC-NeuroCore's NAS uses NSGA-II
evolutionary search to explore neuron types, layer widths, bitstream
lengths, and delays — under strict FPGA resource constraints.

## Why Hardware-Aware NAS

Manual architecture design is guesswork. You pick a hidden size,
train for hours, then discover it doesn't fit on your FPGA. NAS
automates this: search the architecture space while respecting hardware
constraints. Every candidate is evaluated for both accuracy *and*
resource cost.

| Manual Design | Hardware-Aware NAS |
|:------------:|:-----------------:|
| Pick architecture by intuition | Explore ~100K candidates |
| Train → check fit → repeat | Accuracy + resource co-optimisation |
| Single objective (accuracy) | Pareto front (accuracy vs cost) |
| Hours of iteration | Automated overnight |

## Step 1: Define Search Space

```python
from sc_neurocore.nas import SearchSpace

space = SearchSpace(
    n_inputs=784,           # MNIST
    n_outputs=10,
    min_layers=1,
    max_layers=3,
    width_choices=[16, 32, 64, 128, 256],
    L_choices=[32, 64, 128, 256],         # bitstream lengths
    neuron_choices=["LIF", "Izhikevich", "HomeostaticLIF", "FixedPointLIF"],
)

print(f"Search space: ~{space.space_size:,} architectures")
# ~100,000+ candidates
```

## Step 2: Run NAS

NSGA-II optimises two objectives simultaneously:
1. **Maximise accuracy** (proxy or real training)
2. **Minimise FPGA cost** (LUTs + BRAMs + energy)

```python
from sc_neurocore.nas import nas

result = nas(
    space,
    target="ice40",          # iCE40 UP5K (5,280 LUTs)
    population_size=50,      # 50 candidates per generation
    generations=20,          # 20 generations
    seed=42,
)

print(result.summary())
# NAS completed: 20 generations, 1000 evaluations
# Pareto front: 12 non-dominated solutions
# Best accuracy: 96.2% (2,340 LUTs, 98 nJ)
# Most efficient: 92.1% (890 LUTs, 31 nJ)
```

## Step 3: Inspect the Pareto Front

The Pareto front shows the accuracy-vs-cost trade-off:

```python
for i, arch in enumerate(result.pareto_front):
    print(f"{i+1}. Accuracy={arch.fitness_accuracy:.1%}, "
          f"LUTs={arch.fitness_luts:,}, "
          f"Energy={arch.fitness_energy_nj:.0f} nJ, "
          f"Layers={arch.layer_widths}")

# 1. Accuracy=96.2%, LUTs=2,340, Energy=98 nJ, Layers=[128, 64]
# 2. Accuracy=95.8%, LUTs=1,680, Energy=72 nJ, Layers=[64, 64]
# 3. Accuracy=95.1%, LUTs=1,200, Energy=51 nJ, Layers=[64, 32]
# ...
# 12. Accuracy=92.1%, LUTs=890,   Energy=31 nJ, Layers=[32]
```

## Step 4: Deploy the Winner

```python
best = result.best_accuracy()
print(f"Architecture: {best.layer_widths}")
print(f"Neuron types: {best.neuron_types}")
print(f"Bitstream lengths: {best.bitstream_lengths}")
print(f"LUTs: {best.fitness_luts:,}")
print(f"Energy: {best.fitness_energy_nj:.1f} nJ")

# Verify with energy estimator
from sc_neurocore.energy import estimate
report = estimate(
    best.layer_sizes, target="ice40",
    bitstream_length=best.bitstream_lengths[0],
)
print(report.summary())
```

## Custom Accuracy Function

Replace the built-in proxy with real SNN training:

```python
def my_accuracy(arch):
    """Train and evaluate an SNN with the given architecture."""
    from sc_neurocore.training import SpikingNet, train_epoch, evaluate
    model = SpikingNet(
        n_input=arch.n_inputs,
        n_hidden=arch.layer_widths[0],
        n_output=arch.n_outputs,
    )
    # Train for 5 epochs (fast, for NAS inner loop)
    for _ in range(5):
        train_epoch(model, train_loader, optimizer, n_timesteps=25)
    _, acc = evaluate(model, test_loader, n_timesteps=25)
    return acc

result = nas(space, target="artix7", accuracy_fn=my_accuracy, generations=10)
```

## Search Dimensions

| Dimension | Options | Trade-off |
|-----------|---------|-----------|
| Layer width | 8–256 | Capacity ↔ LUTs |
| Neuron type | LIF, Izhikevich, Homeostatic, FixedPoint | Biology ↔ resources |
| Bitstream length | 32–512 | Precision ↔ latency |
| Delay range | 0–8 cycles | Temporal processing ↔ BRAM |
| Number of layers | 1–3 | Depth ↔ total resources |

## Comparison

| Feature | SC-NeuroCore NAS | NAS-SNN (Kim) | SpikeDHS |
|---------|:---------------:|:------------:|:--------:|
| Hardware-aware | Yes (FPGA) | No | Yes (Loihi) |
| Multi-objective | Yes (NSGA-II) | Single | Yes |
| Neuron type search | Yes | No | No |
| Bitstream length search | Yes | No | No |
| FPGA resource constraint | Yes | No | No |

## References

- Kim et al. (2022). "Neural Architecture Search for Spiking Neural
  Networks." ECCV 2022.
- Che et al. (2022). "Differentiable Hierarchical Search for SNNs."
  IEEE TNNLS.
- Deb et al. (2002). "A Fast and Elitist Multiobjective Genetic
  Algorithm: NSGA-II." IEEE TEC 6(2):182-197.
