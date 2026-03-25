# Tutorial 74: Auto-Critical Reservoir Computing

Zero-hyperparameter Liquid State Machine with mean-field auto-criticality.
The reservoir self-tunes to edge-of-chaos dynamics for maximum computational
capacity. Only the readout layer needs training.

## AutoCriticalReservoir

```python
import numpy as np
from sc_neurocore.reservoir import AutoCriticalReservoir

# Create reservoir: 64 inputs, 1000 recurrent neurons, 10 outputs
reservoir = AutoCriticalReservoir(
    n_inputs=64,
    n_neurons=1000,
    n_outputs=10,
)

# Train readout on data (reservoir weights are fixed)
train_x = np.random.randn(500, 64)
train_y = np.eye(10)[np.random.randint(0, 10, 500)]
test_x = np.random.randn(100, 64)

predictions = reservoir.train_and_predict(train_x, train_y, test_x)

# Criticality metrics
metrics = reservoir.metrics(test_x)
print(metrics.summary())
```

## Why Auto-Criticality

Standard reservoirs require manual tuning of spectral radius, input scaling,
and leak rate. The auto-critical reservoir adjusts these automatically using
mean-field theory to maintain the edge of chaos — the regime where
computational capacity is maximized.

## API Reference

::: sc_neurocore.reservoir
    options:
      show_root_heading: true
