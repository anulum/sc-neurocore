# Tutorial 74: Auto-Critical Reservoir Computing

Zero-hyperparameter Liquid State Machine.

```python
from sc_neurocore.reservoir import AutoCriticalReservoir

reservoir = AutoCriticalReservoir(n_inputs=64, n_neurons=1000, n_outputs=10)
predictions = reservoir.train_and_predict(train_x, train_y, test_x)
metrics = reservoir.metrics(test_x)
print(metrics.summary())
```
