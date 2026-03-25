# Tutorial 68: Homeostatic Network Regulation

Self-stabilizing SNN: deploy and forget.

## Network Regulator

```python
from sc_neurocore.homeostasis import NetworkRegulator

reg = NetworkRegulator(target_rate=0.1, threshold_step=0.01)

# In training loop:
new_thresholds, new_lr, metrics = reg.regulate(
    firing_rates, thresholds, learning_rate, weights=model_weights,
)
print(metrics.summary())
```

## Sleep Consolidation

```python
from sc_neurocore.homeostasis import SleepConsolidation

sleep = SleepConsolidation(decay_exponent=0.5, duration_fraction=0.1)

for epoch in range(100):
    train_one_epoch()
    if sleep.should_sleep(epoch, 100):
        model_weights = sleep.apply(model_weights)
```
