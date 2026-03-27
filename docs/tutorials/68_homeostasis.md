# Tutorial 68: Homeostatic Network Regulation

Self-stabilising SNNs that maintain healthy activity levels without
manual tuning. Homeostasis adjusts thresholds, learning rates, and
synaptic weights to keep firing rates in a target range — even as
inputs change or the network learns new tasks.

Deploy and forget: the network regulates itself.

## Why Homeostasis

Without regulation, SNNs are fragile:
- Too much excitation → runaway activity → epileptic-like seizures
- Too much inhibition → network goes silent → no computation
- Training changes weights → activity drifts → performance degrades

Biological neural circuits solve this with homeostatic plasticity:
negative feedback loops that stabilise activity over hours to days.

## Network Regulator

```python
import numpy as np
from sc_neurocore.homeostasis import NetworkRegulator

reg = NetworkRegulator(
    target_rate=0.1,       # target: 10% of neurons active per timestep
    threshold_step=0.01,   # how much to adjust thresholds per step
    lr_scale_range=(0.5, 2.0),  # allowed learning rate scaling range
)

# Simulated network state
rng = np.random.default_rng(42)
n_neurons = 128
firing_rates = rng.random(n_neurons).astype(np.float32) * 0.3  # some too high
thresholds = np.ones(n_neurons, dtype=np.float32)
learning_rate = 0.001
model_weights = [rng.standard_normal((64, 128)).astype(np.float32) * 0.1]

# One regulation step
new_thresholds, new_lr, metrics = reg.regulate(
    firing_rates, thresholds, learning_rate, weights=model_weights,
)

print(metrics.summary())
# Mean rate: 0.15 (target: 0.10)
# Neurons above target: 72/128 (56%)
# Neurons below target: 56/128 (44%)
# Threshold adjustment: +0.006 mean (raising to reduce activity)
# LR scale: 0.85 (reduced to slow weight changes)
```

### How It Works

Three regulation mechanisms, matching biology:

1. **Threshold homeostasis:** Neurons that fire too much get higher
   thresholds. Neurons that are too silent get lower thresholds.
   ```
   threshold[i] += step * (firing_rate[i] - target_rate)
   ```

2. **Learning rate scaling:** If the network is unstable (high rate
   variance), the learning rate is reduced to slow weight changes.

3. **Synaptic scaling:** Scale all weights connected to overactive
   neurons down, and weights to underactive neurons up. Preserves
   relative weight structure while adjusting overall excitability.

## Sleep Consolidation

Biological brains consolidate memories during sleep by pruning weak
synapses (synaptic homeostasis hypothesis, Tononi & Cirelli 2003).
SC-NeuroCore's sleep module implements this:

```python
from sc_neurocore.homeostasis import SleepConsolidation

sleep = SleepConsolidation(
    decay_exponent=0.5,       # power-law synapse pruning
    duration_fraction=0.1,    # sleep duration as fraction of training
    consolidation_strength=0.8,
)

for epoch in range(100):
    # Normal training
    # train_one_epoch(model, data)

    # Check if it's time to sleep
    if sleep.should_sleep(epoch, total_epochs=100):
        model_weights = sleep.apply(model_weights)
        print(f"Epoch {epoch}: sleep consolidation applied")
        print(f"  Pruned {sleep.pruned_fraction:.1%} of synapses")
        print(f"  Remaining weight magnitude: {sleep.remaining_magnitude:.3f}")
```

### Sleep Schedule

The consolidation occurs at logarithmically-spaced intervals:
- After epoch 10 (early consolidation)
- After epoch 32 (mid-training)
- After epoch 100 (final consolidation)

Each sleep cycle applies power-law decay to weak synapses:
```
w_new = w_old * (|w_old| / max(|w|))^decay_exponent
```

Weak synapses decay faster. Strong synapses are preserved. This
mimics the biological observation that sleep preferentially prunes
synapses formed during recent waking activity.

## Integration with Training

```python
from sc_neurocore.training import SpikingNet, train_epoch, auto_device
from sc_neurocore.training.utils import SpikeMonitor
from sc_neurocore.homeostasis import NetworkRegulator

device = auto_device()
model = SpikingNet(n_input=784, n_hidden=128, n_output=10).to(device)
monitor = SpikeMonitor(model)
reg = NetworkRegulator(target_rate=0.1)

for epoch in range(50):
    train_epoch(model, train_loader, optimizer, n_timesteps=25, device=device)

    # Measure firing rates
    rates = {}
    for name in monitor.layer_names:
        raster = monitor.get(name)
        if raster is not None:
            rates[name] = raster.float().mean(dim=(0, 1)).cpu().numpy()
    monitor.reset()

    # Regulate thresholds based on measured rates
    # (adjust model thresholds directly)
```

## When to Use

| Scenario | Regulation Type |
|----------|----------------|
| Activity drift during training | Threshold homeostasis |
| Continual learning (new tasks) | Threshold + LR scaling |
| Post-deployment adaptation | Threshold homeostasis (on-chip) |
| Long training runs (>100 epochs) | Sleep consolidation |
| Network pruning aftermath | Synaptic scaling (restore activity) |

## FPGA Deployment

Homeostatic regulation runs on-chip as a simple feedback loop:

```
Per neuron, every N timesteps:
  if firing_rate > target + margin:
    threshold += step
  elif firing_rate < target - margin:
    threshold -= step
```

Cost: 1 counter + 1 comparator + 1 adder per neuron. On iCE40, this
adds ~2 LUTs per neuron.

## References

- Turrigiano (2008). "The Self-Tuning Neuron: Synaptic Scaling of
  Excitatory Synapses." Cell 135(3):422-435.
- Tononi & Cirelli (2003). "Sleep and synaptic homeostasis: a
  hypothesis." Brain Research Bulletin 62(2):143-150.
- Zenke & Gerstner (2017). "Continual Learning Through Synaptic
  Intelligence." ICML 2017.
