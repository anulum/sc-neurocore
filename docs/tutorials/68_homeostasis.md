# Tutorial 68: Homeostatic Network Regulation

Self-stabilising SNNs that maintain healthy activity levels without
manual tuning. Homeostasis adjusts thresholds, learning rates, and
synaptic weights to keep firing rates in a target range — even as
inputs change or the network learns new tasks.

The regulator provides bounded feedback for training or deployment loops.

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
    rate_tolerance=0.5,    # acceptable population-rate band around target
    threshold_step=0.01,   # how much to adjust thresholds per step
    lr_scale_factor=0.95,  # high variance multiplies LR by this factor
)

# Simulated network state
rng = np.random.default_rng(42)
n_neurons = 128
firing_rates = rng.random(n_neurons).astype(np.float32) * 0.2 + 0.16
thresholds = np.ones(n_neurons, dtype=np.float32)
learning_rate = 0.001
model_weights = [rng.standard_normal((64, 128)).astype(np.float32) * 0.1]

# One regulation step
new_thresholds, new_lr, metrics = reg.regulate(
    firing_rates, thresholds, learning_rate, weights=model_weights,
)

print(metrics.summary())
# Network Stability: UNSTABLE
#   Mean firing rate: 0.2578
#   Rate variance: 0.0030
#   E/I ratio: 1.00
#   Weight norm: 9.0837
#   Adjustments: thresholds +0.010
```

### How It Works

Three regulation mechanisms, matching biology:

1. **Threshold homeostasis:** Populations above the target band get higher
   thresholds. Populations below the target band get lower thresholds.
   ```
   thresholds += threshold_step        # overactive population
   thresholds -= threshold_step        # quiet population
   ```

2. **Learning rate scaling:** If the network is unstable (high rate
   variance), the learning rate is reduced to slow weight changes.

3. **Weight monitoring:** Optional weight matrices are validated and summarised
   as a mean norm in the returned metrics. Direct synaptic scaling lives in the
   sleep-consolidation path below.

## Sleep Consolidation

Biological brains consolidate memories during sleep by pruning weak
synapses (synaptic homeostasis hypothesis, Tononi & Cirelli 2003).
SC-NeuroCore's sleep module implements this:

```python
from sc_neurocore.homeostasis import SleepConsolidation

sleep = SleepConsolidation(
    decay_exponent=0.5,       # power-law synapse pruning
    noise_amplitude=0.01,     # spontaneous replay noise
    duration_fraction=0.1,    # sleep duration as fraction of training
)

for epoch in range(100):
    # Normal training
    # train_one_epoch(model, data)

    # Check if it's time to sleep
    if sleep.should_sleep(epoch, total_epochs=100):
        model_weights = sleep.apply(model_weights, seed=epoch)
        print(f"Epoch {epoch}: sleep consolidation applied")
        print(f"  Consolidated {len(model_weights)} weight arrays")
```

### Sleep Schedule

The consolidation interval is derived from `duration_fraction`:

```
interval = max(1, int(1.0 / duration_fraction))
```

With `duration_fraction=0.1`, sleep runs at positive epoch multiples of 10
(`10`, `20`, `30`, ...).

Each sleep cycle applies power-law decay to weak synapses:
```
relative = abs(w_old) / max(abs(w_old))
decay_factor = clip(1 - duration_fraction * relative^decay_exponent, 0.5, 1.0)
w_new = w_old * decay_factor + replay_noise
```

Larger-magnitude synapses receive proportionally stronger down-scaling while
small weights are relatively preserved. The optional replay noise is
deterministic for a validated seed.

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
