# Tutorial 57: Spike-Domain Augmentation & Curriculum Learning

Data augmentation designed for spike trains, plus curriculum scheduling.

## Spike Augmentation

Standard augmentations (flip, rotate) don't work on spike trains.
SpikeAugment provides spike-native transforms:

```python
from sc_neurocore.augmentation import SpikeAugment
import numpy as np

aug = SpikeAugment(
    jitter_steps=2,        # shift spikes +/- 2 timesteps
    dropout_rate=0.1,      # drop 10% of spikes
    bg_noise_rate=0.01,    # 1% background noise
    hot_pixel_prob=0.005,  # 0.5% hot pixel simulation
    seed=42,
)

spikes = np.random.randint(0, 2, (100, 64), dtype=np.int8)
augmented = aug(spikes)
```

## Available Transforms

| Transform | Parameter | Effect |
|-----------|-----------|--------|
| Temporal jitter | `jitter_steps=2` | Shift each spike +/- N steps |
| Spike dropout | `dropout_rate=0.1` | Randomly remove spikes |
| Rate scaling | `rate_scale=(0.8, 1.2)` | Scale firing rates |
| Polarity flip | `polarity_flip_prob=0.5` | Swap ON/OFF channels (DVS) |
| Background noise | `bg_noise_rate=0.01` | Random noise spikes |
| Hot pixel | `hot_pixel_prob=0.005` | Simulate stuck-on sensors |

## Curriculum Learning

Start training on easy patterns, ramp to hard:

```python
from sc_neurocore.augmentation import SpikeCurriculum

curriculum = SpikeCurriculum(
    total_epochs=100,
    start_timesteps=10,     # start with short sequences
    end_timesteps=200,      # end with full-length
    start_rate_scale=2.0,   # amplify rates early (easier to learn)
    end_rate_scale=1.0,     # natural rates at convergence
    start_noise=0.0,        # no noise at start
    end_noise=0.05,         # add noise later (harder)
    warmup_fraction=0.3,    # ramp over first 30% of training
)

for epoch in range(100):
    T = curriculum.timesteps(epoch)
    scale = curriculum.rate_scale(epoch)
    noise = curriculum.noise_rate(epoch)

    # Or apply directly to spike data
    augmented = curriculum.apply_to_spikes(spike_data, epoch=epoch)
    # Train on augmented...

print(curriculum.schedule_summary())
```
