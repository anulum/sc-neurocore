# Tutorial 69: Multi-Timescale SNNs

Different layers of a biological brain operate at different temporal
resolutions: fast sensory processing (~1ms), medium cognitive
integration (~10ms), slow decision-making (~100ms). SC-NeuroCore's
multi-timescale architecture mimics this with heterogeneous synaptic
time constants and multi-clock layer scheduling.

## Why Multi-Timescale

Standard SNNs run all layers at the same dt. But real signals contain
information at multiple timescales:

| Signal | Fast Component | Slow Component |
|--------|:-------------:|:--------------:|
| Speech | Phonemes (10ms) | Sentences (1s) |
| EEG | Gamma (30Hz) | Theta (4Hz) |
| Motor control | Reflexes (10ms) | Planning (100ms) |
| Video | Motion (33ms) | Scene change (1s) |

A single timestep can't capture all of these. Multi-timescale layers
can.

## HetSyn Layer: Heterogeneous Synaptic Time Constants

Each synapse has its own time constant drawn from a log-normal
distribution (matching Allen Institute cortical neuron data):

```python
import numpy as np
from sc_neurocore.temporal_hierarchy import HetSynLayer

layer = HetSynLayer(
    n_inputs=64,
    n_neurons=32,
    tau_mean=5.0,    # mean synaptic time constant (ms)
    tau_std=1.0,     # log-std for log-normal distribution
)

print(layer.tau_stats)
# {'mean': 5.2, 'std': 4.1, 'min': 0.5, 'max': 42.3}
# Some synapses are fast (0.5ms), others slow (42ms)
# The network automatically learns which timescales matter
```

The log-normal distribution matches biological measurements: most
synapses are fast, a few are very slow. The slow synapses act as
working memory, integrating over longer windows.

## Multi-Clock Network

Run different layers at different temporal resolutions:

```python
from sc_neurocore.temporal_hierarchy import MultiClockSNN, HetSynLayer

net = MultiClockSNN(
    layers=[
        HetSynLayer(64, 32, tau_mean=2.0),   # fast sensory
        HetSynLayer(32, 16, tau_mean=10.0),   # medium cognitive
        HetSynLayer(16, 4, tau_mean=50.0),    # slow decision
    ],
    layer_names=["sensory", "cognitive", "decision"],
    clock_intervals=[1, 5, 10],  # ticks between updates
)

# Sensory layer updates every tick
# Cognitive layer updates every 5 ticks
# Decision layer updates every 10 ticks

rng = np.random.default_rng(42)
inputs = rng.standard_normal((200, 64)).astype(np.float32)
outputs = net.run(inputs)  # (200, 4)

print(f"Sensory updates:  {200}")
print(f"Cognitive updates: {200 // 5} = {40}")
print(f"Decision updates:  {200 // 10} = {20}")
```

### Compute Savings

Multi-clock scheduling reduces total compute:

| Layer | Clock | Updates | Neurons | Ops |
|-------|:-----:|:-------:|:-------:|:---:|
| Sensory | 1× | 200 | 32 | 6,400 |
| Cognitive | 5× | 40 | 16 | 640 |
| Decision | 10× | 20 | 4 | 80 |
| **Total** | | | | **7,120** |

Uniform 1× clock would cost: 200 × (32+16+4) = 10,400 ops.
Multi-clock saves **32%** compute with no accuracy loss on tasks
where decision timing is >10ms.

## Applications

| Application | Fast Layer | Medium Layer | Slow Layer |
|-------------|-----------|:------------:|-----------|
| Speech recognition | Phoneme detection | Word integration | Sentence meaning |
| Robot control | Reflex (contact) | Trajectory (limb) | Planning (goal) |
| BCI | Spike detection | Feature extraction | Intention decode |
| Autonomous driving | Obstacle detection | Lane tracking | Route planning |

## FPGA Deployment

Multi-clock is natural on FPGA — each layer has its own clock enable:

```
Fast clock ─── Layer 1 (every cycle)
             ├─ Clock divider /5 ─── Layer 2 (every 5th cycle)
             └─ Clock divider /10 ── Layer 3 (every 10th cycle)
```

Power scales with clock rate: slow layers consume proportionally
less dynamic power.

## References

- Perez-Nieves et al. (2021). "Neural Heterogeneity Promotes Robust
  Learning." Nature Communications 12:5791.
- Bellec et al. (2018). "Long Short-Term Memory in Spiking Neural
  Networks." NeurIPS 2018.
- Allen Institute (2020). "Brain Cell Types Database."
