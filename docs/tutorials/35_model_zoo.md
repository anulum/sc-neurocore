<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 35: Model Zoo — Pre-Built Networks

SC-NeuroCore ships 10 pre-built network configurations and 3 pre-trained weight
sets. Use them as starting points, baselines, or building blocks for larger systems.

## Available Configurations

| Config | Neurons | Topology | Task |
|--------|--------:|----------|------|
| `mnist_classifier` | 922 | feedforward 784→128→10 | Digit classification |
| `shd_speech_classifier` | 976 | feedforward 700→256→20 | Spoken digit recognition |
| `dvs_gesture_classifier` | 523 | feedforward 256→256→11 | DVS gesture recognition |
| `brunel_balanced_network` | 1000 | random E/I | Balanced asynchronous state |
| `cortical_column` | 600 | laminar | 6-layer cortical microcircuit |
| `central_pattern_generator` | 16 | ring (mutual inhibition) | Locomotion rhythm |
| `decision_making_circuit` | 720 | competing pools | Evidence accumulation |
| `working_memory_circuit` | 500 | recurrent excitatory | Persistent activity |
| `auditory_processing` | 192 | tonotopic | Frequency decomposition |
| `visual_cortex_v1` | 400 | orientation columns | Edge detection |

## 1. Load and Run a Configuration

```python
from sc_neurocore.model_zoo.configs import brunel_balanced_network

# Create a Brunel balanced network
net = brunel_balanced_network(n_exc=800, n_inh=200)

# Run for 1 second
net.run(duration=1.0, dt=0.001)

# Access spike monitors (automatically attached)
for mon in net.monitors:
    print(f"{mon.label}: {mon.count} spikes")
```

## 2. MNIST Classifier

The MNIST config creates a 784→128→10 feedforward SNN:

```python
from sc_neurocore.model_zoo.configs import mnist_classifier

net = mnist_classifier(n_hidden=128)
# Input layer: 784 neurons (one per pixel)
# Hidden layer: 128 LIF neurons
# Output layer: 10 LIF neurons (one per digit)
```

### With Pre-Trained Weights

```python
from sc_neurocore.model_zoo.pretrained import load_pretrained

net = load_pretrained("mnist_784_128_10")
# Weights trained with surrogate gradients (95.5% accuracy)
# Ready to classify — feed pixel values as input currents
```

## 3. All Pre-Trained Weight Sets

| Name | Architecture | Accuracy | Training method |
|------|-------------|----------|-----------------|
| `mnist_784_128_10` | FC 784→128→10 | 95.5% | Surrogate gradient (ATan) |
| `shd_700_256_20` | FC 700→256→20 | — | Surrogate gradient |
| `dvs_256_256_11` | FC 256→256→11 | — | Surrogate gradient |

```python
from sc_neurocore.model_zoo.pretrained import load_pretrained

# Load any pretrained network
net = load_pretrained("shd_700_256_20")
net = load_pretrained("dvs_256_256_11")
```

## 4. Central Pattern Generator (CPG)

Generates rhythmic motor patterns for locomotion:

```python
from sc_neurocore.model_zoo.configs import central_pattern_generator

cpg = central_pattern_generator(n_oscillators=4)
cpg.run(duration=2.0, dt=0.001)

# 4 oscillators with mutual inhibition produce quadruped gait patterns
# Phase-locked at ~90° intervals (walk), ~180° (trot), or ~0° (gallop)
```

## 5. Decision-Making Circuit

Two competing neural pools accumulate evidence until one reaches threshold
(Wang 2002 model):

```python
from sc_neurocore.model_zoo.configs import decision_making_circuit

dm = decision_making_circuit(n_per_pool=240)
# Pool A receives slightly stronger input → wins after ~300ms
dm.run(duration=1.0, dt=0.001)
```

## 6. Working Memory

Persistent activity via recurrent excitation (Compte et al. 2000):

```python
from sc_neurocore.model_zoo.configs import working_memory_circuit

wm = working_memory_circuit(n_neurons=500)
# Brief input pulse → sustained firing for seconds after input ends
wm.run(duration=5.0, dt=0.001)
```

## 7. Cortical Column

6-layer cortical microcircuit (see Tutorial 24 for the 5-population version):

```python
from sc_neurocore.model_zoo.configs import cortical_column

col = cortical_column(n_layers=6)
col.run(duration=0.5, dt=0.001)
```

## 8. Sensory Processing

```python
from sc_neurocore.model_zoo.configs import auditory_processing, visual_cortex_v1

# Tonotopic frequency decomposition
aud = auditory_processing(n_channels=32)
aud.run(duration=0.5, dt=0.001)

# V1 orientation columns (8 orientations × 50 neurons)
v1 = visual_cortex_v1(n_orientation=8, n_per_orientation=50)
v1.run(duration=0.3, dt=0.001)
```

## 9. Customizing Configurations

All configs return a `Network` object — modify before running:

```python
from sc_neurocore.model_zoo.configs import brunel_balanced_network

net = brunel_balanced_network(n_exc=400, n_inh=100)

# Add a stimulus
from sc_neurocore.network.stimulus import PoissonInput
stim = PoissonInput(n=400, rate_hz=20.0, weight=1.5, dt=0.001)
net.add(stim)

# Add more monitors
from sc_neurocore.network.monitor import SpikeMonitor
for pop in net.populations:
    net.add(SpikeMonitor(pop))

net.run(duration=1.0, dt=0.001)
```

## 10. Using with Rust Backend

All 10 configs work with the Rust NetworkRunner for 10-100x speedup:

```python
net = brunel_balanced_network(n_exc=5000, n_inh=1250)
net.run(duration=1.0, dt=0.001, backend="rust")
# 6250 neurons, Rayon-parallel, near-linear scaling
```

## Further Reading

- [Tutorial 31: Network Simulation Engine](31_network_simulation_engine.md) — Population/Projection/Network API
- [Tutorial 06: Brunel Network Translation](06_brunel_network_translation.md) — Brian2 comparison
- [Example 12: Load Pretrained Model](https://github.com/anulum/sc-neurocore/blob/main/examples/12_load_pretrained_model.py)
