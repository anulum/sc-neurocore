# Robotics — CPG + Swarm Coupling

Spiking neural network primitives for robotic control: central pattern generation for locomotion and multi-agent spike-based swarm synchronization.

## StochasticCPG — Central Pattern Generator

Two mutually inhibiting `HomeostaticLIFNeuron` instances produce rhythmic alternating spike outputs (e.g., left/right leg stepping). The mutual inhibition ensures anti-phase firing: when neuron 1 fires, its spike trace suppresses neuron 2 via inhibitory current, and vice versa.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `drive_current` | 2.0 | Tonic excitatory input to both neurons |
| `inhibition_weight` | 2.0 | Strength of mutual inhibition |

The `step()` method returns a `(spike1, spike2)` tuple on each timestep. The adaptation rate (0.1) and target rate (0.3) of the homeostatic neurons control oscillation frequency.

## SwarmCoupling — Multi-Agent Synchronization

Synchronizes two `SCLearningLayer` agents by shifting their weights toward each other via Hebbian cross-correlation: `W_a += α * (W_b - W_a)`, `W_b -= α * (W_b - W_a)`. After sufficient synchronization steps, both agents converge to identical weight configurations.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `coupling_strength` | 0.1 | Fraction of weight difference applied per step |

Both agents must have the same neuron count (raises `ValueError` otherwise).

`sc_neurocore.robotics.swarm` is in the scoped public-docstring policy. Its
dedicated robotics swarm tests are strict typed and cover construction,
weight-shift mutation through `SCLearningLayer`, repeated convergence, and
fail-closed mismatched agent sizes at 100% isolated module coverage. This slice
touches only the Python robotics coupling surface; it has no polyglot or
benchmark counterpart.

## Usage

```python
from sc_neurocore.robotics import StochasticCPG, SwarmCoupling

# Locomotion CPG
cpg = StochasticCPG(drive_current=2.0, inhibition_weight=2.0)
left_steps, right_steps = [], []
for _ in range(500):
    s1, s2 = cpg.step()
    left_steps.append(s1)
    right_steps.append(s2)
print(f"Left spikes: {sum(left_steps)}, Right spikes: {sum(right_steps)}")

# Swarm synchronization
from sc_neurocore.layers.sc_learning_layer import SCLearningLayer
agent_a = SCLearningLayer(n_inputs=8, n_neurons=4)
agent_b = SCLearningLayer(n_inputs=8, n_neurons=4)
coupler = SwarmCoupling(coupling_strength=0.3)
for _ in range(20):
    coupler.synchronize(agent_a, agent_b)
# Weights now converged
```

::: sc_neurocore.robotics
    options:
      show_root_heading: true
