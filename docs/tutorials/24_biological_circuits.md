<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 24: Biological Circuit Primitives

SC-NeuroCore includes 7 ready-to-use biological circuit primitives that
go beyond basic LIF neurons. These circuits model real neural phenomena —
electrical coupling, astrocyte modulation, dendritic computation, cortical
architecture, competitive dynamics, and oscillatory rhythms.

## Why Biological Circuits Matter

Most SNN frameworks model neurons as isolated units connected by weighted
edges. Real brains use fundamentally different connectivity mechanisms:

- **Gap junctions** provide instantaneous electrical coupling (no synaptic delay)
- **Astrocytes** modulate synaptic strength based on network activity
- **Dendrites** perform nonlinear computation before signals reach the soma
- **Cortical columns** organize neurons into functional microcircuits
- **Lateral inhibition** sharpens population responses
- **Gamma oscillations** synchronize distributed processing

SC-NeuroCore is the only SC framework that ships all seven as composable primitives.

## 1. Gap Junctions (Electrical Synapses)

Gap junctions create bidirectional resistive coupling between neurons.
Current flows proportional to the voltage difference — no neurotransmitter,
no delay, no plasticity. They're used for synchronization and rapid signaling.

```python
import numpy as np
from sc_neurocore.synapses.gap_junction import GapJunction

# Create a gap junction with conductance 0.1 nS
gj = GapJunction(conductance=0.1)

# Define connectivity: all-to-all (adjacency matrix)
n = 5
adjacency = np.ones((n, n)) - np.eye(n)  # connected to all except self

# Given membrane voltages, compute coupling currents
voltages = np.array([-65.0, -60.0, -70.0, -55.0, -68.0])
currents = gj.current_matrix(voltages, adjacency)
# Each neuron receives current from its coupled neighbors
# Current flows from high voltage to low voltage
print(f"Coupling currents: {currents}")
# Neuron 3 (V=-55) pushes current outward
# Neuron 2 (V=-70) pulls current inward
```

### How it works

For neurons $i$ and $j$ coupled by conductance $g$:

$$I_{ij} = g \cdot (V_j - V_i)$$

Total current into neuron $i$ is the sum over all coupled neighbors.
Gap junctions are symmetric: $I_{ij} = -I_{ji}$.

### When to use

- Fast synchronization between nearby neurons
- Coupled oscillator models (e.g., inferior olive)
- Modeling interneuron networks where gap junctions dominate

## 2. Tripartite Synapse (Astrocyte Coupling)

The tripartite synapse adds an astrocyte to the classical pre/post pair.
Pre-synaptic spikes trigger glutamate release, which drives astrocyte IP3
production and Ca²⁺ oscillations. When Ca²⁺ exceeds a threshold, the
astrocyte releases gliotransmitter that modulates synaptic weight.

```python
from sc_neurocore.synapses.tripartite import TripartiteSynapse

syn = TripartiteSynapse(
    glut_per_spike=5.0,    # IP3 production rate per spike (µM/s)
    ca_threshold=0.1,      # Ca²⁺ threshold for gliotransmitter release (µM)
    facilitation=1.5,      # weight gain when astrocyte active
    depression_rate=0.001, # weight depression rate below Ca²⁺ threshold
)

# Simulate 10 seconds with spikes every 50ms
weights_over_time = []
for t in range(10000):
    pre_spike = (t % 50 == 0)
    w = syn.step(pre_spike=pre_spike, post_spike=False, dt=0.001)
    weights_over_time.append(w)
    if t % 1000 == 0:
        print(f"  t={t/1000:.0f}s: Ca²⁺={syn.ca:.4f} µM, "
              f"IP3={syn.ip3:.4f}, Weight={w:.4f}")

# The weight oscillates as Ca²⁺ crosses the threshold repeatedly
# This is slow neuromodulation — timescale of seconds, not milliseconds
```

### The cascade

1. Pre-synaptic spike → glutamate release
2. Glutamate → astrocyte IP3 production
3. IP3 → Ca²⁺ release from internal stores
4. Ca²⁺ > threshold → gliotransmitter release
5. Gliotransmitter → synaptic weight modulation

Reference: Araque et al. 1999, "Tripartite synapses"

## 3. Rall Branching Dendrite

Dendrites are not passive cables — they perform computation. Rall's 3/2 power
rule governs how current flows through branching dendritic trees. Distal inputs
are attenuated more than proximal inputs, and the branch pattern determines
which input combinations produce supralinear summation.

```python
import numpy as np
from sc_neurocore.layers.rall_dendrite import RallDendrite

# 4 dendritic branches, each with 5 compartments
dendrite = RallDendrite(n_branches=4, branch_length=5, coupling=0.3)

# Stimulate branch 0 and branch 2 simultaneously
voltages = []
for t in range(200):
    inputs = [0.0, 0.0, 0.0, 0.0]
    if 20 < t < 80:
        inputs[0] = 3.0  # branch 0 active
    if 50 < t < 120:
        inputs[2] = 2.5  # branch 2 active (overlapping)
    soma_v = dendrite.step(branch_inputs=inputs)
    voltages.append(soma_v)

# When both branches are active simultaneously, the soma voltage
# exceeds the sum of individual branch contributions (supralinear)
peak_overlap = max(voltages[50:80])
print(f"Peak during overlap: {peak_overlap:.3f}")
```

### Rall's 3/2 power rule

At a branch point with parent diameter $d_p$ and child diameters $d_1, d_2$:

$$d_p^{3/2} = d_1^{3/2} + d_2^{3/2}$$

This ensures impedance matching — no signal reflection at branch points.

## 4. Canonical Cortical Microcircuit

The 5-population cortical column implements the Douglas & Martin (2004) canonical
microcircuit: L4 receives thalamic input, L2/3 performs recurrent processing,
L5 generates output, L6 provides feedback.

```python
import numpy as np
from sc_neurocore.network.cortical_column import CorticalColumn

col = CorticalColumn(n_per_layer=20, seed=42)

# Drive with thalamic input
thalamic = np.ones(20) * 5.0
results = col.run(thalamic_input=thalamic, steps=500)

# Each layer has distinct firing patterns
for layer_name in ['l23_exc', 'l23_inh', 'l4', 'l5', 'l6']:
    spikes = results[layer_name]
    rate = spikes.sum() / (500 * 0.001 * 20)  # Hz
    print(f"  {layer_name}: {rate:.1f} Hz mean rate")

# L5 is the output layer — its activity represents the column's response
print(f"L5 total spikes: {results['l5'].sum()}")
```

### Information flow

```
Thalamus → L4 → L2/3 exc ←→ L2/3 inh
                  ↓
                 L5 → cortical output
                  ↓
                 L6 → thalamic feedback
```

## 5. Lateral Inhibition

Lateral inhibition sharpens population responses — the most active neuron
suppresses its neighbors. This is the mechanism behind contrast enhancement
in the retina and orientation selectivity in V1.

```python
import numpy as np
from sc_neurocore.layers.circuit_primitives import LateralInhibition

li = LateralInhibition(n_neurons=10, inhibition_strength=0.5, radius=2)

# Input: broad activation with a peak at neuron 5
activations = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 0.7, 0.5, 0.3, 0.1])
inhibited = li.apply(activations)

print("Before:", np.round(activations, 2))
print("After: ", np.round(inhibited, 2))
# The peak at neuron 5 is preserved, neighbors are suppressed
# Result is a sharper, more selective response
```

### Parameters

- `sigma`: spatial extent of inhibition (Gaussian falloff)
- `strength`: inhibition amplitude (0 = no effect, 1 = strong suppression)

## 6. Winner-Take-All (WTA)

WTA is the extreme case of lateral inhibition — only the top-k neurons
survive, all others are silenced.

```python
import numpy as np
from sc_neurocore.layers.circuit_primitives import WinnerTakeAll

wta = WinnerTakeAll(n_neurons=10, k=3)

activations = np.array([0.1, 0.9, 0.4, 0.6, 0.8, 1.0, 0.7, 0.5, 0.3, 0.2])
winners = wta.apply(activations)

print("Input:  ", np.round(activations, 2))
print("Winners:", np.round(winners, 2))
# Only neurons 1, 4, 5 (top-3) retain their activation
```

### Use cases

- Classification layers (the winning class is the prediction)
- Competitive learning (only winners update their weights)
- Sparse coding (enforce sparsity in representations)

## 7. PING Gamma Oscillation

Pyramidal-Interneuron Network Gamma (PING) produces 30-80 Hz oscillations
through the interaction of excitatory and inhibitory populations. Pyramidal
cells fire, drive interneurons, interneurons inhibit pyramidal cells, inhibition
decays, pyramidal cells fire again — the cycle repeats at gamma frequency.

```python
import numpy as np
from sc_neurocore.network.gamma_oscillation import PINGCircuit

ping = PINGCircuit(n_excitatory=80, n_inhibitory=20)

# Step the circuit in a loop (no run() — use step())
exc_spikes_total = 0
for t in range(1000):
    exc_spikes, inh_spikes = ping.step(drive=5.0, dt=0.1)
    exc_spikes_total += exc_spikes.sum()

print(f"Total excitatory spikes: {exc_spikes_total}")
# The frequency depends on the drive and E/I balance
# drive=5.0 typically produces 40-50 Hz gamma
```

### Why gamma matters

Gamma oscillations are implicated in:
- **Attention**: attended stimuli produce stronger gamma
- **Working memory**: items are held in gamma-frequency activity
- **Binding**: features of an object oscillate in phase
- **Communication**: cortical areas communicate via gamma coherence

The PING mechanism is the simplest circuit that produces realistic gamma —
it requires only excitatory-inhibitory feedback with appropriate time constants.

## Combining Primitives

These primitives compose. A cortical column with gap-coupled interneurons,
astrocyte-modulated synapses, and dendritic computation:

```python
from sc_neurocore.network.cortical_column import CorticalColumn
from sc_neurocore.synapses.gap_junction import GapJunction

# Build a column
col = CorticalColumn(n_per_layer=20, seed=42)

# Add gap junctions between L2/3 inhibitory neurons
gj = GapJunction(conductance=0.05, n_neurons=20)

# Run with coupled dynamics
results = col.run(thalamic_input=np.ones(20) * 5.0, steps=500)
```

## Further Reading

- [Tutorial 02: Building Your First SNN](02_building_your_first_snn.md) — basic neuron/synapse concepts
- [Tutorial 31: Network Simulation Engine](31_network_simulation_engine.md) — Population-Projection-Network
- [API: Bio](../api/bio.md) — biological circuit API docs
