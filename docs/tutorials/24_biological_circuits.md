# Tutorial 24: Biological Circuit Primitives

SC-NeuroCore includes 7 ready-to-use biological circuit primitives that
go beyond basic LIF neurons. These circuits are unique to SC-NeuroCore —
no other SC framework ships them.

## Gap Junctions (Electrical Synapses)

```python
from sc_neurocore.synapses.gap_junction import GapJunction

gj = GapJunction(conductance=0.1, n_neurons=5)
currents = gj.current_matrix(voltages)  # (5,) mutual coupling currents
```

## Tripartite Synapse (Astrocyte Coupling)

Pre-synaptic spikes drive astrocyte IP3 production → Ca²⁺ oscillations
modulate synaptic weight via gliotransmitter release (Araque et al. 1999).

```python
from sc_neurocore.synapses.tripartite import TripartiteSynapse

syn = TripartiteSynapse(glut_per_spike=5.0, ca_threshold=0.1)
for t in range(1000):
    w = syn.step(pre_spike=True, post_spike=False, dt=0.01)
print(f"Ca²⁺: {syn.ca:.3f} µM, Weight: {w:.3f}")
```

## Rall Branching Dendrite

Compartmental dendritic tree with Rall's 3/2 power rule. Distal inputs
propagate toward soma with inter-compartment coupling.

```python
from sc_neurocore.layers.rall_dendrite import RallDendrite

dendrite = RallDendrite(n_branches=4, branch_length=5, coupling=0.3)
for t in range(50):
    soma_v = dendrite.step(branch_inputs=[1.0, 0.0, 0.5, 0.0])
```

## Canonical Cortical Microcircuit

5-population cortical column (Douglas & Martin 2004):
L4 (thalamic) → L2/3 exc ↔ L2/3 inh → L5 (output) → L6 (feedback).

```python
from sc_neurocore.network.cortical_column import CorticalColumn

col = CorticalColumn(n_per_layer=20, seed=42)
results = col.run(thalamic_input=np.ones(20) * 5.0, steps=200)
print(f"L5 spikes: {results['l5'].sum()}")
```

## Lateral Inhibition + Winner-Take-All

```python
from sc_neurocore.layers.circuit_primitives import LateralInhibition, WinnerTakeAll

li = LateralInhibition(n_neurons=10, sigma=2.0, strength=0.5)
inhibited = li.apply(activations)

wta = WinnerTakeAll(n_neurons=10, k=3)
winners = wta.apply(activations)  # top-3 survive
```

## PING Gamma Oscillation

Pyramidal-Interneuron Network Gamma (Wang-Buzsaki):

```python
from sc_neurocore.network.gamma_oscillation import PINGCircuit

ping = PINGCircuit(n_exc=80, n_inh=20, seed=42)
results = ping.run(drive=5.0, steps=500)
# Expect 30-80 Hz gamma oscillation
```
