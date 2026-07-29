<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# SC Compte working-memory network

`SC-COMPTE-WM-NETWORK` is the retained SC-NeuroCore network-level successor to
the source-bounded [`CompteWMNeuron`](models/compte_wm.md). Its public Python
specification is `sc_neurocore.network.SCCompteWMNetworkSpec`.

This is deliberately an **SC project model**. It uses the architecture and
control parameters reported by Compte, Brunel, Goldman-Rakic, and Wang,
*Cerebral Cortex* 10(9), 910–923 (2000), DOI
`10.1093/cercor/10.9.910`, while freezing reproducibility choices that the
paper does not define as a portable executable contract. It is neither another
neuron nor the legacy 500-cell `working_memory_circuit` approximation.

## Frozen v1 surface

- 2,048 pyramidal cells plus 512 inhibitory interneurons on uniform
  preferred-cue rings;
- independent 1,800 Hz per-cell external Poisson drive through AMPA;
- control conductances `G_EE=0.381 nS`, `G_EI=0.292 nS`,
  `G_IE=1.336 nS`, and `G_II=1.024 nS`;
- a unit-mean E-to-E footprint with `J_plus=1.62` and `sigma=18 degrees`;
- optional tuned E-to-I connectivity with `J_plus=1.25` and
  `sigma=18 degrees`;
- source cell constants and AMPA/NMDA/GABAA kinetics at `dt=0.02 ms`;
- named control and modulated sets, where the latter scales recurrent NMDA by
  1.2 and recurrent GABAA by 1.4; and
- a deterministic SC compact cue profile plus explicit circular population
  statistics.

The implementation computes shortest circular distances, exact discrete
unit-mean connectivity footprints, cue currents, signed distractor
displacements, population firing rates, bump angle, resultant length, and
circular width. Invalid sizes, non-finite values, non-positive parameters,
empty target grids, and spike-count shape mismatches fail closed.

## Claim boundary

The specification is not a simulator receipt. Persistent-bump formation,
delay stability, random drift, response reset, distractor resistance,
cross-language parity, benchmark performance, and silicon behavior remain
open until demonstrated by separately committed executable and statistical
evidence. The network therefore does not increment the 49/155 neuron-model
fidelity count.

## Example

```python
from sc_neurocore.network import SCCompteWMNetworkSpec

spec = SCCompteWMNetworkSpec(modulated=True)
angles = spec.preferred_angles_deg("excitatory")
cue_pa = spec.cue_current_pa(180.0, angles)
ee_footprint = spec.connectivity_footprint("ee", 180.0, angles)

assert spec.n_cells == 2560
assert cue_pa.max() == 200.0
assert abs(ee_footprint.mean() - 1.0) < 1e-12
```
