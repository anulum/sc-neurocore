<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# SC Compte working-memory network

`SC-COMPTE-WM-NETWORK` is the retained SC-NeuroCore network-level successor to
the source-bounded [`CompteWMNeuron`](models/compte_wm.md). Its public Python
specification and executor are `sc_neurocore.network.SCCompteWMNetworkSpec`
and `sc_neurocore.network.SCCompteWMNetwork`.

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
- a counter-addressed SplitMix64/inverse-CDF input stream whose seed, stream,
  step, and cell mapping is stable across batching and intended for direct
  native-language ports;
- control conductances `G_EE=0.381 nS`, `G_EI=0.292 nS`,
  `G_IE=1.336 nS`, and `G_II=1.024 nS`;
- a unit-mean E-to-E footprint with `J_plus=1.62` and `sigma=18 degrees`;
- optional tuned E-to-I connectivity with `J_plus=1.25` and
  `sigma=18 degrees`;
- source cell constants and AMPA/NMDA/GABAA kinetics at `dt=0.02 ms`;
- no recurrent E-to-E or I-to-I autapses in the SC v1 execution contract;
- named control and modulated sets, where the latter scales recurrent NMDA by
  1.2 and recurrent GABAA by 1.4; and
- a deterministic SC compact cue profile plus explicit circular population
  statistics.

The implementation computes shortest circular distances, exact discrete
unit-mean connectivity footprints, cue currents, signed distractor
displacements, population firing rates, bump angle, resultant length, and
circular width. The vectorized executor advances all 2,560 cells with coupled
midpoint RK2 channel flow, circular E-to-E convolution through a real FFT,
optional structured E-to-I convolution, uniform inhibitory projections,
sampled threshold/reset/refractory behaviour, explicit event overrides, and
atomic candidate validation. Every step receipts external counts plus input
and state digests; every run receipts input, spike, final-state, and bounded
window statistics. Invalid sizes, non-finite values, non-positive parameters,
partial event overrides, out-of-run stimuli, empty target grids, and
spike-count shape mismatches fail closed.

## Source and SC choices

The paper supplies the biological architecture, control conductances, channel
equations, timestep, population sizes, and Poisson rate. It does not supply a
portable pseudorandom stream, a cross-language aggregate-input mapping, or an
unambiguous autapse convention. The counter stream, inverse-CDF sampler,
per-cell aggregate counts, no-autapse rule, compact raised-cosine cue, sampled
threshold detector, digest encoding, and reduction order are therefore
explicit SC project choices. The source used a larger 4,096+1,024 network for
its reported distractor experiment; the frozen SC v1 identity remains the
requested 2,048+512 ring, so future distractor evidence must be described as
SC-network evidence rather than a reproduction of that larger figure.

The preserved scalar `CompteWMNeuron` remains a separate original model. A
focused executable parity test isolates one network pyramidal cell and proves
its external-AMPA midpoint step agrees with that original model. Another test
compares the FFT ring path against an independently reduced dense target sum.

## Claim boundary

The Python executor and committed benchmark are deterministic simulator
receipts, not behavioral validation. The benchmark exercises 1,000 full
network steps and binds its result to source hashes, but records local loaded-
host regression timing only. Persistent-bump formation, delay stability,
random drift, response reset, distractor resistance, native-network parity,
and silicon behavior remain open until demonstrated by separately committed
ensemble and backend evidence. The network therefore does not increment the
49/155 neuron-model fidelity count.

## Example

```python
from sc_neurocore.network import (
    SCCompteWMNetwork,
    SCCompteWMNetworkSpec,
    SCCompteWMStimulus,
)

spec = SCCompteWMNetworkSpec(modulated=True)
angles = spec.preferred_angles_deg("excitatory")
cue_pa = spec.cue_current_pa(180.0, angles)
ee_footprint = spec.connectivity_footprint("ee", 180.0, angles)

assert spec.n_cells == 2560
assert cue_pa.max() == 200.0
assert abs(ee_footprint.mean() - 1.0) < 1e-12

network = SCCompteWMNetwork(spec)
cue = SCCompteWMStimulus(0.0, 250.0, 200.0, center_deg=180.0)
receipt = network.run(250.0, stimuli=(cue,))
assert receipt.steps == 12_500
assert len(receipt.final_state_sha256) == 64
```
