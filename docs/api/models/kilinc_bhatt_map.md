<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# KilincBhattMapNeuron compatibility alias

`KilincBhattMapNeuron` is a deprecated compatibility alias for
[`SCAdaptiveThresholdMapNeuron`](sc_adaptive_threshold_map.md). It is not a scientific model identity and has no descriptor or independent fidelity count.

Existing imports continue to work, but new code must use the canonical SC name.
For the publication-derived one-state neuron, use
[`NagumoSatoMapNeuron`](nagumo_sato_map.md). The alias must never be used to
attribute the SC recurrence to Nagumo, Sato, Aihara, Kilinc, or Bhatt.
