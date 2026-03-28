# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BrainChip Akida 2021 — event-domain rank-order IF neuron

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AkidaNeuron:
    """BrainChip Akida 2021 — event-domain rank-order IF neuron.

    Membrane integrates weighted spikes with rank-order decay:
    V += weight * modulation^rank
    Spike when V >= threshold. No leak between events.
    """

    v: int = 0
    threshold: int = 100
    modulation: float = 0.75
    _rank: int = 0
    _spiked: bool = False

    def step(self, weight: int) -> int:
        """Process one input spike event with given synaptic weight."""
        if weight != 0:
            scaled = int(weight * self.modulation**self._rank)
            self.v += scaled
            self._rank += 1
        if self.v >= self.threshold and not self._spiked:
            self._spiked = True
            return 1
        return 0

    def reset(self) -> None:
        self.v = 0
        self._rank = 0
        self._spiked = False
