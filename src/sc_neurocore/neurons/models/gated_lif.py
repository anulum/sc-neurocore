# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GatedLIFNeuron:
    """Yao et al. 2022 NeurIPS — LIF with learnable gates."""

    v: float = 0.0
    gate_v: float = 0.9
    gate_i: float = 1.0
    v_threshold: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        self.v = self.gate_v * self.v + self.gate_i * current
        if self.v >= self.v_threshold:
            self.v -= self.v_threshold
            return 1
        return 0

    def reset(self):
        self.v = 0.0


# ── POPULATION / RATE / NEURAL MASS ───────────────────────────────
