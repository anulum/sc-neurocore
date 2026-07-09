# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


from dataclasses import dataclass
import numpy as np
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer


@dataclass
class RadHardLayer(VectorizedSCLayer):
    """
    Space-Hardened Layer with TMR (Triple Modular Redundancy) logic.
    Simulates radiation effects (SEU) and correction.
    """

    seu_rate: float = 0.001  # Probability of bit flip per step
    tmr_enabled: bool = True

    def forward(self, input_values):
        # 1. Run 3 parallel instances (TMR)
        if self.tmr_enabled:
            out1 = self._noisy_forward(input_values)
            out2 = self._noisy_forward(input_values)
            out3 = self._noisy_forward(input_values)

            # Majority Vote
            # Simply average and threshold?
            # Or bitwise vote.
            # Here we just average the output values (Simulating TMR at system level)

            avg = (out1 + out2 + out3) / 3.0
            return avg
        else:
            return self._noisy_forward(input_values)

    def _noisy_forward(self, input_values):
        # Simulate SEU in weights
        original_w = self.weights.copy()

        # Inject faults
        mask = np.random.random(self.weights.shape) < self.seu_rate
        # Flip bits: If probability p -> 1-p ?
        # Or just random noise.
        # Let's say bit flip in weight Memory:
        # weights are probabilities. A digital flip changes prob drastically.
        # We simulate this as noise.
        self.weights[mask] = np.abs(self.weights[mask] - 0.5)  # Flip around 0.5?

        # Refresh packed
        self._refresh_packed_weights()

        res = super().forward(input_values)

        # Scrubbing (Restore)
        self.weights = original_w
        self._refresh_packed_weights()

        return res
