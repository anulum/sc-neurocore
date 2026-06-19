# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Global Emotional/Chemical System

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class NeuromodulatorSystem:
    """
    Global Emotional/Chemical System.
    Modulates neuron parameters based on Dopamine (DA), Serotonin (5HT), Norepinephrine (NE).
    """

    da_level: float = 0.5  # Baseline
    ht_level: float = 0.5
    ne_level: float = 0.1

    def update_levels(self, reward: float, stress: float) -> None:
        """
        Adjust chemicals based on environmental feedback.
        """
        # Reward boosts Dopamine
        self.da_level += 0.1 * (reward - self.da_level)

        # Stress boosts Adrenaline (NE) and drops Serotonin (5HT)
        self.ne_level += 0.2 * (stress - self.ne_level)
        self.ht_level -= 0.1 * stress
        self.ht_level = np.clip(self.ht_level, 0.1, 1.0)

    def modulate_neuron(self, neuron_params: dict[str, Any]) -> dict[str, Any]:
        """
        Returns modified parameters for a StochasticLIFNeuron.
        """
        mod_params = neuron_params.copy()

        # Dopamine: Lowers Threshold (Excitation)
        if "v_threshold" in mod_params:
            mod_params["v_threshold"] *= 1.0 - 0.2 * self.da_level

        # 5-HT reduces noise (stabilisation effect)
        if "noise_std" in mod_params:
            mod_params["noise_std"] *= 1.0 - 0.5 * self.ht_level

        # Adrenaline: Increases Noise (Exploration) and Gain
        if "noise_std" in mod_params:
            mod_params["noise_std"] += 0.1 * self.ne_level

        return mod_params
