# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


from dataclasses import dataclass


@dataclass
class DarkForestAgent:
    """
    Game Theoretic Agent for the Fermi Paradox (Dark Forest Theory).
    Decides whether to Broadcast or Hide.
    """

    hostility_factor: float = 0.9  # Probability that other civs are hostile
    detection_threshold: float = 0.5

    def decide(self, alien_signal_strength: float) -> str:
        """
        Input: Strength of detected alien signal [0, 1].
        Output: 'HIDE', 'BROADCAST', 'STRIKE'
        """
        # If signal is strong, we are detected.
        if alien_signal_strength > self.detection_threshold:
            # High risk.
            if self.hostility_factor > 0.5:
                return "STRIKE"  # Pre-emptive strike (Game Theory optimal in Dark Forest)
            else:
                return "BROADCAST"  # Hope for peace

        # If signal is weak, we stay hidden to avoid detection
        return "HIDE"
