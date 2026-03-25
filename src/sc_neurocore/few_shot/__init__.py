# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-based few-shot meta-learning

"""Few-shot learning with spike-timing rules. Not gradient-based MAML."""

from .haam import HebbianFewShot, SpikePrototypeNet

__all__ = ["HebbianFewShot", "SpikePrototypeNet"]
