# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-domain data augmentation and curriculum learning

"""Spike-aware augmentation and curriculum scheduling for SNN training."""

from .spike_augment import SpikeAugment
from .curriculum import SpikeCurriculum

__all__ = ["SpikeAugment", "SpikeCurriculum"]
