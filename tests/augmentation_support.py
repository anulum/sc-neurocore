# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_augmentation.py

from __future__ import annotations

import numpy as np
from sc_neurocore.augmentation import SpikeAugment, SpikeCurriculum
def _make_spikes(T=20, N=10, rate=0.2, seed=42):
    rng = np.random.RandomState(seed)
    return (rng.random((T, N)) < rate).astype(np.int8)

__all__ = ['np', 'SpikeAugment', 'SpikeCurriculum', '_make_spikes']
