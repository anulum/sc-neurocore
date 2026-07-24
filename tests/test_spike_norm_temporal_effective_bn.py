# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTemporalEffectiveBN from former test_spike_norm.py

"""Focused suite: TestTemporalEffectiveBN from former test_spike_norm.py."""

from __future__ import annotations

from tests.spike_norm_support import *  # noqa: F403


class TestTemporalEffectiveBN:
    def test_forward(self):
        bn = TemporalEffectiveBN(n_features=8, T=10)
        out = bn.forward(np.random.randn(4, 8), t=5, training=True)
        assert out.shape == (4, 8)
