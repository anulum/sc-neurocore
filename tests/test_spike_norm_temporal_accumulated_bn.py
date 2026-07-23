# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTemporalAccumulatedBN from former test_spike_norm.py

"""Focused suite: TestTemporalAccumulatedBN from former test_spike_norm.py."""

from __future__ import annotations

from tests.spike_norm_support import *  # noqa: F403

class TestTemporalAccumulatedBN:
    def test_forward(self):
        bn = TemporalAccumulatedBN(n_features=8)
        for _ in range(5):
            out = bn.forward(np.random.randn(4, 8), training=True)
        assert out.shape == (4, 8)

    def test_reset(self):
        bn = TemporalAccumulatedBN(n_features=4)
        bn.forward(np.ones(4), training=True)
        bn.reset()
        assert np.allclose(bn._accumulated, 0)
