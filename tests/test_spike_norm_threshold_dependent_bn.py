# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThresholdDependentBN from former test_spike_norm.py

"""Focused suite: TestThresholdDependentBN from former test_spike_norm.py."""

from __future__ import annotations

from tests.spike_norm_support import *  # noqa: F403

class TestThresholdDependentBN:
    def test_forward(self):
        bn = ThresholdDependentBN(n_features=8, threshold=1.0)
        out = bn.forward(np.random.randn(4, 8), training=True)
        assert out.shape == (4, 8)

    def test_inference(self):
        bn = ThresholdDependentBN(n_features=4)
        bn.forward(np.random.randn(10, 4), training=True)
        out = bn.forward(np.random.randn(3, 4), training=False)
        assert out.shape == (3, 4)
