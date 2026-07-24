# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPSNCustomKernel from former test_model_psn.py

"""Focused suite: TestPSNCustomKernel from former test_model_psn.py."""

from __future__ import annotations

from tests.model_psn_support import *  # noqa: F403


class TestPSNCustomKernel:
    def test_custom_kernel_affects_scoring(self):
        """Non-uniform kernel weights recent inputs differently."""
        n = ParallelSpikingNeuron(kernel_size=4, v_threshold=1.0)
        n.kernel = np.array([0.0, 0.0, 0.0, 1.0])  # only last entry
        # Only the value at position 3 matters
        n.step(0.0)
        n.step(0.0)
        n.step(0.0)
        s = n.step(1.0)  # pos 3 gets 1.0, score = 1.0
        assert s == 1
