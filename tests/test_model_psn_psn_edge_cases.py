# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPSNEdgeCases from former test_model_psn.py

"""Focused suite: TestPSNEdgeCases from former test_model_psn.py."""

from __future__ import annotations

from tests.model_psn_support import *  # noqa: F403


class TestPSNEdgeCases:
    @pytest.mark.parametrize("ks", [2, 4, 8, 16])
    def test_kernel_size_variations(self, ks: int):
        n = SCResettingParallelSpikingNeuron(kernel_size=ks, v_threshold=1.0)
        spikes = sum(n.step(1.0) for _ in range(500))
        assert spikes > 0

    def test_zero_input(self):
        n = SCResettingParallelSpikingNeuron()
        spikes = sum(n.step(0.0) for _ in range(100))
        assert spikes == 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SCResettingParallelSpikingNeuron()
            trace = [(n.step(1.5), float(n.buffer.sum())) for _ in range(50)]
            traces.append(trace)
        assert traces[0] == traces[1]
