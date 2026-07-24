# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFLIFParameters from former test_model_fractional_lif.py

"""Focused suite: TestFLIFParameters from former test_model_fractional_lif.py."""

from __future__ import annotations

from tests.model_fractional_lif_support import *  # noqa: F403


class TestFLIFParameters:
    @pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0])
    def test_alpha_variations(self, alpha: float):
        n = FractionalLIFNeuron(alpha=alpha)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = FractionalLIFNeuron()
            trace = [(n.step(5.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
