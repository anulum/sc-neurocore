# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFLIFIsolation from former test_model_fractional_lif.py

"""Focused suite: TestFLIFIsolation from former test_model_fractional_lif.py."""

from __future__ import annotations

from tests.model_fractional_lif_support import *  # noqa: F403


class TestFLIFIsolation:
    def test_defaults(self):
        n = FractionalLIFNeuron()
        assert n.v == 0.0 and n.alpha == 0.8 and n.v_threshold == 1.0
        assert n._max_history == 100

    def test_step_returns_binary(self):
        assert FractionalLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = FractionalLIFNeuron()
        for _ in range(10000):
            n.step(5.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = FractionalLIFNeuron(v_rest=0.2, v_reset=0.2, v_threshold=1.0)
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.v == n.v_rest
        assert len(n._history) == n._max_history
        assert n._history[-1] == n.v_rest
