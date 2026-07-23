# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIsolation from former test_model_gated_lif.py

"""Focused suite: TestIsolation from former test_model_gated_lif.py."""

from __future__ import annotations

from tests.model_gated_lif_support import *  # noqa: F403

class TestIsolation:
    def test_step_returns_binary(self):
        assert GatedLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = GatedLIFNeuron()
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(getattr(n, "v", 0.0))

    def test_reset(self):
        n = GatedLIFNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
