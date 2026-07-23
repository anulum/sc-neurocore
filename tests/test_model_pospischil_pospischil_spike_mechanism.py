# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPospischilSpikeMechanism from former test_model_pospischil.py

"""Focused suite: TestPospischilSpikeMechanism from former test_model_pospischil.py."""

from __future__ import annotations

from tests.model_pospischil_support import *  # noqa: F403

class TestPospischilSpikeMechanism:
    def test_upward_crossing_only(self):
        n = PospischilNeuron()
        spike_count = 0
        for _ in range(50000):
            previous_v = n.v
            spike = n.step(10.0)
            crossed_upward = previous_v < n.v_threshold <= n.v

            assert spike == int(crossed_upward)
            spike_count += spike

        assert spike_count > 100
