# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonInput from former test_network_monitors_stimulus.py

"""Focused suite: TestPoissonInput from former test_network_monitors_stimulus.py."""

from __future__ import annotations

from tests.network_monitors_stimulus_support import *  # noqa: F403


class TestPoissonInput:
    def test_creation(self):
        pi = PoissonInput(n=20, rate_hz=100.0, weight=1.0, dt=0.001, seed=42)
        assert pi.n == 20
        assert pi.rate_hz == 100.0

    def test_rate_stored(self):
        pi = PoissonInput(n=5, rate_hz=50.0, weight=2.0, dt=0.001, seed=42)
        assert pi.rate_hz == 50.0
        assert pi.weight == 2.0
