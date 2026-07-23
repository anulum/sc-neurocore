# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHomeostaticPlasticity from former test_plasticity.py

"""Focused suite: TestHomeostaticPlasticity from former test_plasticity.py."""

from __future__ import annotations

from tests.test_bioware.plasticity_support import *  # noqa: F403

class TestHomeostaticPlasticity:
    def test_at_target_no_change(self) -> None:
        hp = HomeostaticPlasticity(target_rate_hz=10.0)
        new = hp.update_threshold(256, observed_rate_hz=10.0, dt_ms=100.0)
        assert new == 256

    def test_too_fast_increases_threshold(self) -> None:
        hp = HomeostaticPlasticity(target_rate_hz=10.0, tau_homeo_ms=1000.0)
        new = hp.update_threshold(256, observed_rate_hz=50.0, dt_ms=1000.0)
        assert new > 256

    def test_too_slow_decreases_threshold(self) -> None:
        hp = HomeostaticPlasticity(target_rate_hz=10.0, tau_homeo_ms=1000.0)
        new = hp.update_threshold(256, observed_rate_hz=1.0, dt_ms=1000.0)
        assert new < 256

    def test_bounded(self) -> None:
        hp = HomeostaticPlasticity(max_threshold_q88=512, min_threshold_q88=64)
        new = hp.update_threshold(500, observed_rate_hz=1000.0, dt_ms=10000.0)
        assert new <= 512
        new = hp.update_threshold(70, observed_rate_hz=0.0, dt_ms=10000.0)
        assert new >= 64
