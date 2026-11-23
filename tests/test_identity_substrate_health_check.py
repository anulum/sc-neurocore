# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHealthCheck from former test_identity_substrate.py

"""Focused suite: TestHealthCheck from former test_identity_substrate.py."""

from __future__ import annotations

from tests.identity_substrate_support import (
    _make_substrate,
)


class TestHealthCheck:
    """The substrate health check."""

    def test_health_check_initial(self) -> None:
        """A fresh substrate is healthy with zero rate."""
        sub = _make_substrate()
        hc = sub.health_check()
        assert hc["is_healthy"] is True
        assert hc["mean_rate"] == 0.0

    def test_health_check_reports_zero_spectral_entropy_for_silent_substrate(self) -> None:
        """A silent substrate has zero spectral entropy."""
        # Once enough silent history accumulates, the population train carries no
        # spectral power, so the spectral entropy collapses to zero.
        sub = _make_substrate()
        for _ in range(110):
            sub.step()
        hc = sub.health_check()
        assert hc["spectral_entropy"] == 0.0
