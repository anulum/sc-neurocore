# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRuntimeReport from former test_sc_runtime.py

"""Focused suite: TestRuntimeReport from former test_sc_runtime.py."""

from __future__ import annotations

import pytest
from sc_neurocore.control.sc_runtime import (
    ECCMode,
    RuntimeConfig,
    RuntimeReport,
)


class TestRuntimeReport:
    """The runtime adaptation report."""

    def test_adaptation_rate(self) -> None:
        """One adaptation in ten observations is a rate of 0.1."""
        report = RuntimeReport(total_observations=100)
        from sc_neurocore.control.sc_runtime import AdaptationEvent
        import time

        for _ in range(10):
            report.adaptations.append(
                AdaptationEvent(
                    timestamp_ns=time.perf_counter_ns(),
                    trigger="test",
                    old_config={},
                    new_config={},
                    metric_value=0.0,
                )
            )
        assert report.adaptation_rate() == pytest.approx(0.1)

    def test_adaptation_rate_zero(self) -> None:
        """No observations give an adaptation rate of 0."""
        report = RuntimeReport(total_observations=0)
        assert report.adaptation_rate() == 0.0

    def test_adaptation_rate_last_n_window(self) -> None:
        """The rate can cover only the last n observations."""
        from sc_neurocore.control.sc_runtime import AdaptationEvent

        report = RuntimeReport(total_observations=100)
        for _ in range(10):
            report.adaptations.append(
                AdaptationEvent(
                    timestamp_ns=0,
                    trigger="test",
                    old_config={},
                    new_config={},
                    metric_value=0.0,
                )
            )
        # last_n=5 windows the rate over the five most recent adaptations.
        assert report.adaptation_rate(last_n=5) == pytest.approx(1.0)

    def test_summary_includes_ecc_mode(self) -> None:
        """The summary names the error-correction mode."""
        config = RuntimeConfig(ecc_enabled=True, ecc_mode=ECCMode.SECDED)
        report = RuntimeReport(total_observations=5, final_config=config)
        s = report.summary()
        assert "secded_8_4" in s

    def test_summary_includes_uncorrectable(self) -> None:
        """The summary counts the uncorrectable errors."""
        report = RuntimeReport(total_observations=1, uncorrectable_errors=3)
        s = report.summary()
        assert "Uncorrectable errors: 3" in s
