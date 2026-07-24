# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLatencyBudget from former test_analysis.py

"""Focused suite: TestLatencyBudget from former test_analysis.py."""

from __future__ import annotations

from tests.test_bioware.analysis_support import *  # noqa: F403


class TestLatencyBudget:
    def test_within_budget(self) -> None:
        lb = LatencyBudget(max_latency_us=1000.0)
        assert lb.record(500.0) is True
        assert lb.violations == 0

    def test_exceeds_budget(self) -> None:
        lb = LatencyBudget(max_latency_us=1000.0)
        assert lb.record(1500.0) is False
        assert lb.violations == 1

    def test_compliance_ratio(self) -> None:
        lb = LatencyBudget(max_latency_us=1000.0)
        lb.record(500.0)
        lb.record(500.0)
        lb.record(1500.0)
        assert lb.compliance_ratio == pytest.approx(2.0 / 3.0)

    def test_p99_latency(self) -> None:
        lb = LatencyBudget()
        for i in range(100):
            lb.record(float(i))
        assert lb.p99_latency_us > 90.0

    def test_mean_latency(self) -> None:
        lb = LatencyBudget()
        lb.record(100.0)
        lb.record(300.0)
        assert lb.mean_latency_us == pytest.approx(200.0)

    def test_compliance_ratio_empty_history(self) -> None:
        assert LatencyBudget().compliance_ratio == 1.0
