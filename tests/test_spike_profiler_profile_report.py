# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProfileReport from former test_spike_profiler.py

"""Focused suite: TestProfileReport from former test_spike_profiler.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.profiling.spike_profiler import (
    Pathology,
    ProfileReport,
    Severity,
    SpikeProfiler,
)
from tests.spike_profiler_support import (
    _random_spikes,
)


class TestProfileReport:
    """The profile report's summary and its critical flag."""

    def test_summary_format(self) -> None:
        """The summary names the report and its layers."""
        p = SpikeProfiler()
        rng = np.random.RandomState(0)
        for _ in range(10):
            p.record_step("h", _random_spikes(8, 0.2, rng))
        r = p.report()
        s = r.summary()
        assert "SpikeProfiler Report" in s
        assert "h:" in s

    def test_has_critical(self) -> None:
        """A report with a critical pathology says so."""
        r = ProfileReport(
            pathologies=[
                Pathology(Severity.CRITICAL, "test", "l", "msg", "fix"),
            ]
        )
        assert r.has_critical is True

    def test_no_critical(self) -> None:
        """A report without critical pathologies says so."""
        r = ProfileReport(
            pathologies=[
                Pathology(Severity.WARNING, "test", "l", "msg", "fix"),
            ]
        )
        assert r.has_critical is False

    def test_empty_has_critical(self) -> None:
        """An empty report has no critical pathology."""
        r = ProfileReport()
        assert r.has_critical is False
