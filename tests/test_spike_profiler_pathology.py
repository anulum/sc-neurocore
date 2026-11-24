# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPathology from former test_spike_profiler.py

"""Focused suite: TestPathology from former test_spike_profiler.py."""

from __future__ import annotations

from sc_neurocore.profiling.spike_profiler import (
    Pathology,
    Severity,
)


class TestPathology:
    """The fields of a profiler pathology."""

    def test_fields(self) -> None:
        """A pathology keeps its severity and metric value."""
        p = Pathology(
            severity=Severity.WARNING,
            category="dead_neurons",
            layer="hidden",
            message="50% dead",
            suggestion="lower threshold",
            metric_value=0.5,
        )
        assert p.severity == Severity.WARNING
        assert p.metric_value == 0.5
