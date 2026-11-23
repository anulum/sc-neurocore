# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFaultResult from former test_resilience.py

"""Focused suite: TestFaultResult from former test_resilience.py."""

from __future__ import annotations

from sc_neurocore.resilience.fault_suite import (
    FaultResult,
    FaultType,
)


class TestFaultResult:
    """The result of one fault run."""

    def test_degradation(self) -> None:
        """Degradation is the accuracy lost to the fault."""
        r = FaultResult(
            fault_type=FaultType.STUCK_AT_ZERO,
            fault_rate=0.1,
            layer_index=None,
            accuracy_before=0.95,
            accuracy_after=0.80,
            degradation=0.15,
        )
        assert r.degradation == 0.15
