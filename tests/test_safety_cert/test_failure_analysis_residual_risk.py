# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestResidualRisk from former test_failure_analysis.py

"""Focused suite: TestResidualRisk from former test_failure_analysis.py."""

from __future__ import annotations

from sc_neurocore.safety_cert.safety_cert import (
    FailureCategory,
    FailureMode,
    FMEDA,
)


class TestResidualRisk:
    """Residual risk from undetected failures."""

    def test_residual_risk_all_safe(self) -> None:
        """Only safe failures leave no residual risk."""
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("FM1", "x", "safe", FailureCategory.SAFE, 100.0))
        assert fmeda.residual_risk_fit == 0.0

    def test_residual_risk_undetected(self) -> None:
        """Undetected dangerous failures are all residual risk."""
        fmeda = FMEDA()
        fmeda.add_failure_mode(
            FailureMode("FM1", "x", "bad", FailureCategory.DANGEROUS_UNDETECTED, 100.0)
        )
        assert fmeda.residual_risk_fit == 100.0

    def test_residual_risk_partial(self) -> None:
        """Partial detection leaves part of the risk."""
        fmeda = FMEDA()
        fmeda.add_failure_mode(
            FailureMode("FM1", "x", "det", FailureCategory.DANGEROUS_DETECTED, 100.0, 0.9)
        )
        assert 0 < fmeda.residual_risk_fit < 100.0
