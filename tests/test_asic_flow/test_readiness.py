# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC tape-out readiness tests

"""Exercise evidence-derived readiness scoring and failing-check reports."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import (
    SignoffCheckResult,
    SignoffSummary,
    TapeOutChecklist,
)


class TestTapeOutChecklist:
    def test_not_ready_default(self) -> None:
        cl = TapeOutChecklist()
        assert not cl.is_tape_out_ready
        assert cl.readiness_score == 0.0

    def test_fully_ready(self) -> None:
        cl = TapeOutChecklist(
            synthesis_clean=True,
            timing_met=True,
            power_within_budget=True,
            area_within_limit=True,
            drc_clean=True,
            lvs_clean=True,
            formal_equiv_pass=True,
            cdc_clean=True,
            ir_drop_ok=True,
            esd_reviewed=True,
        )
        assert cl.is_tape_out_ready
        assert cl.readiness_score == 1.0
        assert cl.failing_checks() == []

    def test_partial_readiness(self) -> None:
        cl = TapeOutChecklist(synthesis_clean=True, timing_met=True)
        assert cl.readiness_score == 0.2
        assert "drc_clean" in cl.failing_checks()

    def test_from_signoff(self) -> None:
        summary = SignoffSummary(
            timing=SignoffCheckResult("STA", True, ""),
            power=SignoffCheckResult("Power", False, ""),
            area=SignoffCheckResult("Area", True, ""),
            lvs_match=True,
        )
        cl = TapeOutChecklist()
        cl.from_signoff(summary)
        assert cl.timing_met is True
        assert cl.power_within_budget is False
        assert cl.lvs_clean is True
