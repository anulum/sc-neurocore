# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC tape-out readiness state

"""Track evidence-backed ASIC tape-out readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sc_neurocore.asic_flow.signoff import SignoffSummary


@dataclass
class TapeOutChecklist:
    """Go/no-go checklist for ASIC tape-out."""

    synthesis_clean: bool = False
    timing_met: bool = False
    power_within_budget: bool = False
    area_within_limit: bool = False
    drc_clean: bool = False
    lvs_clean: bool = False
    formal_equiv_pass: bool = False
    cdc_clean: bool = False
    ir_drop_ok: bool = False
    esd_reviewed: bool = False

    @property
    def readiness_score(self) -> float:
        """Return the fraction of the ten required checks that passed."""
        checks = [
            self.synthesis_clean,
            self.timing_met,
            self.power_within_budget,
            self.area_within_limit,
            self.drc_clean,
            self.lvs_clean,
            self.formal_equiv_pass,
            self.cdc_clean,
            self.ir_drop_ok,
            self.esd_reviewed,
        ]
        return sum(1 for c in checks if c) / len(checks)

    @property
    def is_tape_out_ready(self) -> bool:
        """Return whether all ten readiness checks passed."""
        return self.readiness_score == 1.0

    def failing_checks(self) -> List[str]:
        """Return stable field names for every incomplete readiness check."""
        names = [
            "synthesis_clean",
            "timing_met",
            "power_within_budget",
            "area_within_limit",
            "drc_clean",
            "lvs_clean",
            "formal_equiv_pass",
            "cdc_clean",
            "ir_drop_ok",
            "esd_reviewed",
        ]
        return [n for n in names if not getattr(self, n)]

    def from_signoff(self, summary: SignoffSummary) -> None:
        """Populate from a signoff summary."""
        self.timing_met = summary.timing.passed
        self.power_within_budget = summary.power.passed
        self.area_within_limit = summary.area.passed
        self.drc_clean = summary.drc_clean
        self.lvs_clean = summary.lvs_match
