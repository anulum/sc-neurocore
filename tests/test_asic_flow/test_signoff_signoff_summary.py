# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSignoffSummary from former test_signoff.py

"""Focused suite: TestSignoffSummary from former test_signoff.py."""

from __future__ import annotations

from tests.test_asic_flow.signoff_support import *  # noqa: F403


class TestSignoffSummary:
    def test_all_pass(self) -> None:
        s = SignoffSummary(
            timing=SignoffCheckResult("STA", True, ""),
            power=SignoffCheckResult("Power", True, ""),
            area=SignoffCheckResult("Area", True, ""),
            lvs_match=True,
        )
        assert s.all_pass

    def test_drc_failure(self) -> None:
        s = SignoffSummary(
            timing=SignoffCheckResult("STA", True, ""),
            power=SignoffCheckResult("Power", True, ""),
            area=SignoffCheckResult("Area", True, ""),
            drc_violations=[DRCViolation("min_width", 5, "error")],
            lvs_match=True,
        )
        assert not s.drc_clean
        assert not s.all_pass

    def test_to_dict(self) -> None:
        s = SignoffSummary(
            timing=SignoffCheckResult("STA", True, "ok"),
            power=SignoffCheckResult("Power", True, "ok"),
            area=SignoffCheckResult("Area", True, "ok"),
        )
        d = s.to_dict()
        assert "timing" in d
        assert "all_pass" in d
