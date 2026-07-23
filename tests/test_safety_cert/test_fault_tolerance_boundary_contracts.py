# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoundaryContracts from former test_fault_tolerance.py

"""Focused suite: TestBoundaryContracts from former test_fault_tolerance.py."""

from __future__ import annotations

from tests.test_safety_cert.fault_tolerance_support import *  # noqa: F403

class TestBoundaryContracts:
    def test_common_cause_default_defence_contracts(self, monkeypatch: Any) -> None:
        original = CCFAnalysis.DEFAULT_DEFENCES
        monkeypatch.setattr(CCFAnalysis, "DEFAULT_DEFENCES", ["bad"])
        with pytest.raises(ValueError, match="DEFAULT_DEFENCES"):
            CCFAnalysis()
        monkeypatch.setattr(CCFAnalysis, "DEFAULT_DEFENCES", original)
        ccf = CCFAnalysis()
        ccf.defences[0].beta_reduction = _unsafe(float("nan"))
        with pytest.raises(ValueError, match="beta_reduction"):
            _ = ccf.beta_factor

    @pytest.mark.parametrize(
        ("sff", "target", "expected"),
        [
            (0.99, SILLevel.SIL_4, HFTLevel.HFT_1),
            (0.9, SILLevel.SIL_2, HFTLevel.HFT_0),
            (0.9, SILLevel.SIL_3, HFTLevel.HFT_1),
            (0.9, SILLevel.SIL_4, HFTLevel.HFT_2),
            (0.6, SILLevel.SIL_1, HFTLevel.HFT_0),
            (0.6, SILLevel.SIL_2, HFTLevel.HFT_1),
            (0.6, SILLevel.SIL_3, HFTLevel.HFT_2),
            (0.59, SILLevel.SIL_1, HFTLevel.HFT_1),
        ],
    )
    def test_hft_threshold_boundaries(self, sff: Any, target: Any, expected: Any) -> None:
        assert HFTAssessment(sff=sff, target_sil=target).required_hft == expected
