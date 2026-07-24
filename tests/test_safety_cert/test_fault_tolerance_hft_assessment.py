# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHFTAssessment from former test_fault_tolerance.py

"""Focused suite: TestHFTAssessment from former test_fault_tolerance.py."""

from __future__ import annotations

from tests.test_safety_cert.fault_tolerance_support import *  # noqa: F403


class TestHFTAssessment:
    def test_high_sff_low_sil(self) -> None:
        hft = HFTAssessment(sff=0.99, target_sil=SILLevel.SIL_2)
        assert hft.required_hft == HFTLevel.HFT_0
        assert hft.is_simplex_ok

    def test_low_sff_high_sil(self) -> None:
        hft = HFTAssessment(sff=0.5, target_sil=SILLevel.SIL_3)
        assert hft.required_hft == HFTLevel.HFT_2
        assert not hft.is_simplex_ok

    def test_mid_sff(self) -> None:
        hft = HFTAssessment(sff=0.92, target_sil=SILLevel.SIL_3)
        assert hft.required_hft == HFTLevel.HFT_1

    def test_required_hft_rejects_corrupted_target_sil(self) -> None:
        hft = HFTAssessment(sff=0.92, target_sil=SILLevel.SIL_3)
        hft.target_sil = _unsafe("SIL_3")
        with pytest.raises(ValueError, match="target_sil"):
            _ = hft.required_hft

    def test_required_hft_rejects_corrupted_sff_state(self) -> None:
        hft = HFTAssessment(sff=0.92, target_sil=SILLevel.SIL_3)
        hft.sff = _unsafe(float("nan"))
        with pytest.raises(ValueError, match="sff"):
            _ = hft.required_hft

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sff": -0.1}, "sff"),
            ({"sff": 1.1}, "sff"),
            ({"sff": float("nan")}, "sff"),
            ({"sff": float("inf")}, "sff"),
            ({"sff": True}, "sff"),
            ({"target_sil": "SIL_2"}, "target_sil"),
        ],
    )
    def test_hft_assessment_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {"sff": 0.9, "target_sil": SILLevel.SIL_2}
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            HFTAssessment(**_unsafe(values))
