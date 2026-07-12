# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safety Certification Generator Tests

"""Focused tests for fault tolerance."""

from typing import Any

import pytest

from sc_neurocore.safety_cert.safety_cert import (
    CCFAnalysis,
    CCFDefence,
    HFTAssessment,
    HFTLevel,
    SILLevel,
)


def _unsafe(value: object) -> Any:
    """Return a deliberately invalid runtime value for boundary tests."""
    return value


class TestCCFAnalysis:
    def test_default_beta(self) -> None:
        ccf = CCFAnalysis()
        assert ccf.beta_factor == 0.1

    def test_beta_factor_rejects_corrupted_internal_state(self) -> None:
        ccf = CCFAnalysis()
        ccf.defences.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="CCFDefence"):
            _ = ccf.beta_factor

    def test_mark_implemented(self) -> None:
        ccf = CCFAnalysis()
        assert ccf.mark_implemented("D1") is True
        assert ccf.implemented_count == 1
        assert ccf.beta_factor < 0.1

    def test_implemented_count_rejects_corrupted_internal_state(self) -> None:
        ccf = CCFAnalysis()
        ccf.defences.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="CCFDefence"):
            _ = ccf.implemented_count

    def test_mark_implemented_normalises_whitespace(self) -> None:
        ccf = CCFAnalysis()
        assert ccf.mark_implemented(" D1 ") is True
        assert ccf.implemented_count == 1

    def test_all_implemented(self) -> None:
        ccf = CCFAnalysis()
        for d in ccf.defences:
            ccf.mark_implemented(d.defence_id)
        assert ccf.beta_factor < 0.02

    def test_sil_compatible(self) -> None:
        ccf = CCFAnalysis()
        assert ccf.sil_compatible(SILLevel.SIL_1) is True
        assert ccf.sil_compatible(SILLevel.SIL_3) is False

    def test_sil_compatible_sil4_threshold(self) -> None:
        ccf = CCFAnalysis()
        for defence in ccf.defences:
            ccf.mark_implemented(defence.defence_id)
        assert ccf.sil_compatible(SILLevel.SIL_4) is True

    def test_sil_compatible_rejects_invalid_target_sil(self) -> None:
        ccf = CCFAnalysis()
        with pytest.raises(ValueError, match="target_sil"):
            ccf.sil_compatible(_unsafe("SIL_2"))

    def test_mark_invalid(self) -> None:
        ccf = CCFAnalysis()
        assert ccf.mark_implemented("NOPE") is False

    def test_init_rejects_duplicate_default_defence_ids(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            CCFAnalysis,
            "DEFAULT_DEFENCES",
            [CCFDefence("D1", "a", "separation", 0.01), CCFDefence("D1", "b", "diversity", 0.01)],
        )
        with pytest.raises(ValueError, match="duplicate"):
            CCFAnalysis()

    def test_mark_implemented_rejects_invalid_defence_id(self) -> None:
        ccf = CCFAnalysis()
        with pytest.raises(ValueError, match="defence_id"):
            ccf.mark_implemented("")

    def test_mark_implemented_rejects_corrupted_internal_state(self) -> None:
        ccf = CCFAnalysis()
        ccf.defences.insert(_unsafe(0), _unsafe("bad"))
        with pytest.raises(ValueError, match="CCFDefence"):
            ccf.mark_implemented("D1")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"defence_id": ""}, "defence_id"),
            ({"description": ""}, "description"),
            ({"category": "other"}, "category"),
            ({"beta_reduction": -0.1}, "beta_reduction"),
            ({"beta_reduction": 1.1}, "beta_reduction"),
            ({"beta_reduction": float("nan")}, "beta_reduction"),
            ({"beta_reduction": True}, "beta_reduction"),
            ({"implemented": "yes"}, "implemented"),
        ],
    )
    def test_ccf_defence_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "defence_id": "D1",
            "description": "Physical separation",
            "category": "separation",
            "beta_reduction": 0.01,
            "implemented": False,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            CCFDefence(**_unsafe(values))


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
