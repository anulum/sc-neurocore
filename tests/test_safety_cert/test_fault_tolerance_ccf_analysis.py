# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCCFAnalysis from former test_fault_tolerance.py

"""Focused suite: TestCCFAnalysis from former test_fault_tolerance.py."""

from __future__ import annotations

from tests.test_safety_cert.fault_tolerance_support import *  # noqa: F403

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
