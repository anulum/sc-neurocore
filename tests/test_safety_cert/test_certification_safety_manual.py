# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSafetyManual from former test_certification.py

"""Focused suite: TestSafetyManual from former test_certification.py."""

from __future__ import annotations

from tests.test_safety_cert.certification_support import *  # noqa: F403


class TestSafetyManual:
    def test_generates(self) -> None:
        manual = SafetyManualGenerator.generate(
            "SC-NeuroCore", SILLevel.SIL_2, ["sc_lif_neuron", "sc_encoder"], 2830.0
        )
        assert "Safety Manual" in manual
        assert "SIL 2" in manual
        assert "sc_lif_neuron" in manual
        assert "2830.0" in manual

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"product_name": ""}, "product_name"),
            ({"sil_level": "SIL_2"}, "sil_level"),
            ({"modules": []}, "modules"),
            ({"modules": ["", "m2"]}, "modules"),
            ({"modules": ["m1", "m1"]}, "duplicates"),
            ({"modules": [" m1", "m2"]}, "whitespace"),
            ({"wcet_ns": -1.0}, "wcet_ns"),
            ({"wcet_ns": float("nan")}, "wcet_ns"),
            ({"wcet_ns": True}, "wcet_ns"),
        ],
    )
    def test_generate_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "product_name": "SC-NeuroCore",
            "sil_level": SILLevel.SIL_2,
            "modules": ["sc_lif_neuron"],
            "wcet_ns": 100.0,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            SafetyManualGenerator.generate(**_unsafe(values))
