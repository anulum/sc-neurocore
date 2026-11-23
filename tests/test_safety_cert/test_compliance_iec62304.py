# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIEC62304 from former test_compliance.py

"""Focused suite: TestIEC62304 from former test_compliance.py."""

from __future__ import annotations

import pytest
from sc_neurocore.safety_cert.safety_cert import (
    IEC62304Assessment,
    SILLevel,
    SWClass,
)
from tests.test_safety_cert.compliance_support import (
    _unsafe,
)
from typing import (
    Any,
)


class TestIEC62304:
    """IEC 62304 software safety classes."""

    def test_from_sil_1(self) -> None:
        """SIL 1 maps to class A without unit testing."""
        a = IEC62304Assessment.from_sil(SILLevel.SIL_1)
        assert a.sw_class == SWClass.CLASS_A
        assert not a.requires_unit_testing

    def test_from_sil_3(self) -> None:
        """SIL 3 maps to class C with unit testing."""
        a = IEC62304Assessment.from_sil(SILLevel.SIL_3)
        assert a.sw_class == SWClass.CLASS_C
        assert a.requires_unit_testing
        assert a.requires_architectural_design

    def test_class_b(self) -> None:
        """Class B needs unit testing but no architectural design."""
        a = IEC62304Assessment(sw_class=SWClass.CLASS_B)
        assert a.requires_unit_testing
        assert not a.requires_architectural_design

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sw_class": "B"}, "sw_class"),
            ({"hazard_description": None}, "hazard_description"),
            ({"risk_control_measures": "measure"}, "risk_control_measures"),
            ({"risk_control_measures": ["", "measure"]}, "risk_control_measures"),
        ],
    )
    def test_iec62304_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        """The assessment rejects each invalid field."""
        values = {
            "sw_class": SWClass.CLASS_B,
            "hazard_description": "hazard",
            "risk_control_measures": ["measure 1"],
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            IEC62304Assessment(**_unsafe(values))
