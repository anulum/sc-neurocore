# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormalGapDetector from former test_formal_evidence.py

"""Focused suite: TestFormalGapDetector from former test_formal_evidence.py."""

from __future__ import annotations

import pytest
from sc_neurocore.safety_cert.safety_cert import (
    FormalProperty,
    FormalPropertyGapDetector,
    PropertyGap,
)
from tests.test_safety_cert.formal_evidence_support import (
    _unsafe,
)
from typing import (
    Any,
)


class TestFormalGapDetector:
    """Finding modules without formal proofs."""

    def test_fully_covered(self) -> None:
        """Proofs for every module leave no gap."""
        props = [
            FormalProperty("P1", "neuron", "d", "assert", "proven"),
            FormalProperty("P2", "neuron", "d", "cover", "proven"),
        ]
        assert FormalPropertyGapDetector.is_fully_covered(props, ["neuron"])

    def test_missing_module(self) -> None:
        """A module without proofs is a gap."""
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        gaps = FormalPropertyGapDetector.detect(props, ["neuron", "encoder"])
        assert len(gaps) >= 1
        assert any(g.module == "encoder" for g in gaps)

    def test_detect_deduplicates_required_modules(self) -> None:
        """A module required twice is one gap."""
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        gaps = FormalPropertyGapDetector.detect(props, ["encoder", "encoder"])
        assert [g.module for g in gaps] == ["encoder"]

    def test_failed_property(self) -> None:
        """A failed property leaves a gap."""
        props = [
            FormalProperty("P1", "neuron", "d", "assert", "failed"),
            FormalProperty("P2", "neuron", "d", "cover", "proven"),
        ]
        gaps = FormalPropertyGapDetector.detect(props, ["neuron"])
        assert len(gaps) == 1
        assert gaps[0].proven_properties == 1

    def test_gap_coverage(self) -> None:
        """One of two proven gives a coverage of 0.5."""
        gap = PropertyGap("m", 4, 2, [])
        assert gap.coverage == 0.5

    @pytest.mark.parametrize(
        ("properties", "required_modules", "match"),
        [
            ("bad", ["neuron"], "properties"),
            ([FormalProperty("P1", "n", "d", "assert", "proven"), "bad"], ["neuron"], "properties"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], "bad", "required_modules"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], [""], "required_modules"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], [" neuron"], "whitespace"),
        ],
    )
    def test_detect_rejects_invalid_contracts(
        self, properties: Any, required_modules: Any, match: Any
    ) -> None:
        """Gap detection rejects each invalid argument."""
        with pytest.raises(ValueError, match=match):
            FormalPropertyGapDetector.detect(_unsafe(properties), _unsafe(required_modules))

    def test_detect_rejects_corrupted_property_type_state(self) -> None:
        """A corrupted property type is rejected."""
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.property_type = _unsafe("bad")
        with pytest.raises(ValueError, match="property_type"):
            FormalPropertyGapDetector.detect([prop], ["neuron"])

    def test_detect_rejects_corrupted_property_status_state(self) -> None:
        """A corrupted property status is rejected."""
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.status = _unsafe("bad")
        with pytest.raises(ValueError, match="statuses"):
            FormalPropertyGapDetector.detect([prop], ["neuron"])

    def test_detect_rejects_corrupted_property_module_state(self) -> None:
        """A corrupted property module is rejected."""
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.module = _unsafe("")
        with pytest.raises(ValueError, match="modules"):
            FormalPropertyGapDetector.detect([prop], ["neuron"])

    def test_detect_rejects_corrupted_property_id_state(self) -> None:
        """A corrupted property id is rejected."""
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.prop_id = _unsafe("")
        with pytest.raises(ValueError, match="prop_id"):
            FormalPropertyGapDetector.detect([prop], ["neuron"])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"module": ""}, "module"),
            ({"total_properties": -1}, "total_properties"),
            ({"total_properties": True}, "total_properties"),
            ({"proven_properties": -1}, "proven_properties"),
            ({"proven_properties": 3, "total_properties": 2}, "proven_properties cannot exceed"),
            ({"missing_types": ["", "cover"]}, "missing_types"),
        ],
    )
    def test_property_gap_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        """A gap record rejects each invalid field."""
        values = {
            "module": "neuron",
            "total_properties": 2,
            "proven_properties": 1,
            "missing_types": ["assert"],
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            PropertyGap(**_unsafe(values))
