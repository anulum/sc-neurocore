# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safety Certification Generator Tests

"""Focused tests for change impact."""

from typing import Any

import pytest

from sc_neurocore.safety_cert.safety_cert import (
    ChangeImpactTracker,
    ChangeRecord,
)


def _unsafe(value: object) -> Any:
    """Return a deliberately invalid runtime value for boundary tests."""
    return value


class TestChangeImpactTracker:
    def test_add_low_risk(self) -> None:
        ct = ChangeImpactTracker()
        ct.add_change(ChangeRecord("C1", "fix typo", ["neuron"], ["R1"], "low"))
        assert not ct.needs_re_certification

    def test_high_risk_triggers_recert(self) -> None:
        ct = ChangeImpactTracker()
        ct.add_change(ChangeRecord("C1", "redesign LIF", ["neuron"], ["R1"], "high"))
        assert ct.needs_re_certification
        assert ct.high_risk_count == 1

    def test_high_risk_count_rejects_corrupted_internal_state(self) -> None:
        ct = ChangeImpactTracker()
        ct.changes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="ChangeRecord"):
            _ = ct.high_risk_count

    def test_high_risk_count_rejects_corrupted_risk_level_state(self) -> None:
        ct = ChangeImpactTracker()
        change = ChangeRecord("C1", "desc", ["neuron"], ["R1"], "low")
        change.risk_level = _unsafe("bad")
        ct.add_change(change)
        with pytest.raises(ValueError, match="risk_level"):
            _ = ct.high_risk_count

    def test_affected_reqs(self) -> None:
        ct = ChangeImpactTracker()
        ct.add_change(ChangeRecord("C1", "a", [], ["R1", "R2"]))
        ct.add_change(ChangeRecord("C2", "b", [], ["R2", "R3"]))
        assert ct.affected_requirements() == ["R1", "R2", "R3"]

    def test_affected_requirements_rejects_corrupted_internal_state(self) -> None:
        ct = ChangeImpactTracker()
        ct.changes.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="ChangeRecord"):
            ct.affected_requirements()

    def test_affected_requirements_rejects_corrupted_requirement_ids(self) -> None:
        ct = ChangeImpactTracker()
        change = ChangeRecord("C1", "desc", ["neuron"], ["R1"], "low")
        change.affected_reqs = _unsafe(["R1", ""])
        ct.add_change(change)
        with pytest.raises(ValueError, match="affected_reqs"):
            ct.affected_requirements()

    def test_affected_requirements_rejects_corrupted_requirement_container(self) -> None:
        ct = ChangeImpactTracker()
        change = ChangeRecord("C1", "desc", ["neuron"], ["R1"], "low")
        change.affected_reqs = _unsafe("R1")
        ct.add_change(change)
        with pytest.raises(ValueError, match="affected_reqs"):
            ct.affected_requirements()

    def test_affected_requirements_rejects_corrupted_change_id_state(self) -> None:
        ct = ChangeImpactTracker()
        change = ChangeRecord("C1", "desc", ["neuron"], ["R1"], "low")
        change.change_id = _unsafe("")
        ct.add_change(change)
        with pytest.raises(ValueError, match="change_id"):
            ct.affected_requirements()

    def test_add_change_rejects_invalid_contract(self) -> None:
        ct = ChangeImpactTracker()
        with pytest.raises(ValueError, match="change"):
            ct.add_change(_unsafe("bad"))

    def test_add_change_rejects_duplicate_change_ids(self) -> None:
        ct = ChangeImpactTracker()
        ct.add_change(ChangeRecord("C1", "a", ["n"], ["R1"], "low"))
        with pytest.raises(ValueError, match="unique"):
            ct.add_change(ChangeRecord("C1", "b", ["n"], ["R2"], "medium"))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"change_id": ""}, "change_id"),
            ({"description": ""}, "description"),
            ({"risk_level": "critical"}, "risk_level"),
            ({"re_verification_needed": "yes"}, "re_verification_needed"),
            ({"affected_modules": "mod1"}, "affected_modules"),
            ({"affected_modules": ["", "mod"]}, "affected_modules"),
            ({"affected_modules": [" mod"]}, "affected_modules"),
            ({"affected_modules": ["mod", "mod"]}, "affected_modules"),
            ({"affected_reqs": "R1"}, "affected_reqs"),
            ({"affected_reqs": ["", "R1"]}, "affected_reqs"),
            ({"affected_reqs": [" R1"]}, "affected_reqs"),
            ({"affected_reqs": ["R1", "R1"]}, "affected_reqs"),
        ],
    )
    def test_change_record_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "change_id": "C1",
            "description": "desc",
            "affected_modules": ["mod1"],
            "affected_reqs": ["R1"],
            "risk_level": "low",
            "re_verification_needed": False,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ChangeRecord(**_unsafe(values))
