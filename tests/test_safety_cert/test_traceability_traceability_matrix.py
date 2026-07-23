# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTraceabilityMatrix from former test_traceability.py

"""Focused suite: TestTraceabilityMatrix from former test_traceability.py."""

from __future__ import annotations

from tests.test_safety_cert.traceability_support import *  # noqa: F403

class TestTraceabilityMatrix:
    def test_add_requirement(self) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        tm.add_requirement(req)
        assert "REQ_001" in tm.requirements

    def test_add_requirement_rejects_duplicate_req_id(self) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        tm.add_requirement(req)
        with pytest.raises(ValueError, match="already exists"):
            tm.add_requirement(req)

    def test_link_implementation(self) -> None:
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("REQ_001", "Test", SafetyStandard.IEC_61508))
        assert tm.link_implementation("REQ_001", "hdl/test.v") is True
        assert tm.requirements["REQ_001"].status == "implemented"

    def test_link_implementation_normalises_whitespace_inputs(self) -> None:
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("REQ_001", "Test", SafetyStandard.IEC_61508))
        assert tm.link_implementation(" REQ_001 ", " hdl/test.v ") is True
        assert tm.requirements["REQ_001"].implementation_refs == ["hdl/test.v"]

    def test_link_implementation_is_idempotent_for_same_reference(self) -> None:
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("REQ_001", "Test", SafetyStandard.IEC_61508))
        tm.link_implementation("REQ_001", "hdl/test.v")
        tm.link_implementation("REQ_001", "hdl/test.v")
        assert tm.requirements["REQ_001"].implementation_refs == ["hdl/test.v"]

    def test_link_implementation_rejects_corrupted_requirement_entry(self) -> None:
        tm = TraceabilityMatrix()
        tm.requirements["REQ_001"] = _unsafe("bad")
        with pytest.raises(ValueError, match="Requirement"):
            tm.link_implementation("REQ_001", "hdl/test.v")

    def test_link_verification(self) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        req.implementation_refs = ["hdl/test.v"]
        tm.add_requirement(req)
        tm.link_verification("REQ_001", "formal/test.sby")
        assert tm.requirements["REQ_001"].status == "verified"

    def test_link_verification_normalises_whitespace_inputs(self) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        req.implementation_refs = ["hdl/test.v"]
        tm.add_requirement(req)
        tm.link_verification(" REQ_001 ", " formal/test.sby ")
        assert tm.requirements["REQ_001"].verification_refs == ["formal/test.sby"]

    def test_link_verification_is_idempotent_for_same_reference(self) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        req.implementation_refs = ["hdl/test.v"]
        tm.add_requirement(req)
        tm.link_verification("REQ_001", "formal/test.sby")
        tm.link_verification("REQ_001", "formal/test.sby")
        assert tm.requirements["REQ_001"].verification_refs == ["formal/test.sby"]

    def test_link_verification_rejects_corrupted_requirement_entry(self) -> None:
        tm = TraceabilityMatrix()
        tm.requirements["REQ_001"] = _unsafe("bad")
        with pytest.raises(ValueError, match="Requirement"):
            tm.link_verification("REQ_001", "formal/test.sby")

    def test_coverage(self) -> None:
        tm = TraceabilityMatrix()
        for i in range(4):
            req = Requirement(f"REQ_{i}", "Test", SafetyStandard.IEC_61508)
            if i < 2:
                req.implementation_refs = ["impl"]
                req.verification_refs = ["verif"]
                req.status = "verified"
            tm.add_requirement(req)
        assert abs(tm.coverage - 0.5) < 0.01

    def test_coverage_rejects_corrupted_internal_state(self) -> None:
        tm = TraceabilityMatrix()
        tm.requirements["R1"] = _unsafe("bad")
        with pytest.raises(ValueError, match="Requirement"):
            _ = tm.coverage

    def test_coverage_rejects_corrupted_requirement_status(self) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("R1", "Test", SafetyStandard.IEC_61508)
        req.status = _unsafe("bad")
        tm.add_requirement(req)
        with pytest.raises(ValueError, match="statuses"):
            _ = tm.coverage

    def test_open_count(self) -> None:
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("R1", "Test", SafetyStandard.IEC_61508))
        assert tm.open_count == 1

    def test_open_count_rejects_corrupted_internal_state(self) -> None:
        tm = TraceabilityMatrix()
        tm.requirements["R1"] = _unsafe("bad")
        with pytest.raises(ValueError, match="Requirement"):
            _ = tm.open_count

    def test_open_count_rejects_corrupted_requirement_status(self) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("R1", "Test", SafetyStandard.IEC_61508)
        req.status = _unsafe("bad")
        tm.add_requirement(req)
        with pytest.raises(ValueError, match="statuses"):
            _ = tm.open_count

    def test_link_nonexistent(self) -> None:
        tm = TraceabilityMatrix()
        assert tm.link_implementation("NOPE", "x.v") is False

    def test_generate_report(self) -> None:
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("R1", "Test", SafetyStandard.IEC_61508))
        report = tm.generate_report()
        assert "Traceability Matrix" in report
        assert "R1" in report

    def test_generate_report_orders_rows_by_requirement_id(self) -> None:
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("R2", "Test", SafetyStandard.IEC_61508))
        tm.add_requirement(Requirement("R1", "Test", SafetyStandard.IEC_61508))
        lines = tm.generate_report().splitlines()
        req_rows = [
            line for line in lines if line.startswith("| R") and (not line.startswith("| Req ID"))
        ]
        assert req_rows == sorted(req_rows)

    def test_generate_report_rejects_corrupted_internal_state(self) -> None:
        tm = TraceabilityMatrix()
        tm.requirements["R1"] = _unsafe("bad")
        with pytest.raises(ValueError, match="Requirement"):
            tm.generate_report()

    def test_generate_report_rejects_corrupted_requirement_status(self) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("R1", "Test", SafetyStandard.IEC_61508)
        req.status = _unsafe("bad")
        tm.add_requirement(req)
        with pytest.raises(ValueError, match="statuses"):
            tm.generate_report()

    @pytest.mark.parametrize("property_name", ["coverage", "open_count"])
    def test_traceability_properties_reject_requirement_key_mismatch(
        self, property_name: Any
    ) -> None:
        tm = TraceabilityMatrix()
        tm.requirements["R1"] = Requirement("R2", "Test", SafetyStandard.IEC_61508)
        with pytest.raises(ValueError, match="key mismatch"):
            _ = getattr(tm, property_name)

    def test_generate_report_rejects_requirement_key_mismatch(self) -> None:
        tm = TraceabilityMatrix()
        tm.requirements["R1"] = Requirement("R2", "Test", SafetyStandard.IEC_61508)
        with pytest.raises(ValueError, match="key mismatch"):
            tm.generate_report()

    @pytest.mark.parametrize(
        ("field_name", "bad_value", "match"),
        [("standard", "IEC 61508", "SafetyStandard"), ("sil_level", 2, "SILLevel")],
    )
    def test_generate_report_rejects_invalid_requirement_types(
        self, field_name: Any, bad_value: Any, match: Any
    ) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("R1", "Test", SafetyStandard.IEC_61508)
        setattr(req, field_name, bad_value)
        tm.add_requirement(req)
        with pytest.raises(ValueError, match=match):
            tm.generate_report()

    def test_add_requirement_rejects_invalid_contract(self) -> None:
        tm = TraceabilityMatrix()
        with pytest.raises(ValueError, match="req"):
            tm.add_requirement(_unsafe("bad"))

    @pytest.mark.parametrize(
        ("req_id", "impl_ref", "match"), [("", "hdl/a.v", "req_id"), ("REQ_1", "", "impl_ref")]
    )
    def test_link_implementation_rejects_invalid_contracts(
        self, req_id: Any, impl_ref: Any, match: Any
    ) -> None:
        tm = TraceabilityMatrix()
        with pytest.raises(ValueError, match=match):
            tm.link_implementation(req_id, impl_ref)

    @pytest.mark.parametrize(
        ("req_id", "verif_ref", "match"),
        [("", "formal/a.sby", "req_id"), ("REQ_1", "", "verif_ref")],
    )
    def test_link_verification_rejects_invalid_contracts(
        self, req_id: Any, verif_ref: Any, match: Any
    ) -> None:
        tm = TraceabilityMatrix()
        with pytest.raises(ValueError, match=match):
            tm.link_verification(req_id, verif_ref)

    def test_update_status_rejects_invalid_requirement_object(self) -> None:
        tm = TraceabilityMatrix()
        with pytest.raises(ValueError, match="req"):
            tm._update_status(_unsafe("bad"))

    def test_update_status_rejects_corrupted_reference_entries(self) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("R1", "d", SafetyStandard.IEC_61508)
        req.implementation_refs = _unsafe([""])
        with pytest.raises(ValueError, match="implementation_refs"):
            tm._update_status(req)

    def test_update_status_downgrades_to_open_when_implementation_removed(self) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("R1", "d", SafetyStandard.IEC_61508)
        req.status = "verified"
        req.implementation_refs = []
        req.verification_refs = ["formal/test.sby"]
        tm._update_status(req)
        assert req.status == "open"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"req_id": ""}, "req_id"),
            ({"description": ""}, "description"),
            ({"standard": "IEC 61508"}, "standard"),
            ({"sil_level": 2}, "sil_level"),
            ({"status": ""}, "status"),
            ({"status": "done"}, "status"),
            ({"implementation_refs": ["", "hdl/top.sv"]}, "implementation_refs"),
            ({"verification_refs": ["", "formal/top.sby"]}, "verification_refs"),
        ],
    )
    def test_requirement_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "req_id": "REQ_100",
            "description": "desc",
            "standard": SafetyStandard.IEC_61508,
            "sil_level": SILLevel.SIL_2,
            "implementation_refs": ["hdl/top.sv"],
            "verification_refs": ["formal/top.sby"],
            "status": "open",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            Requirement(**_unsafe(values))
