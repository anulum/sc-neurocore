# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safety Certification Generator Tests

"""Focused tests for certification."""

from typing import Any

import pytest

from sc_neurocore.safety_cert.safety_cert import (
    CertificationGenerator,
    CertificationPackage,
    ChecklistItem,
    FormalProperty,
    SafetyManualGenerator,
    SafetyStandard,
    SILLevel,
)


def _unsafe(value: object) -> Any:
    """Return a deliberately invalid runtime value for boundary tests."""
    return value


class TestCertificationGenerator:
    def _props(self) -> list[FormalProperty]:
        return [
            FormalProperty(
                "P1",
                "sc_lif_neuron",
                "No overflow",
                "assert",
                "proven",
                sby_file="sc_lif_neuron.sby",
            ),
            FormalProperty(
                "P2", "sc_lif_neuron", "Reset", "assert", "proven", sby_file="sc_lif_neuron.sby"
            ),
            FormalProperty(
                "P3",
                "sc_bitstream_encoder",
                "Cover",
                "cover",
                "proven",
                sby_file="sc_bitstream_encoder.sby",
            ),
        ]

    def test_generate_iec(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["sc_lif_neuron", "sc_bitstream_encoder"],
            self._props(),
        )
        assert isinstance(pkg, CertificationPackage)
        assert pkg.standard == SafetyStandard.IEC_61508
        assert pkg.package_hash != ""

    def test_generate_iso(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.ISO_26262, SILLevel.SIL_3, ["sc_lif_neuron"], self._props()
        )
        assert pkg.standard == SafetyStandard.ISO_26262

    def test_generate_fda(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.FDA_CLASS_III, SILLevel.SIL_2, ["sc_lif_neuron"], self._props()
        )
        assert len(pkg.checklist) == 7

    def test_traceability_in_package(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], self._props()
        )
        assert "Traceability" in pkg.traceability_report

    def test_fmeda_in_package(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], self._props()
        )
        assert "FMEDA" in pkg.fmeda_report

    def test_wcet_in_package(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["sc_lif_neuron"],
            self._props(),
            {"bitstream_length": 512, "num_inputs": 16, "num_neurons": 32, "clock_mhz": 200},
        )
        assert "WCET" in pkg.wcet_report
        assert "cycles" in pkg.wcet_report

    def test_checklist_coverage(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], self._props()
        )
        assert pkg.checklist_coverage == 0.0

    def test_checklist_coverage_rejects_corrupted_internal_state(self) -> None:
        pkg = CertificationPackage(
            standard=SafetyStandard.IEC_61508,
            sil_level=SILLevel.SIL_2,
            traceability_report="t",
            fmeda_report="f",
            formal_cert_report="p",
            wcet_report="w",
            checklist=[],
        )
        pkg.checklist.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="ChecklistItem"):
            _ = pkg.checklist_coverage

    def test_checklist_coverage_rejects_corrupted_status_state(self) -> None:
        pkg = CertificationPackage(
            standard=SafetyStandard.IEC_61508,
            sil_level=SILLevel.SIL_2,
            traceability_report="t",
            fmeda_report="f",
            formal_cert_report="p",
            wcet_report="w",
            checklist=[ChecklistItem("id", "7.4.2", "desc", "formal/", "partial")],
        )
        pkg.checklist[0].status = _unsafe("bad")
        with pytest.raises(ValueError, match="statuses"):
            _ = pkg.checklist_coverage

    def test_package_rejects_corrupted_checklist_status_state(self) -> None:
        item = ChecklistItem("id", "7.4.2", "desc", "formal/", "partial")
        item.status = _unsafe("bad")
        with pytest.raises(ValueError, match="statuses"):
            CertificationPackage(
                standard=SafetyStandard.IEC_61508,
                sil_level=SILLevel.SIL_2,
                traceability_report="t",
                fmeda_report="f",
                formal_cert_report="p",
                wcet_report="w",
                checklist=[item],
            )

    def test_package_rejects_corrupted_checklist_clause_state(self) -> None:
        item = ChecklistItem("id", "7.4.2", "desc", "formal/", "partial")
        item.clause = _unsafe("")
        with pytest.raises(ValueError, match="clauses"):
            CertificationPackage(
                standard=SafetyStandard.IEC_61508,
                sil_level=SILLevel.SIL_2,
                traceability_report="t",
                fmeda_report="f",
                formal_cert_report="p",
                wcet_report="w",
                checklist=[item],
            )

    def test_package_rejects_corrupted_checklist_item_id_state(self) -> None:
        item = ChecklistItem("id", "7.4.2", "desc", "formal/", "partial")
        item.item_id = _unsafe("")
        with pytest.raises(ValueError, match="item_id"):
            CertificationPackage(
                standard=SafetyStandard.IEC_61508,
                sil_level=SILLevel.SIL_2,
                traceability_report="t",
                fmeda_report="f",
                formal_cert_report="p",
                wcet_report="w",
                checklist=[item],
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"standard": "IEC 61508"}, "standard"),
            ({"target_sil": "SIL_2"}, "target_sil"),
            ({"modules": []}, "modules"),
            ({"modules": ["", "m2"]}, "modules"),
            ({"modules": ["sc_lif_neuron", "sc_lif_neuron"]}, "duplicates"),
            ({"modules": [" sc_lif_neuron", "m2"]}, "whitespace"),
            ({"formal_properties": "bad"}, "formal_properties"),
            ({"formal_properties": ["bad"]}, "formal_properties"),
            ({"network_config": "bad"}, "network_config"),
            ({"network_config": {"unsupported": 1}}, "network_config"),
            ({"network_config": {"bitstream_length": 0}}, "bitstream_length"),
            ({"network_config": {"num_inputs": 0}}, "num_inputs"),
            ({"network_config": {"num_neurons": 0}}, "num_neurons"),
            ({"network_config": {"clock_mhz": 0.0}}, "clock_mhz"),
        ],
    )
    def test_generate_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        gen = CertificationGenerator()
        values = {
            "standard": SafetyStandard.IEC_61508,
            "target_sil": SILLevel.SIL_2,
            "modules": ["sc_lif_neuron"],
            "formal_properties": self._props(),
            "network_config": None,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            gen.generate(**_unsafe(values))

    def test_generate_rejects_corrupted_formal_property_module_state(self) -> None:
        gen = CertificationGenerator()
        prop = FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")
        prop.module = _unsafe("")
        with pytest.raises(ValueError, match="formal_properties modules"):
            gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], [prop])

    def test_generate_rejects_corrupted_formal_property_id_state(self) -> None:
        gen = CertificationGenerator()
        prop = FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")
        prop.prop_id = _unsafe("")
        with pytest.raises(ValueError, match="formal_properties prop_id"):
            gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], [prop])

    def test_generate_rejects_formal_property_whitespace_state(self) -> None:
        gen = CertificationGenerator()
        prop = FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")
        prop.module = _unsafe(" sc_lif_neuron")
        with pytest.raises(ValueError, match="whitespace"):
            gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], [prop])

    def test_generate_rejects_corrupted_formal_property_status_state(self) -> None:
        gen = CertificationGenerator()
        prop = FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")
        prop.status = _unsafe("bad")
        with pytest.raises(ValueError, match="formal_properties statuses"):
            gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], [prop])

    def test_generate_rejects_corrupted_formal_property_type_state(self) -> None:
        gen = CertificationGenerator()
        prop = FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")
        prop.property_type = _unsafe("bad")
        with pytest.raises(ValueError, match="formal_properties property_type"):
            gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], [prop])

    def test_generate_rejects_duplicate_formal_property_ids(self) -> None:
        gen = CertificationGenerator()
        with pytest.raises(ValueError, match="duplicate prop_id"):
            gen.generate(
                SafetyStandard.IEC_61508,
                SILLevel.SIL_2,
                ["sc_lif_neuron"],
                [
                    FormalProperty("P1", "sc_lif_neuron", "d1", "assert", "proven"),
                    FormalProperty("P1", "sc_lif_neuron", "d2", "cover", "proven"),
                ],
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"standard": "IEC 61508"}, "standard"),
            ({"sil_level": 2}, "sil_level"),
            ({"traceability_report": None}, "traceability_report"),
            ({"checklist": ["not-item"]}, "checklist"),
            ({"generated": None}, "generated"),
        ],
    )
    def test_certification_package_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "standard": SafetyStandard.IEC_61508,
            "sil_level": SILLevel.SIL_2,
            "traceability_report": "trace",
            "fmeda_report": "fmeda",
            "formal_cert_report": "formal",
            "wcet_report": "wcet",
            "checklist": [
                ChecklistItem(
                    item_id="IEC 61508_7.4.2",
                    clause="7.4.2",
                    description="Formal verification of safety functions",
                    evidence="formal/",
                    status="partial",
                )
            ],
            "package_hash": "",
            "generated": "",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            CertificationPackage(**_unsafe(values))


class TestAdditionalStandards:
    def _props(self) -> list[FormalProperty]:
        return [FormalProperty("P1", "sc_lif_neuron", "No overflow", "assert", "proven")]

    def test_generate_do254(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(SafetyStandard.DO_254, SILLevel.SIL_2, ["sc_lif_neuron"], self._props())
        assert len(pkg.checklist) == 6
        assert pkg.standard == SafetyStandard.DO_254

    def test_generate_en50129(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.EN_50129, SILLevel.SIL_3, ["sc_lif_neuron"], self._props()
        )
        assert len(pkg.checklist) == 6
        assert pkg.standard == SafetyStandard.EN_50129


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


class TestBoundaryContracts:
    def test_empty_checklist_package_reports_zero_fraction(self) -> None:
        pkg = CertificationPackage(
            standard=SafetyStandard.IEC_61508,
            sil_level=SILLevel.SIL_2,
            traceability_report="trace",
            fmeda_report="fmeda",
            formal_cert_report="formal",
            wcet_report="wcet",
            checklist=[],
        )
        assert pkg.checklist_coverage == 0.0

    def test_package_revalidates_clause_after_status_contract(self) -> None:
        item = ChecklistItem("IEC 61508_7.4.2", "7.4.2", "desc", "formal/", "partial")
        item.clause = ""
        with pytest.raises(ValueError, match="checklist clauses"):
            CertificationPackage(
                standard=SafetyStandard.IEC_61508,
                sil_level=SILLevel.SIL_2,
                traceability_report="t",
                fmeda_report="f",
                formal_cert_report="p",
                wcet_report="w",
                checklist=[item],
            )

    def test_certification_generator_rejects_formal_property_id_whitespace(self) -> None:
        prop = FormalProperty(" P1", "neuron", "desc", "assert", "proven")
        with pytest.raises(ValueError, match="prop_id"):
            CertificationGenerator().generate(
                SafetyStandard.IEC_61508, SILLevel.SIL_2, ["neuron"], [prop]
            )

    def test_certification_generator_rejects_boolean_clock_configuration(self) -> None:
        prop = FormalProperty("P1", "neuron", "desc", "assert", "proven")
        with pytest.raises(ValueError, match="clock_mhz"):
            CertificationGenerator().generate(
                SafetyStandard.IEC_61508,
                SILLevel.SIL_2,
                ["neuron"],
                [prop],
                network_config={"clock_mhz": True},
            )
