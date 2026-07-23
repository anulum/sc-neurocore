# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormalProofCertificate from former test_formal_evidence.py

"""Focused suite: TestFormalProofCertificate from former test_formal_evidence.py."""

from __future__ import annotations

from tests.test_safety_cert.formal_evidence_support import *  # noqa: F403

class TestFormalProofCertificate:
    def _props(self) -> list[FormalProperty]:
        return [
            FormalProperty("P1", "sc_lif_neuron", "No overflow", "assert", "proven"),
            FormalProperty("P2", "sc_lif_neuron", "Reset works", "assert", "proven"),
            FormalProperty("P3", "sc_encoder", "Cover fire", "cover", "proven"),
            FormalProperty("P4", "sc_dense", "Weight range", "assert", "failed"),
        ]

    def test_proven_count(self) -> None:
        cert = FormalProofCertificate(properties=self._props())
        assert cert.proven_count == 3

    def test_proven_count_rejects_corrupted_internal_state(self) -> None:
        cert = FormalProofCertificate()
        cert.properties.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FormalProperty"):
            _ = cert.proven_count

    def test_proven_count_rejects_corrupted_property_status(self) -> None:
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.status = _unsafe("bad")
        cert.properties.append(prop)
        with pytest.raises(ValueError, match="statuses"):
            _ = cert.proven_count

    def test_pass_rate(self) -> None:
        cert = FormalProofCertificate(properties=self._props())
        assert abs(cert.pass_rate - 0.75) < 0.01

    def test_total_count_rejects_corrupted_internal_state(self) -> None:
        cert = FormalProofCertificate()
        cert.properties.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FormalProperty"):
            _ = cert.total_count

    def test_total_count_rejects_corrupted_property_id(self) -> None:
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = _unsafe("")
        cert.properties.append(prop)
        with pytest.raises(ValueError, match="prop_id"):
            _ = cert.total_count

    def test_compute_hash(self) -> None:
        cert = FormalProofCertificate(properties=self._props())
        h = cert.compute_hash()
        assert len(h) == 32

    def test_add_property_rejects_invalid_contract(self) -> None:
        cert = FormalProofCertificate()
        with pytest.raises(ValueError, match="prop"):
            cert.add_property(_unsafe("bad"))

    def test_hash_deterministic(self) -> None:
        cert = FormalProofCertificate(properties=self._props())
        assert cert.compute_hash() == cert.compute_hash()

    def test_compute_hash_rejects_duplicate_property_ids(self) -> None:
        cert = FormalProofCertificate(
            properties=[
                FormalProperty("P1", "m1", "d1", "assert", "proven"),
                FormalProperty("P1", "m2", "d2", "assert", "proven"),
            ]
        )
        with pytest.raises(ValueError, match="duplicate"):
            cert.compute_hash()

    def test_compute_hash_rejects_corrupted_internal_state(self) -> None:
        cert = FormalProofCertificate()
        cert.properties.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FormalProperty"):
            cert.compute_hash()

    def test_compute_hash_rejects_corrupted_property_module(self) -> None:
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.module = _unsafe("")
        cert.properties.append(prop)
        with pytest.raises(ValueError, match="modules"):
            cert.compute_hash()

    def test_generate_report(self) -> None:
        cert = FormalProofCertificate(properties=self._props())
        report = cert.generate_report()
        assert "Formal Proof Certificate" in report
        assert "P1" in report

    def test_generate_report_rejects_corrupted_internal_state(self) -> None:
        cert = FormalProofCertificate()
        cert.properties.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FormalProperty"):
            cert.generate_report()

    def test_generate_report_rejects_corrupted_property_fields(self) -> None:
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = _unsafe("")
        cert.properties.append(prop)
        with pytest.raises(ValueError, match="prop_id"):
            cert.generate_report()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"generation_timestamp": None}, "generation_timestamp"),
            ({"tool_version": ""}, "tool_version"),
            ({"certificate_hash": None}, "certificate_hash"),
            ({"properties": ["not-prop"]}, "properties"),
        ],
    )
    def test_formal_proof_certificate_rejects_invalid_contracts(
        self, kwargs: Any, match: Any
    ) -> None:
        values = {
            "properties": self._props(),
            "generation_timestamp": "",
            "tool_version": "SymbiYosys",
            "certificate_hash": "",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            FormalProofCertificate(**_unsafe(values))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"prop_id": ""}, "prop_id"),
            ({"module": ""}, "module"),
            ({"description": ""}, "description"),
            ({"property_type": "prove"}, "property_type"),
            ({"status": "ok"}, "status"),
            ({"engine": ""}, "engine"),
            ({"depth": -1}, "depth"),
            ({"depth": True}, "depth"),
            ({"sby_file": None}, "sby_file"),
        ],
    )
    def test_formal_property_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "prop_id": "P1",
            "module": "sc_lif_neuron",
            "description": "desc",
            "property_type": "assert",
            "status": "proven",
            "engine": "SymbiYosys",
            "depth": 20,
            "sby_file": "sc_lif_neuron.sby",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            FormalProperty(**_unsafe(values))
