# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProofTestCoverage from former test_formal_evidence.py

"""Focused suite: TestProofTestCoverage from former test_formal_evidence.py."""

from __future__ import annotations

from tests.test_safety_cert.formal_evidence_support import *  # noqa: F403

class TestProofTestCoverage:
    def test_full_coverage(self) -> None:
        props = [
            FormalProperty("P1", "m", "d", "assert", "proven"),
            FormalProperty("P2", "m", "d", "assert", "proven"),
        ]
        assert ProofTestCoverage.coverage_from_proofs(props) == 1.0

    def test_partial_coverage(self) -> None:
        props = [
            FormalProperty("P1", "m", "d", "assert", "proven"),
            FormalProperty("P2", "m", "d", "assert", "failed"),
        ]
        assert abs(ProofTestCoverage.coverage_from_proofs(props) - 0.5) < 0.01

    def test_uncovered_modules(self) -> None:
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        uncovered = ProofTestCoverage.uncovered_modules(props, ["neuron", "encoder"])
        assert uncovered == ["encoder"]

    def test_uncovered_modules_deduplicates_preserving_order(self) -> None:
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        uncovered = ProofTestCoverage.uncovered_modules(props, ["encoder", "encoder", "decoder"])
        assert uncovered == ["encoder", "decoder"]

    def test_dc_to_sil(self) -> None:
        assert ProofTestCoverage.dc_to_sil(0.99).value >= 3
        assert ProofTestCoverage.dc_to_sil(0.97) == SILLevel.SIL_3
        assert ProofTestCoverage.dc_to_sil(0.5) == SILLevel.SIL_1

    @pytest.mark.parametrize("dc", [-0.1, 1.1, float("nan"), float("inf"), True, "0.9"])
    def test_dc_to_sil_rejects_invalid_contracts(self, dc: Any) -> None:
        with pytest.raises(ValueError, match="dc"):
            ProofTestCoverage.dc_to_sil(_unsafe(dc))

    @pytest.mark.parametrize(
        ("props", "modules", "match"),
        [
            ("invalid", ["neuron"], "properties"),
            ([FormalProperty("P1", "n", "d", "assert", "proven"), "bad"], ["neuron"], "properties"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], "invalid", "all_modules"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], ["", "neuron"], "all_modules"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], [" neuron"], "whitespace"),
        ],
    )
    def test_uncovered_modules_rejects_invalid_contracts(
        self, props: Any, modules: Any, match: Any
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ProofTestCoverage.uncovered_modules(_unsafe(props), _unsafe(modules))

    def test_uncovered_modules_rejects_corrupted_property_module(self) -> None:
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.module = _unsafe("")
        with pytest.raises(ValueError, match="modules"):
            ProofTestCoverage.uncovered_modules([prop], ["m"])

    def test_uncovered_modules_rejects_corrupted_property_id(self) -> None:
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = _unsafe("")
        with pytest.raises(ValueError, match="prop_id"):
            ProofTestCoverage.uncovered_modules([prop], ["m"])

    @pytest.mark.parametrize(
        "props", ["invalid", [FormalProperty("P1", "n", "d", "assert", "proven"), "bad"]]
    )
    def test_coverage_from_proofs_rejects_invalid_contracts(self, props: Any) -> None:
        with pytest.raises(ValueError, match="properties"):
            ProofTestCoverage.coverage_from_proofs(_unsafe(props))

    def test_coverage_from_proofs_rejects_corrupted_property_status(self) -> None:
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.status = _unsafe("bad")
        with pytest.raises(ValueError, match="statuses"):
            ProofTestCoverage.coverage_from_proofs([prop])

    def test_coverage_from_proofs_rejects_corrupted_property_type(self) -> None:
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.property_type = _unsafe("bad")
        with pytest.raises(ValueError, match="property_type"):
            ProofTestCoverage.coverage_from_proofs([prop])

    def test_coverage_from_proofs_rejects_corrupted_property_id(self) -> None:
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = _unsafe("")
        with pytest.raises(ValueError, match="prop_id"):
            ProofTestCoverage.coverage_from_proofs([prop])
