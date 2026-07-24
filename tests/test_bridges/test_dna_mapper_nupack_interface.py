# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNUPACKInterface from former test_dna_mapper.py

"""Focused suite: TestNUPACKInterface from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestNUPACKInterface:
    """Thermodynamic validation."""

    def test_mfe_returns_tuple(self, nupack_interface: NUPACKInterface) -> None:
        energy, structure = nupack_interface.compute_mfe("ACGTACGT")
        assert isinstance(energy, float)
        assert isinstance(structure, str)

    def test_pair_probabilities_shape(self, nupack_interface: NUPACKInterface) -> None:
        seq = "ACGTACGT"
        probs = nupack_interface.compute_pair_probabilities(seq)
        assert probs.shape == (8, 8)

    def test_fallback_predicts_intramolecular_pairing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dna_mapper, "_HAS_NUPACK", False)
        interface = NUPACKInterface()
        sequence = "GCGCAAAGCGC"

        energy, structure = interface.compute_mfe(sequence)
        probs = interface.compute_pair_probabilities(sequence)

        assert energy < 0.0
        assert "(" in structure and ")" in structure
        assert probs.shape == (len(sequence), len(sequence))
        assert probs[0, -1] > 0.0
        assert probs[1, -2] > 0.0
        assert np.allclose(probs, probs.T)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_fallback_rejects_invalid_bases_and_handles_empty_or_unpairable_sequences(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dna_mapper, "_HAS_NUPACK", False)
        interface = NUPACKInterface()

        assert interface.compute_mfe("") == (0.0, "")
        assert not np.any(interface.compute_pair_probabilities("AAAA"))
        with pytest.raises(ValueError, match="invalid bases"):
            interface.compute_mfe("ACGX")

    def test_nupack_backend_path_uses_module_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeModel:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class FakeStrand:
            def __init__(self, sequence: str, name: str) -> None:
                self.sequence = sequence
                self.name = name

        class FakePairs:
            def to_array(self) -> np.ndarray[Any, Any]:
                return np.array([[0.0, 0.25], [0.25, 0.0]])

        fake_nupack = SimpleNamespace(
            Model=FakeModel,
            Strand=FakeStrand,
            mfe=lambda strands, model: [SimpleNamespace(energy=-3.5, structure="()")],
            pairs=lambda strands, model: FakePairs(),
        )
        monkeypatch.setattr(dna_mapper, "_HAS_NUPACK", True)
        monkeypatch.setattr(dna_mapper, "nupack", fake_nupack)
        interface = NUPACKInterface(temperature_c=25.0, na_concentration_M=0.5)

        assert interface.has_nupack is True
        assert interface.compute_mfe("AT") == (-3.5, "()")
        assert np.allclose(interface.compute_pair_probabilities("AT"), [[0.0, 0.25], [0.25, 0.0]])

    def test_validate_design(
        self,
        nupack_interface: NUPACKInterface,
        simple_and_circuit: DNACircuitDesign,
    ) -> None:
        report = nupack_interface.validate_design(simple_and_circuit)
        assert "valid" in report
        assert "strand_results" in report
        assert "warnings" in report
        assert isinstance(report["valid"], bool)

    def test_validate_design_marks_design_rule_warnings_invalid(self) -> None:
        design = DNACircuitDesign(
            name="invalid_rules",
            input_strands=[DNAStrand(name="poly_a", sequence="AAAAAAA", role="output")],
        )

        report = NUPACKInterface().validate_design(design)

        assert report["valid"] is False
        assert report["warnings"]
