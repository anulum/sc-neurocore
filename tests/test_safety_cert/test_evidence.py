# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safety Certification Generator Tests

"""Focused tests for evidence."""

from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.safety_cert.safety_cert import (
    CertificationGenerator,
    EvidenceBag,
    EvidenceItem,
    FormalProperty,
    SafetyStandard,
    SILLevel,
)


def _unsafe(value: object) -> Any:
    """Return a deliberately invalid runtime value for boundary tests."""
    return value


class TestEvidenceBag:
    def test_add_items(self) -> None:
        bag = EvidenceBag()
        bag.add(EvidenceItem("test.md", "report", "test"))
        assert bag.file_count == 1

    def test_file_count_rejects_corrupted_internal_state(self) -> None:
        bag = EvidenceBag()
        bag.items.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="EvidenceItem"):
            _ = bag.file_count

    def test_from_package(self) -> None:
        gen = CertificationGenerator()
        props = [FormalProperty("P1", "m", "d", "assert", "proven")]
        pkg = gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["m"], props)
        bag = EvidenceBag()
        bag.add_from_package(pkg)
        assert bag.file_count == 5

    def test_manifest(self) -> None:
        bag = EvidenceBag()
        bag.add(EvidenceItem("x.md", "formal", "proof"))
        m = bag.manifest()
        assert "Evidence Bag" in m
        assert "x.md" in m

    def test_manifest_rejects_corrupted_internal_state(self) -> None:
        bag = EvidenceBag()
        bag.items.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="EvidenceItem"):
            bag.manifest()

    def test_hash(self) -> None:
        bag = EvidenceBag()
        bag.add(EvidenceItem("x.md", "formal", "proof"))
        assert len(bag.compute_hashes()) == 32

    def test_hash_rejects_corrupted_internal_state(self) -> None:
        bag = EvidenceBag()
        bag.items.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="EvidenceItem"):
            bag.compute_hashes()

    def test_hash_rejects_corrupted_duplicate_filenames_state(self) -> None:
        bag = EvidenceBag()
        bag.items = _unsafe(
            [EvidenceItem("x.md", "formal", "a"), EvidenceItem("x.md", "report", "b")]
        )
        with pytest.raises(ValueError, match="unique"):
            bag.compute_hashes()

    def test_hash_changes_with_declared_sha256(self) -> None:
        bag_a = EvidenceBag()
        bag_a.add(EvidenceItem("x.md", "formal", "proof", sha256="a" * 64))
        bag_b = EvidenceBag()
        bag_b.add(EvidenceItem("x.md", "formal", "proof", sha256="b" * 64))
        assert bag_a.compute_hashes() != bag_b.compute_hashes()

    def test_add_rejects_invalid_item(self) -> None:
        bag = EvidenceBag()
        with pytest.raises(ValueError, match="item"):
            bag.add(_unsafe("bad"))

    def test_add_rejects_duplicate_filenames(self) -> None:
        bag = EvidenceBag()
        bag.add(EvidenceItem("x.md", "formal", "proof"))
        with pytest.raises(ValueError, match="unique"):
            bag.add(EvidenceItem("x.md", "report", "duplicate"))

    def test_add_from_package_rejects_invalid_package(self) -> None:
        bag = EvidenceBag()
        with pytest.raises(ValueError, match="pkg"):
            bag.add_from_package(_unsafe("bad"))

    def test_add_from_package_rejects_corrupted_package_checklist_state(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["sc_lif_neuron"],
            [FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")],
        )
        pkg.checklist.append(_unsafe("bad"))
        bag = EvidenceBag()
        with pytest.raises(ValueError, match="ChecklistItem"):
            bag.add_from_package(pkg)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"filename": ""}, "filename"),
            ({"category": "unsafe"}, "category"),
            ({"description": ""}, "description"),
            ({"sha256": None}, "sha256"),
            ({"sha256": "not_hex"}, "hexadecimal"),
        ],
    )
    def test_evidence_item_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "filename": "formal_proof_cert.md",
            "category": "formal",
            "description": "Formal proof certificate",
            "sha256": "",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            EvidenceItem(**_unsafe(values))

    def test_verify_rejects_invalid_directory_type_and_missing_file(
        self,
        tmp_path: Path,
    ) -> None:
        bag = EvidenceBag()
        bag.add(EvidenceItem("missing.md", "report", "missing", "a" * 64))
        with pytest.raises(ValueError, match="directory"):
            bag.verify(_unsafe(42))
        assert not bag.verify(tmp_path)
