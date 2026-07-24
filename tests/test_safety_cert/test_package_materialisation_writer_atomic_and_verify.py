# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (writer_atomic_and_verify) from former test_package_materialisation.py

from __future__ import annotations

from tests.test_safety_cert.package_materialisation_support import *  # noqa: F403

def test_atomic_write_emits_verified_manifest_and_private_files(tmp_path: Path) -> None:
    """Materialisation must emit the complete hash-bound six-file bundle."""
    package = _package()
    destination = package.write(tmp_path / "bundle")
    assert destination == tmp_path / "bundle"
    assert {path.name for path in destination.iterdir()} == {
        *package.artifacts(),
        "manifest.json",
    }
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "sc-neurocore.safety-evidence-package"
    assert manifest["content_sha256"] == package.content_sha256()
    assert manifest["package_id"] == package.package_hash
    for artifact in manifest["artifacts"]:
        artifact_path = destination / artifact["filename"]
        payload = artifact_path.read_bytes()
        assert artifact["bytes"] == len(payload)
        assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()
        assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o600

    evidence = EvidenceBag()
    evidence.add_from_package(package)
    assert evidence.file_count == 5
    assert evidence.verify(destination)
    assert all(len(item.sha256) == 64 for item in evidence.items)


def test_evidence_verification_detects_tamper_and_missing_hash(tmp_path: Path) -> None:
    """Digest verification must fail for tampered content or unhashed rows."""
    package = _package()
    destination = package.write(tmp_path / "bundle")
    evidence = EvidenceBag()
    evidence.add_from_package(package)
    (destination / "formal_proof_cert.md").write_text("tampered", encoding="utf-8")
    assert not evidence.verify(destination)
    assert not evidence.verify(tmp_path / "missing")

    unhashed = EvidenceBag()
    unhashed.add(EvidenceItem("traceability_matrix.md", "report", "traceability"))
    assert not unhashed.verify(destination)


def test_writer_refuses_overwrite_stale_id_and_incomplete_metadata(tmp_path: Path) -> None:
    """Writer preconditions must fail before replacing existing evidence."""
    package = _package()
    destination = tmp_path / "bundle"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        package.write(destination)

    package.traceability_report += "\nchanged"
    with pytest.raises(ValueError, match="does not match"):
        package.write(tmp_path / "stale")

    incomplete = CertificationPackage(
        SafetyStandard.IEC_61508,
        SILLevel.SIL_2,
        "trace",
        "fmeda",
        "formal",
        "wcet",
        [],
    )
    with pytest.raises(ValueError, match="generated and package_hash"):
        incomplete.write(tmp_path / "incomplete")


def test_writer_cleans_temporary_directory_after_io_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed write must leave neither destination nor private scratch data."""
    original_write = certification_module._write_bytes

    def failing_write(path: Path, payload: bytes) -> None:
        if path.name == "fmeda_report.md":
            raise OSError("injected write failure")
        original_write(path, payload)

    monkeypatch.setattr(certification_module, "_write_bytes", failing_write)
    with pytest.raises(OSError, match="injected"):
        _package().write(tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
    assert not tuple(tmp_path.glob(".bundle.*"))


def test_writer_rejects_empty_artifact_after_matching_rehash(tmp_path: Path) -> None:
    """A recomputed package ID must not make an empty report writable."""
    package = _package()
    package.traceability_report = ""
    package.package_hash = package.content_sha256()[:32]
    with pytest.raises(ValueError, match="must not be empty"):
        package.write(tmp_path / "empty")


def test_writer_detects_destination_created_during_materialisation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destination race must fail before the atomic rename."""
    destination = tmp_path / "bundle"
    original_write = certification_module._write_bytes

    def racing_write(path: Path, payload: bytes) -> None:
        original_write(path, payload)
        if path.name == "manifest.json":
            destination.mkdir()

    monkeypatch.setattr(certification_module, "_write_bytes", racing_write)
    with pytest.raises(FileExistsError, match="already exists"):
        _package().write(destination)
    assert destination.is_dir()
    assert not tuple(tmp_path.glob(".bundle.*"))
