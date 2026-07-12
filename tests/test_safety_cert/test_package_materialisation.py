# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — safety-evidence package integration tests

"""Exercise fail-closed assembly, hashing, atomic writes, and verification."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import sc_neurocore.safety_cert.certification as certification_module
from sc_neurocore.safety_cert import (
    CertificationGenerator,
    CertificationPackage,
    ChecklistItem,
    EvidenceBag,
    EvidenceItem,
    FailureCategory,
    FailureMode,
    FormalProofCertificate,
    FormalProperty,
    SILLevel,
    SafetyManualGenerator,
    SafetyStandard,
)

_GENERATED_AT = "2026-07-12T18:30:00+00:00"
_NETWORK_CONFIG = {
    "bitstream_length": 256,
    "num_inputs": 8,
    "num_neurons": 16,
    "clock_mhz": 100.0,
}


def _unsafe(value: object) -> Any:
    """Return an invalid runtime value for a deliberate boundary test."""
    return value


def _property() -> FormalProperty:
    return FormalProperty(
        prop_id="P-SAFE-001",
        module="neuron",
        description="Accumulator remains in range",
        property_type="assert",
        status="proven",
        engine="SymbiYosys 2.4.0",
        depth=32,
        sby_file="formal/neuron.sby",
    )


def _package(*, explicit_evidence: bool = True) -> CertificationPackage:
    generator = CertificationGenerator()
    if not explicit_evidence:
        return generator.generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["neuron"],
            [_property()],
            generated_at=_GENERATED_AT,
        )
    return generator.generate(
        SafetyStandard.IEC_61508,
        SILLevel.SIL_2,
        ["neuron"],
        [_property()],
        _NETWORK_CONFIG,
        implementation_evidence={"neuron": ["rtl/neuron.sv"]},
        failure_modes=[
            FailureMode(
                "FM-NEURON-001",
                "neuron",
                "Accumulator register upset",
                FailureCategory.DANGEROUS_DETECTED,
                12.5,
                0.95,
                "Parity monitor; verification report E-17",
            )
        ],
        checklist_evidence={"7.4.2": "evidence/formal-review.md"},
        generated_at=_GENERATED_AT,
    )


def test_default_generator_is_fail_closed() -> None:
    """Missing evidence must remain visible rather than being fabricated."""
    package = _package(explicit_evidence=False)
    assert package.checklist_coverage == 0.0
    assert all(item.status == "not_addressed" and not item.evidence for item in package.checklist)
    assert "Coverage: 0.0% (0/1)" in package.traceability_report
    assert "hdl/neuron.v" not in package.traceability_report
    assert "Status: not assessed" in package.fmeda_report
    assert "Status: not assessed" in package.wcet_report
    assert package.package_hash == package.content_sha256()[:32]


def test_explicit_evidence_flows_through_every_report() -> None:
    """Caller evidence and assumptions must reach their owning artifacts."""
    package = _package()
    assert "Coverage: 100.0% (1/1)" in package.traceability_report
    assert "rtl/neuron.sv" in package.traceability_report
    assert "| REQ_001 | IEC 61508 | SIL 2 | verified | 1 | 1 |" in package.traceability_report
    assert "12.5 FIT" in package.fmeda_report
    assert "Input-derived bound" in package.wcet_report
    assert package.checklist_coverage == pytest.approx(1 / 7)
    assert "evidence/formal-review.md" in package.checklist_report()


def test_fixed_timestamp_makes_package_content_reproducible() -> None:
    """Equivalent inputs must yield identical package and artifact digests."""
    first = _package()
    second = _package()
    assert first.artifacts() == second.artifacts()
    assert first.content_sha256() == second.content_sha256()
    assert first.package_hash == second.package_hash


def test_formal_digest_covers_all_material_fields_and_tool_version() -> None:
    """Changing any material proof field must change the full digest."""
    base = _property()
    variants = [
        replace(base, prop_id="P-SAFE-002"),
        replace(base, module="encoder"),
        replace(base, description="Reset dominates"),
        replace(base, property_type="cover"),
        replace(base, status="failed"),
        replace(base, engine="SymbiYosys 2.5.0"),
        replace(base, depth=64),
        replace(base, sby_file="formal/other.sby"),
    ]
    digests = {FormalProofCertificate([base], tool_version="sby-2.4").content_sha256()}
    digests.update(
        FormalProofCertificate([variant], tool_version="sby-2.4").content_sha256()
        for variant in variants
    )
    digests.add(FormalProofCertificate([base], tool_version="sby-2.5").content_sha256())
    assert len(digests) == len(variants) + 2


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


@pytest.mark.parametrize(
    "filename",
    ("../escape", "/absolute", "nested\\windows", "a/./b", "a//b"),
)
def test_evidence_item_rejects_non_normalised_paths(filename: str) -> None:
    """Manifest filenames must stay below their verification root."""
    with pytest.raises(ValueError, match="relative POSIX"):
        EvidenceItem(filename, "report", "invalid")


@pytest.mark.parametrize(
    "generated_at",
    ("", "not-a-date", "2026-07-12T18:30:00"),
)
def test_generator_rejects_invalid_or_naive_timestamps(generated_at: str) -> None:
    """Reproducibility timestamps must be non-empty, valid, and offset-aware."""
    with pytest.raises(ValueError, match="generated_at"):
        CertificationGenerator().generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["neuron"],
            [_property()],
            generated_at=generated_at,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"implementation_evidence": {"other": ["rtl/other.sv"]}},
        {"implementation_evidence": "invalid"},
        {"implementation_evidence": {42: ["rtl/neuron.sv"]}},
        {"implementation_evidence": {"neuron": "rtl/neuron.sv"}},
        {"implementation_evidence": {"neuron": [""]}},
        {"implementation_evidence": {"neuron": ["rtl/n.sv", "rtl/n.sv"]}},
        {"failure_modes": "invalid"},
        {"failure_modes": ["invalid"]},
        {"checklist_evidence": {"unknown": "evidence.md"}},
        {"checklist_evidence": {"7.4.2": ""}},
    ),
)
def test_generator_rejects_malformed_explicit_evidence(kwargs: dict[str, object]) -> None:
    """Every explicit evidence input must satisfy its typed boundary contract."""
    with pytest.raises(ValueError):
        CertificationGenerator().generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["neuron"],
            [_property()],
            _NETWORK_CONFIG,
            **_unsafe(kwargs),
        )


def test_manual_template_is_deterministic_and_non_certifying() -> None:
    """Manual output must state its limits and use a non-normative crosswalk."""
    report = SafetyManualGenerator.generate(
        "Example Controller",
        SILLevel.SIL_2,
        ["neuron"],
        42.5,
        generated_on="2026-07-12",
    )
    assert "2026-07-12" in report
    assert "Draft evidence template only" in report
    assert "does not establish equivalence" in report
    with pytest.raises(ValueError, match="generated_on"):
        SafetyManualGenerator.generate(
            "Example Controller",
            SILLevel.SIL_2,
            ["neuron"],
            42.5,
            generated_on="12 July 2026",
        )


def test_writer_rejects_invalid_directory_type_and_nul_path(tmp_path: Path) -> None:
    """Directory input must be path-like and safe for local filesystem calls."""
    package = _package()
    with pytest.raises(ValueError, match="directory"):
        package.write(_unsafe(42))
    with pytest.raises(ValueError, match="NUL"):
        package.write(str(tmp_path / "bad") + "\x00")


@pytest.mark.parametrize(
    "network_config",
    (
        {
            "bitstream_length": 0,
            "num_inputs": 8,
            "num_neurons": 16,
            "clock_mhz": 100.0,
        },
        {
            "bitstream_length": 256,
            "num_inputs": 8,
            "num_neurons": 16,
            "clock_mhz": True,
        },
        {
            "bitstream_length": 256,
            "num_inputs": 8,
            "num_neurons": 16,
            "clock_mhz": float("nan"),
        },
    ),
)
def test_generator_rejects_invalid_complete_network_config(
    network_config: dict[str, object],
) -> None:
    """Complete timing mappings still require positive typed values."""
    with pytest.raises(ValueError, match="network_config"):
        CertificationGenerator().generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["neuron"],
            [_property()],
            _unsafe(network_config),
            generated_at=_GENERATED_AT,
        )


def test_package_rejects_invalid_id_checklist_container_and_evidence_state() -> None:
    """Manual package construction must enforce its integrity boundary."""
    common: dict[str, object] = {
        "standard": SafetyStandard.IEC_61508,
        "sil_level": SILLevel.SIL_2,
        "traceability_report": "trace",
        "fmeda_report": "fmeda",
        "formal_cert_report": "formal",
        "wcet_report": "wcet",
        "checklist": [],
        "generated": _GENERATED_AT,
    }
    with pytest.raises(ValueError, match="package_hash"):
        CertificationPackage(**_unsafe({**common, "package_hash": "G" * 32}))
    with pytest.raises(ValueError, match="checklist must be a list"):
        CertificationPackage(**_unsafe({**common, "checklist": "invalid"}))

    item = ChecklistItem("IEC 61508_7.4.2", "7.4.2", "description")
    item.status = _unsafe("partial")
    with pytest.raises(ValueError, match="require evidence"):
        CertificationPackage(**_unsafe({**common, "checklist": [item]}))


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


def test_manual_rejects_non_string_date() -> None:
    """Runtime validation must reject dynamically typed date inputs."""
    with pytest.raises(ValueError, match="generated_on"):
        SafetyManualGenerator.generate(
            "Example Controller",
            SILLevel.SIL_2,
            ["neuron"],
            42.5,
            generated_on=_unsafe(20260712),
        )
