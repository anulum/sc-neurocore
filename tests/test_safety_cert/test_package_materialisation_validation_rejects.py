# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (validation_rejects) from former test_package_materialisation.py

from __future__ import annotations

from tests.test_safety_cert.package_materialisation_support import *  # noqa: F403


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
