# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — WaveformCodec polyglot validation contract tests

"""Contract checks for WaveformCodec validation ranges across language mirrors."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_MARKERS = (
    "WAVEFORM_CODEC_MIN_SNIPPET_SAMPLES",
    "WAVEFORM_CODEC_MAX_SNIPPET_SAMPLES",
    "WAVEFORM_CODEC_MIN_TEMPLATES",
    "WAVEFORM_CODEC_MAX_HEADER_COUNT",
    "WAVEFORM_CODEC_MAX_TEMPLATES",
    "WAVEFORM_CODEC_MIN_QUANTIZE_BITS",
    "WAVEFORM_CODEC_MAX_QUANTIZE_BITS",
    "WAVEFORM_CODEC_VALID_MODES",
)

MIRROR_PATHS = (
    "src/sc_neurocore/spike_codec/waveform_codec.py",
    "src/sc_neurocore/accel/rust/safety/waveform_codec.rs",
    "src/sc_neurocore/accel/go/services/waveform_codec/waveform_codec.go",
    "src/sc_neurocore/accel/julia/spike_codec/waveform_codec.jl",
    "src/sc_neurocore/accel/mojo/kernels/waveform_codec.mojo",
)


def test_waveform_codec_validation_markers_exist_in_all_language_surfaces() -> None:
    """Every language surface carries the shared wire-header validation contract."""
    missing: list[str] = []
    for relative_path in MIRROR_PATHS:
        body = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for marker in CONTRACT_MARKERS:
            if marker not in body:
                missing.append(f"{relative_path}: {marker}")

    assert missing == []


def test_authoritative_waveform_codec_mirrors_have_runtime_validators() -> None:
    """Compiled mirrors expose fail-closed validators instead of only comments."""
    rust_body = (REPO_ROOT / "src/sc_neurocore/accel/rust/safety/waveform_codec.rs").read_text(
        encoding="utf-8"
    )
    go_body = (
        REPO_ROOT / "src/sc_neurocore/accel/go/services/waveform_codec/waveform_codec.go"
    ).read_text(encoding="utf-8")

    assert "pub fn validate_waveform_codec" in rust_body
    assert "finite_integer_in_range" in rust_body
    assert "func ValidateConfig" in go_body
    assert "math.IsNaN" in go_body
