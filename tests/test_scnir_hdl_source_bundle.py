# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR HDL source-bundle emission

"""Contracts for ``build_scnir_source_bundle`` source materialisation."""

from __future__ import annotations

import pytest

from sc_neurocore.ir import build_scnir_source_bundle
from sc_neurocore.ir.scnir_schema import (
    SCNIRDocument,
    SCNIRPrecision,
    SCNIRSource,
    SCNIRStream,
)


def _stream(stream_id: str, source: SCNIRSource) -> SCNIRStream:
    return SCNIRStream(
        stream_id=stream_id,
        layer="layer0",
        bitstream_length=1024,
        encoding="bipolar",
        signal_kind="spike",
        precision=SCNIRPrecision(
            signed=True,
            total_bits=16,
            fractional_bits=8,
            accumulator_bits=32,
            rounding="nearest_even",
            overflow="saturate",
        ),
        source=source,
    )


def _document(*sources: SCNIRSource) -> SCNIRDocument:
    return SCNIRDocument(
        producer="sc-neurocore-test",
        streams=[_stream(f"stream_{index}", source) for index, source in enumerate(sources)],
    )


def test_source_bundle_emits_lfsr_and_sobol_with_manifest_dicts() -> None:
    """Both LFSR and Sobol streams materialise, and manifest_dicts is JSON-ready."""
    document = _document(
        SCNIRSource(
            kind="lfsr",
            seed=17,
            lfsr_polynomial="x^16 + x^14 + x^13 + x^11 + 1",
            tap_mask=0xB400,
        ),
        SCNIRSource(kind="sobol", seed=5, sobol_dimension=3),
    )

    bundle = build_scnir_source_bundle(document)

    assert len(bundle.modules) == 2
    assert len(bundle.manifest) == 2
    dicts = bundle.manifest_dicts()
    assert [entry["source_kind"] for entry in dicts] == ["lfsr16", "sobol16"]
    assert all(isinstance(entry, dict) for entry in dicts)


def test_source_bundle_fails_closed_without_a_seed() -> None:
    """A source without a seed cannot be lowered to a deterministic RTL module."""
    document = _document(SCNIRSource(kind="sobol", sobol_dimension=3))

    with pytest.raises(ValueError, match="needs seed"):
        build_scnir_source_bundle(document)


def test_source_bundle_rejects_unmaterialisable_source_kind() -> None:
    """A schema-valid but non-threshold source kind fails closed instead of mislowering."""
    document = _document(SCNIRSource(kind="halton", seed=5, halton_base=2))

    with pytest.raises(ValueError, match="cannot be materialised"):
        build_scnir_source_bundle(document)
