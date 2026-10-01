# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Federation source freshness through measured inference

"""Exercise source freshness with actual Rust/NumPy inference and wire admission."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("scpn_studio_platform")

from scpn_studio_platform.evidence import Freshness, validate_studio_bundle
from scpn_studio_platform.manifest import validate_studio_manifest

from sc_neurocore.accel import pack_bitstream, sc_forward
from sc_neurocore.federation import ScInferenceResult, build_manifest, sc_inference_evidence


@pytest.fixture
def inference_result() -> ScInferenceResult:
    """Measure the public Rust entry point against NumPy on retained packed inputs."""
    length, seed = 257, 173
    bits = np.random.default_rng(31).integers(0, 2, size=(2, 3, length), dtype=np.uint8)
    weights = np.stack([pack_bitstream(row) for row in bits.reshape(6, length)]).reshape(2, 3, -1)
    inputs = np.array([0.125, 0.5, 0.875], dtype=np.float64)
    reference = sc_forward(weights, inputs, length=length, seed=seed, backend="numpy")
    measured = sc_forward(weights, inputs, length=length, seed=seed, backend="rust")
    np.testing.assert_array_equal(measured, reference)
    source = weights.astype("<u8").tobytes() + inputs.astype("<f8").tobytes()
    source += seed.to_bytes(4, "little") + length.to_bytes(4, "little")
    return ScInferenceResult(
        "rust",
        "numpy",
        float(np.max(np.abs(measured - reference))),
        length,
        "sha256:" + hashlib.sha256(source).hexdigest(),
        "sha256:" + hashlib.sha256(measured.astype("<f8").tobytes()).hexdigest(),
    )


def test_rechecked_inference_federates_with_explicit_freshness(
    inference_result: ScInferenceResult,
) -> None:
    """Admit a source-recomputed measurement through the actual platform wire gate."""
    bundle = sc_inference_evidence(
        inference_result,
        operator="test:source-recheck",
        studio_version="3.16.0",
        started="2026-09-30T20:21:00Z",
        ended="2026-09-30T20:21:01Z",
        freshness=Freshness.VERIFIED_AT_SOURCE,
    )
    wire = bundle.to_dict()
    assert wire["freshness"] == "verified-at-source"
    assert wire["prov"]["entity"]["digest"] == inference_result.result_digest
    verdict = validate_studio_bundle(wire)
    assert verdict.admitted and verdict.mode == "validated"


@pytest.mark.parametrize("freshness", [None, Freshness.TRACEABLE_UNCHECKED, Freshness.UNTRACEABLE])
def test_parity_does_not_promote_unchecked_source(
    inference_result: ScInferenceResult, freshness: Freshness | None
) -> None:
    """Refuse a parity claim when its producer cannot supply a source re-check."""
    with pytest.raises(ValueError, match="requires evidence re-checked at source"):
        sc_inference_evidence(
            inference_result,
            operator="test:unchecked",
            studio_version="3.16.0",
            started="2026-09-30T20:21:00Z",
            ended="2026-09-30T20:21:01Z",
            freshness=freshness,
        )


def test_unchecked_mismatch_remains_a_boundary(inference_result: ScInferenceResult) -> None:
    """Preserve an unvalidated claim without manufacturing source freshness."""
    bundle = sc_inference_evidence(
        replace(inference_result, max_abs_error=0.125),
        operator="test:mismatch-contract",
        studio_version="3.16.0",
        started="2026-09-30T20:21:00Z",
        ended="2026-09-30T20:21:01Z",
        freshness=Freshness.TRACEABLE_UNCHECKED,
    )
    verdict = validate_studio_bundle(bundle.to_dict())
    assert verdict.admitted and verdict.mode == "boundary"
    assert bundle.to_dict()["freshness"] == "traceable-unchecked"


def test_manifest_requires_a_consumer_with_the_same_era() -> None:
    """Check the real manifest handshake before admitting the vertical."""
    wire = build_manifest().to_dict()
    assert validate_studio_manifest(wire, supported_eras={"v2"}).admitted
    refusal = validate_studio_manifest(wire, supported_eras={"v1"})
    assert not refusal.admitted
    assert refusal.contract_era == "v2"
    assert any("no common era" in reason for reason in refusal.rejections)
