# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio federation seal/attestation tests

"""Tests for the verifiable-honesty seals over SC-NeuroCore federation evidence.

Covers both verifiability modes (recompute for sc-inference, attestation for
fpga-deployment), the grade-recompute invariant (a forged rendered grade is caught),
the transcript-backs-bit-exact rule, and an adversarial strip-resistance battery
(stripped / tampered / forged-key envelopes must never verify).
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import replace
from typing import Any

import numpy as np

import pytest

pytest.importorskip("scpn_studio_platform")

from scpn_studio_platform.evidence import Freshness
from scpn_studio_platform.seal import Ed25519Signer, Keyring, Verdict, content_digest  # noqa: E402

from sc_neurocore.accel import pack_bitstream, sc_forward

from sc_neurocore.federation import (  # noqa: E402
    FpgaArtifact,
    FpgaDeploymentResult,
    ScInferenceResult,
    attest_fpga_deployment,
    regrade_fpga,
    regrade_sc_inference,
    seal_sc_inference,
    verify_envelope,
)


def _signer_and_keyring() -> tuple[Ed25519Signer, Keyring]:
    signer = Ed25519Signer.generate("sc-neurocore:k1")
    keyring = Keyring()
    keyring.add(signer.key_id, signer.verifier())
    return signer, keyring


def _bit_true_inference() -> ScInferenceResult:
    """Recompute the public Rust and NumPy paths over identical retained inputs."""
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
        active_backend="rust",
        reference_backend="numpy",
        max_abs_error=float(np.max(np.abs(measured - reference))),
        bitstream_length=length,
        input_digest="sha256:" + hashlib.sha256(source).hexdigest(),
        result_digest="sha256:" + hashlib.sha256(measured.astype("<f8").tobytes()).hexdigest(),
    )


def _cosim_contract_result(*, cosim_bit_exact: bool = True) -> FpgaDeploymentResult:
    """Supply explicit protocol fixtures; these values are not a physical run."""
    return FpgaDeploymentResult(
        device="xc7z020-1clg400",
        cosim_bit_exact=cosim_bit_exact,
        lut_used=1317,
        ff_used=900,
        worst_negative_slack_ns=4.048,
        clock_mhz=100.0,
        result_digest="sha256:" + "c" * 64,
    )


def _artifacts(*, with_transcript: bool = True) -> list[FpgaArtifact]:
    """Declare well-formed contract addresses without claiming measured artifacts."""
    arts = [
        FpgaArtifact("synthesis-timing", "sha256:" + "1" * 64, "text/vivado-timing"),
        FpgaArtifact("synthesis-utilisation", "sha256:" + "2" * 64, "text/vivado-util"),
        FpgaArtifact("bitstream", "sha256:" + "3" * 64, "application/octet-stream"),
    ]
    if with_transcript:
        arts.append(FpgaArtifact("cosim-transcript", "sha256:" + "4" * 64, "text/cosim-log"))
    return arts


# --- recompute mode (sc-inference) -----------------------------------------------------------


@pytest.mark.parametrize("error", [0.0, -0.0])
def test_sc_inference_recompute_envelope_verifies(error: float) -> None:
    signer, keyring = _signer_and_keyring()
    env = seal_sc_inference(
        replace(_bit_true_inference(), max_abs_error=error),
        signer=signer,
        freshness=Freshness.VERIFIED_AT_SOURCE,
    ).to_dict()
    assert env["verifiability_mode"] == "recompute"
    assert env["attestation"] is None
    assert verify_envelope(env, "reference-validated", keyring=keyring) is Verdict.VERIFIED


def test_sc_inference_forged_grade_is_caught() -> None:
    signer, keyring = _signer_and_keyring()
    env = seal_sc_inference(
        _bit_true_inference(), signer=signer, freshness=Freshness.VERIFIED_AT_SOURCE
    ).to_dict()
    # The page claims a grade the signed unit does not earn.
    assert verify_envelope(env, "formally-proven", keyring=keyring) is Verdict.FORGED


def test_sc_inference_non_bit_identical_grades_bounded() -> None:
    signer, keyring = _signer_and_keyring()
    drifting = ScInferenceResult(
        active_backend="rust",
        reference_backend="numpy",
        max_abs_error=1e-3,
        bitstream_length=1024,
        input_digest="sha256:" + "a" * 64,
        result_digest="sha256:" + "b" * 64,
    )
    env = seal_sc_inference(
        drifting, signer=signer, freshness=Freshness.VERIFIED_AT_SOURCE
    ).to_dict()
    assert verify_envelope(env, "bounded-model", keyring=keyring) is Verdict.VERIFIED
    assert verify_envelope(env, "reference-validated", keyring=keyring) is Verdict.FORGED


# --- attestation mode (fpga-deployment) ------------------------------------------------------


def test_fpga_attestation_envelope_verifies_and_carries_result_pack() -> None:
    signer, keyring = _signer_and_keyring()
    env = attest_fpga_deployment(
        _cosim_contract_result(),
        _artifacts(),
        signer=signer,
        freshness=Freshness.VERIFIED_AT_SOURCE,
    ).to_dict()
    assert env["verifiability_mode"] == "attestation"
    attestation = env["attestation"]
    assert attestation["provider"] == "sc-neurocore@xc7z020-1clg400"
    assert attestation["result_pack_digest"].startswith("sha256:")
    assert attestation["provider_sig"]
    assert verify_envelope(env, "reference-validated", keyring=keyring) is Verdict.VERIFIED


def test_fpga_cosim_mismatch_grades_validation_gap() -> None:
    signer, keyring = _signer_and_keyring()
    env = attest_fpga_deployment(
        _cosim_contract_result(cosim_bit_exact=False),
        _artifacts(),
        signer=signer,
        freshness=Freshness.VERIFIED_AT_SOURCE,
    ).to_dict()
    assert verify_envelope(env, "validation-gap", keyring=keyring) is Verdict.VERIFIED
    assert verify_envelope(env, "reference-validated", keyring=keyring) is Verdict.FORGED


def test_fpga_bit_exact_without_transcript_does_not_earn_validated() -> None:
    # The honesty rule: a bit-exact flag needs a cosim-transcript artefact to back it.
    signer, keyring = _signer_and_keyring()
    env = attest_fpga_deployment(
        _cosim_contract_result(cosim_bit_exact=True),
        _artifacts(with_transcript=False),
        signer=signer,
        freshness=Freshness.VERIFIED_AT_SOURCE,
    ).to_dict()
    assert regrade_fpga(env["unit"]) == "validation-gap"
    assert verify_envelope(env, "reference-validated", keyring=keyring) is Verdict.FORGED


def test_fpga_attestation_requires_artifacts() -> None:
    signer, _ = _signer_and_keyring()
    with pytest.raises(ValueError, match="at least one"):
        attest_fpga_deployment(_cosim_contract_result(), [], signer=signer)


@pytest.mark.parametrize("blank", ["", "   "])
def test_fpga_artifact_rejects_blank_fields(blank: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FpgaArtifact(blank, "sha256:" + "1" * 64, "text/plain")


# --- adversarial strip-resistance ------------------------------------------------------------


def test_absent_envelope_with_rendered_grade_is_stripped() -> None:
    _, keyring = _signer_and_keyring()
    assert verify_envelope(None, "reference-validated", keyring=keyring) is Verdict.STRIPPED


def test_absent_envelope_without_grade_is_ungraded() -> None:
    _, keyring = _signer_and_keyring()
    assert verify_envelope(None, None, keyring=keyring) is Verdict.UNGRADED


def test_tampered_unit_is_forged() -> None:
    signer, keyring = _signer_and_keyring()
    env: dict[str, Any] = seal_sc_inference(
        _bit_true_inference(), signer=signer, freshness=Freshness.VERIFIED_AT_SOURCE
    ).to_dict()
    tampered = copy.deepcopy(env)
    tampered["unit"]["max_abs_error"] = 0.0  # already 0; mutate something material instead
    tampered["unit"]["result_digest"] = "sha256:" + "f" * 64
    assert verify_envelope(tampered, "reference-validated", keyring=keyring) is Verdict.FORGED


def test_foreign_key_is_forged() -> None:
    signer, _ = _signer_and_keyring()
    env = attest_fpga_deployment(
        _cosim_contract_result(),
        _artifacts(),
        signer=signer,
        freshness=Freshness.VERIFIED_AT_SOURCE,
    ).to_dict()
    # A keyring that does not trust the signer's key.
    other = Ed25519Signer.generate("attacker:k1")
    foreign = Keyring()
    foreign.add(other.key_id, other.verifier())
    assert verify_envelope(env, "reference-validated", keyring=foreign) is Verdict.FORGED


def test_verify_dispatches_regrade_by_schema() -> None:
    # An fpga envelope verified through the public entry point uses the fpga regrade
    # (validation-gap on cosim fail), proving schema dispatch, not the sc-inference grade.
    signer, keyring = _signer_and_keyring()
    env = attest_fpga_deployment(
        _cosim_contract_result(cosim_bit_exact=False),
        _artifacts(),
        signer=signer,
        freshness=Freshness.VERIFIED_AT_SOURCE,
    ).to_dict()
    assert verify_envelope(env, "validation-gap", keyring=keyring) is Verdict.VERIFIED


@pytest.mark.parametrize("freshness", [None, Freshness.TRACEABLE_UNCHECKED, Freshness.UNTRACEABLE])
def test_unchecked_exact_results_cannot_be_sealed(freshness: Freshness | None) -> None:
    """Keep the public signed path subject to the same re-check gate as bundles."""
    signer, _ = _signer_and_keyring()
    with pytest.raises(ValueError, match="requires evidence re-checked at source"):
        seal_sc_inference(_bit_true_inference(), signer=signer, freshness=freshness)
    with pytest.raises(ValueError, match="requires evidence re-checked at source"):
        attest_fpga_deployment(
            _cosim_contract_result(), _artifacts(), signer=signer, freshness=freshness
        )


@pytest.mark.parametrize("digest", ["", "invalid-digest", "sha256:" + "A" * 64, "sha256:abcd"])
def test_unaddressed_artifacts_cannot_be_attested(digest: str) -> None:
    """Refuse malformed addresses before they can acquire a producer signature."""
    with pytest.raises(ValueError, match="SHA-256"):
        FpgaArtifact("cosim-transcript", digest, "text/plain")


@pytest.mark.parametrize("count", [1317, 2**53 - 1])
def test_signed_units_survive_real_browser_json_transport(count: int) -> None:
    """Preserve both real signatures across Node JSON.parse and JSON.stringify."""
    signer, keyring = _signer_and_keyring()
    inference = seal_sc_inference(
        _bit_true_inference(), signer=signer, freshness=Freshness.VERIFIED_AT_SOURCE
    ).to_dict()
    cosim = attest_fpga_deployment(
        replace(_cosim_contract_result(), lut_used=count, ff_used=count),
        _artifacts(),
        signer=signer,
        freshness=Freshness.VERIFIED_AT_SOURCE,
    ).to_dict()
    command = [
        "node",
        "-e",
        'let s="";process.stdin.on("data",c=>s+=c);'
        'process.stdin.on("end",()=>process.stdout.write(JSON.stringify(JSON.parse(s))));',
    ]
    transported = subprocess.run(
        command,
        input=json.dumps([inference, cosim]),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    restored = json.loads(transported.stdout)
    for original, envelope in zip([inference, cosim], restored, strict=True):
        assert envelope == original
        assert envelope["unit"]["freshness"] == "verified-at-source"
        assert verify_envelope(envelope, "reference-validated", keyring=keyring) is Verdict.VERIFIED
    assert isinstance(inference["unit"]["max_abs_error"], str)
    assert isinstance(cosim["unit"]["clock_mhz"], str)
    assert isinstance(cosim["unit"]["worst_negative_slack_ns"], str)
    assert cosim["unit"]["substrate"] == "simulator"
    assert cosim["unit"]["evidence_kind"] == "measured"
    assert cosim["attestation"]["result_pack_digest"] == content_digest(cosim["unit"])


@pytest.mark.parametrize("error", [False, True, 0, 0.0, None, "nan", "Infinity", "0.1"])
def test_signed_inference_regrader_refuses_malformed_or_nonzero_error(error: object) -> None:
    """A zero-looking JSON scalar cannot promote an unsupported signed unit."""
    signer, _ = _signer_and_keyring()
    unit = seal_sc_inference(
        _bit_true_inference(), signer=signer, freshness=Freshness.VERIFIED_AT_SOURCE
    ).to_dict()["unit"]
    unit["max_abs_error"] = error
    assert regrade_sc_inference(unit) == "bounded-model"


@pytest.mark.parametrize("freshness", [None, "traceable-unchecked", "untraceable", "future-state"])
def test_regraders_do_not_promote_legacy_or_unchecked_units(freshness: object) -> None:
    """Refuse source-unchecked grades even when a signed claim_status says validated."""
    signer, _ = _signer_and_keyring()
    inference = seal_sc_inference(
        _bit_true_inference(), signer=signer, freshness=Freshness.VERIFIED_AT_SOURCE
    ).to_dict()["unit"]
    fpga = attest_fpga_deployment(
        _cosim_contract_result(),
        _artifacts(),
        signer=signer,
        freshness=Freshness.VERIFIED_AT_SOURCE,
    ).to_dict()["unit"]
    for unit in [inference, fpga]:
        unit["freshness"] = freshness
        unit["claim_status"] = "reference-validated"
    assert regrade_sc_inference(inference) == "bounded-model"
    assert regrade_fpga(fpga) == "validation-gap"
    for unit in [inference, fpga]:
        del unit["freshness"]
    assert regrade_sc_inference(inference) == "bounded-model"
    assert regrade_fpga(fpga) == "validation-gap"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("substrate", "fpga"),
        ("evidence_kind", "hardware-validated"),
        ("clock_mhz", 100.0),
        ("clock_mhz", "0.0"),
        ("clock_mhz", "nan"),
        ("clock_mhz", "not-a-number"),
        ("clock_mhz", "1" * 65),
        ("worst_negative_slack_ns", None),
        ("lut_used", False),
        ("lut_used", 2**53),
        ("ff_used", 2**53 + 1),
        ("result_digest", "invalid"),
        ("artifacts", []),
        (
            "artifacts",
            [{"role": "cosim-transcript", "digest": "invalid", "media_type": "text/plain"}],
        ),
        ("artifacts", [{"role": "cosim-transcript", "digest": "sha256:" + "a" * 64}]),
    ],
)
def test_incomplete_or_physical_claims_do_not_regrade_as_cosim(field: str, value: object) -> None:
    """Require addressed pre-silicon evidence without treating board labels as proof."""
    signer, _ = _signer_and_keyring()
    unit = attest_fpga_deployment(
        _cosim_contract_result(),
        _artifacts(),
        signer=signer,
        freshness=Freshness.VERIFIED_AT_SOURCE,
    ).to_dict()["unit"]
    unit[field] = value
    assert regrade_fpga(unit) == "validation-gap"


@pytest.mark.parametrize("freshness", [None, Freshness.TRACEABLE_UNCHECKED, Freshness.UNTRACEABLE])
def test_unchecked_mismatches_are_signed_only_as_boundaries(freshness: Freshness | None) -> None:
    """Retain mismatch evidence without silently adding source freshness."""
    signer, keyring = _signer_and_keyring()
    inference = seal_sc_inference(
        replace(_bit_true_inference(), max_abs_error=0.125),
        signer=signer,
        freshness=freshness,
    ).to_dict()
    fpga = attest_fpga_deployment(
        _cosim_contract_result(cosim_bit_exact=False),
        _artifacts(),
        signer=signer,
        freshness=freshness,
    ).to_dict()
    assert verify_envelope(inference, "bounded-model", keyring=keyring) is Verdict.VERIFIED
    assert verify_envelope(fpga, "validation-gap", keyring=keyring) is Verdict.VERIFIED
    expected = freshness.value if freshness is not None else None
    assert inference["unit"]["freshness"] == fpga["unit"]["freshness"] == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_abs_error", float("nan")),
        ("max_abs_error", float("inf")),
        ("max_abs_error", False),
        ("max_abs_error", "0"),
        ("max_abs_error", 10**1000),
        ("bitstream_length", True),
        ("bitstream_length", 2**53),
        ("active_backend", "unavailable"),
        ("reference_backend", "unverified"),
        ("input_digest", "invalid"),
        ("result_digest", "sha256:" + "A" * 64),
    ],
)
def test_malformed_inference_results_cannot_reach_a_seal(field: str, value: Any) -> None:
    """Reject raw untrusted fixture values, preserving their deliberately wrong types."""
    with pytest.raises(ValueError):
        replace(_bit_true_inference(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", ""),
        ("cosim_bit_exact", 1),
        ("lut_used", -1),
        ("ff_used", True),
        ("lut_used", 2**53),
        ("ff_used", 2**53 + 1),
        ("clock_mhz", 0.0),
        ("clock_mhz", float("nan")),
        ("worst_negative_slack_ns", float("inf")),
        ("result_digest", "invalid"),
    ],
)
def test_malformed_cosim_reports_cannot_reach_attestation(field: str, value: Any) -> None:
    """Keep raw untrusted fixture types out of a producer-signed result pack."""
    with pytest.raises(ValueError):
        replace(_cosim_contract_result(), **{field: value})


def test_artifact_requires_a_media_description() -> None:
    """Addressing alone does not supply the artifact's declared media type."""
    with pytest.raises(ValueError, match="non-empty"):
        FpgaArtifact("cosim-transcript", "sha256:" + "a" * 64, " ")


@pytest.mark.parametrize("freshness", ["verified-at-source", False, 1, {}])
def test_untyped_freshness_is_not_a_source_recheck(freshness: Any) -> None:
    """Require the SDK member when an untrusted caller supplies a lookalike value."""
    signer, _ = _signer_and_keyring()
    with pytest.raises(ValueError, match="Freshness member"):
        seal_sc_inference(_bit_true_inference(), signer=signer, freshness=freshness)


@pytest.mark.parametrize("envelope", [{}, {"unit": None}, {"unit": []}])
def test_malformed_envelope_never_verifies_a_validated_badge(envelope: dict[str, Any]) -> None:
    """Treat malformed wire documents as missing seals through the public verifier."""
    assert verify_envelope(envelope, "reference-validated", keyring=Keyring()) is Verdict.STRIPPED


@pytest.mark.parametrize("length", [False, 0, -1, 2**53, 2**53 + 1])
def test_regrader_refuses_unsafe_wire_lengths(length: int) -> None:
    """Refuse raw untrusted lengths that cannot describe a safe signed wire count."""
    signer, _ = _signer_and_keyring()
    unit = seal_sc_inference(
        _bit_true_inference(), signer=signer, freshness=Freshness.VERIFIED_AT_SOURCE
    ).to_dict()["unit"]
    unit["bitstream_length"] = length
    assert regrade_sc_inference(unit) == "bounded-model"
