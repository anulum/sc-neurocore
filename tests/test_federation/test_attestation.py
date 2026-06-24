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
from typing import Any

import pytest

pytest.importorskip("scpn_studio_platform")

from scpn_studio_platform.seal import Ed25519Signer, Keyring, Verdict  # noqa: E402

from sc_neurocore.federation import (  # noqa: E402
    FpgaArtifact,
    FpgaDeploymentResult,
    ScInferenceResult,
    attest_fpga_deployment,
    regrade_fpga,
    seal_sc_inference,
    verify_envelope,
)


def _signer_and_keyring() -> tuple[Ed25519Signer, Keyring]:
    signer = Ed25519Signer.generate("sc-neurocore:k1")
    keyring = Keyring()
    keyring.add(signer.key_id, signer.verifier())
    return signer, keyring


def _bit_true_inference() -> ScInferenceResult:
    return ScInferenceResult(
        active_backend="rust",
        reference_backend="numpy",
        max_abs_error=0.0,
        bitstream_length=1024,
        input_digest="sha256:" + "a" * 64,
        result_digest="sha256:" + "b" * 64,
    )


def _silicon_result(*, cosim_bit_exact: bool = True) -> FpgaDeploymentResult:
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
    arts = [
        FpgaArtifact("synthesis-timing", "sha256:" + "1" * 64, "text/vivado-timing"),
        FpgaArtifact("synthesis-utilisation", "sha256:" + "2" * 64, "text/vivado-util"),
        FpgaArtifact("bitstream", "sha256:" + "3" * 64, "application/octet-stream"),
    ]
    if with_transcript:
        arts.append(FpgaArtifact("cosim-transcript", "sha256:" + "4" * 64, "text/cosim-log"))
    return arts


# --- recompute mode (sc-inference) -----------------------------------------------------------


def test_sc_inference_recompute_envelope_verifies() -> None:
    signer, keyring = _signer_and_keyring()
    env = seal_sc_inference(_bit_true_inference(), signer=signer).to_dict()
    assert env["verifiability_mode"] == "recompute"
    assert env["attestation"] is None
    assert verify_envelope(env, "reference-validated", keyring=keyring) is Verdict.VERIFIED


def test_sc_inference_forged_grade_is_caught() -> None:
    signer, keyring = _signer_and_keyring()
    env = seal_sc_inference(_bit_true_inference(), signer=signer).to_dict()
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
    env = seal_sc_inference(drifting, signer=signer).to_dict()
    assert verify_envelope(env, "bounded-model", keyring=keyring) is Verdict.VERIFIED
    assert verify_envelope(env, "reference-validated", keyring=keyring) is Verdict.FORGED


# --- attestation mode (fpga-deployment) ------------------------------------------------------


def test_fpga_attestation_envelope_verifies_and_carries_result_pack() -> None:
    signer, keyring = _signer_and_keyring()
    env = attest_fpga_deployment(_silicon_result(), _artifacts(), signer=signer).to_dict()
    assert env["verifiability_mode"] == "attestation"
    attestation = env["attestation"]
    assert attestation["provider"] == "vivado@xc7z020-1clg400"
    assert attestation["result_pack_digest"].startswith("sha256:")
    assert attestation["provider_sig"]
    assert verify_envelope(env, "reference-validated", keyring=keyring) is Verdict.VERIFIED


def test_fpga_cosim_mismatch_grades_validation_gap() -> None:
    signer, keyring = _signer_and_keyring()
    env = attest_fpga_deployment(
        _silicon_result(cosim_bit_exact=False), _artifacts(), signer=signer
    ).to_dict()
    assert verify_envelope(env, "validation-gap", keyring=keyring) is Verdict.VERIFIED
    assert verify_envelope(env, "reference-validated", keyring=keyring) is Verdict.FORGED


def test_fpga_bit_exact_without_transcript_does_not_earn_validated() -> None:
    # The honesty rule: a bit-exact flag needs a cosim-transcript artefact to back it.
    signer, keyring = _signer_and_keyring()
    env = attest_fpga_deployment(
        _silicon_result(cosim_bit_exact=True), _artifacts(with_transcript=False), signer=signer
    ).to_dict()
    assert regrade_fpga(env["unit"]) == "validation-gap"
    assert verify_envelope(env, "reference-validated", keyring=keyring) is Verdict.FORGED


def test_fpga_attestation_requires_artifacts() -> None:
    signer, _ = _signer_and_keyring()
    with pytest.raises(ValueError, match="at least one"):
        attest_fpga_deployment(_silicon_result(), [], signer=signer)


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
    env: dict[str, Any] = seal_sc_inference(_bit_true_inference(), signer=signer).to_dict()
    tampered = copy.deepcopy(env)
    tampered["unit"]["max_abs_error"] = 0.0  # already 0; mutate something material instead
    tampered["unit"]["result_digest"] = "sha256:" + "f" * 64
    assert verify_envelope(tampered, "reference-validated", keyring=keyring) is Verdict.FORGED


def test_foreign_key_is_forged() -> None:
    signer, _ = _signer_and_keyring()
    env = attest_fpga_deployment(_silicon_result(), _artifacts(), signer=signer).to_dict()
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
        _silicon_result(cosim_bit_exact=False), _artifacts(), signer=signer
    ).to_dict()
    assert verify_envelope(env, "validation-gap", keyring=keyring) is Verdict.VERIFIED
