# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio federation verifiable-honesty seals

"""Seal SC-NeuroCore federation evidence into verifiable honesty envelopes.

Wraps the platform :mod:`scpn_studio_platform.seal` so a SC-NeuroCore claim's grade
is *provable*, not merely rendered: the grade is recomputed from the signed unit by a
pure regrade function, so a stripped or forged badge is detectable by anyone holding
the public key. The two evidence surfaces carry the two verifiability modes:

- :func:`seal_sc_inference` is **recompute-verifiable** — the stochastic-computing
  forward pass re-runs (in the browser via WASM, or anywhere) and its
  ``content_digest`` must match; no attestation is carried.
- :func:`attest_fpga_deployment` is **attestation-verifiable** — an FPGA run cannot be
  re-executed client-side, so the claim carries a signed result-pack over the
  content-addressed Vivado/cosim/bitstream artifacts instead of a recompute. The pack
  is studio-self-attested (the studio's signature vouches the artifacts are from a real
  synthesis/co-simulation run — the honest floor until a hardware-rooted attestation
  exists), and the grade still recomputes from the signed cosim evidence.

Importing this module requires the optional ``federation`` extra (the platform SDK).
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from scpn_studio_platform.seal import (
    HonestyEnvelope,
    Keyring,
    Signer,
    Verdict,
    content_digest,
    seal,
    verify,
)

from .evidence import FpgaDeploymentResult, ScInferenceResult
from .verbs import FPGA_DEPLOYMENT_SCHEMA, SC_INFERENCE_SCHEMA, STUDIO_ID

#: The grading code recorded in every envelope; its own error rate is WS-3's concern.
GRADER: Mapping[str, str] = {"name": "sc-neurocore.federation", "version": "1"}

_REFERENCE_VALIDATED = "reference-validated"
_BOUNDED_MODEL = "bounded-model"
_VALIDATION_GAP = "validation-gap"


@dataclass(frozen=True)
class FpgaArtifact:
    """A content-addressed FPGA evidence artifact backing a hardware claim.

    Parameters
    ----------
    role
        What the artifact is — e.g. ``"synthesis-timing"``, ``"synthesis-utilisation"``,
        ``"cosim-transcript"`` (the bit-exactness proof), or ``"bitstream"``.
    digest
        SHA-256 of the artifact (``"sha256:<hex>"``); the identity that binds the claim
        to the real file.
    media_type
        The artifact's media type (e.g. ``"text/vivado-timing"``).

    Raises
    ------
    ValueError
        If any field is blank.
    """

    role: str
    digest: str
    media_type: str

    def __post_init__(self) -> None:
        """Reject blank fields — every artifact must be addressable."""
        if not self.role.strip() or not self.digest.strip() or not self.media_type.strip():
            raise ValueError("FpgaArtifact fields must be non-empty")

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-serialisable mapping of the artifact."""
        return {"role": self.role, "digest": self.digest, "media_type": self.media_type}


def _sorted_artifacts(artifacts: Iterable[FpgaArtifact]) -> list[dict[str, str]]:
    """Return the artifacts as deterministically-ordered dicts (stable digest)."""
    return [a.to_dict() for a in sorted(artifacts, key=lambda a: (a.role, a.digest))]


def _sc_inference_unit(result: ScInferenceResult) -> dict[str, Any]:
    """Build the signed unit for an sc-inference result (recompute-verifiable)."""
    bit_identical = result.max_abs_error == 0.0
    return {
        "schema": SC_INFERENCE_SCHEMA,
        "studio": STUDIO_ID,
        "evidence_kind": "measured",
        "active_backend": result.active_backend,
        "reference_backend": result.reference_backend,
        "max_abs_error": result.max_abs_error,
        "bitstream_length": result.bitstream_length,
        "input_digest": result.input_digest,
        "result_digest": result.result_digest,
        "claim_status": _REFERENCE_VALIDATED if bit_identical else _BOUNDED_MODEL,
    }


def regrade_sc_inference(unit: Mapping[str, Any]) -> str:
    """Recompute the sc-inference grade from the signed unit.

    ``reference-validated`` only when the accelerated backend is bit-identical to the
    NumPy floor (``max_abs_error == 0``); otherwise ``bounded-model``. Derived from the
    evidence, never read from a (forgeable) ``claim_status`` field.
    """
    return _REFERENCE_VALIDATED if unit.get("max_abs_error") == 0.0 else _BOUNDED_MODEL


def seal_sc_inference(result: ScInferenceResult, *, signer: Signer) -> HonestyEnvelope:
    """Seal an sc-inference result in **recompute** mode (no attestation).

    Parameters
    ----------
    result
        The path-free stochastic-computing inference result.
    signer
        The studio's :class:`~scpn_studio_platform.seal.keys.Signer`.

    Returns
    -------
    HonestyEnvelope
        A recompute-verifiable envelope: the WASM/NumPy kernel re-derives the result and
        its digest must match.
    """
    return seal(
        _sc_inference_unit(result),
        signer=signer,
        grader=GRADER,
        verifiability_mode="recompute",
        exactness_class="bit-exact",
    )


def _fpga_unit(result: FpgaDeploymentResult, artifacts: Iterable[FpgaArtifact]) -> dict[str, Any]:
    """Build the signed unit for an FPGA deployment (attestation-verifiable)."""
    return {
        "schema": FPGA_DEPLOYMENT_SCHEMA,
        "studio": STUDIO_ID,
        "substrate": "fpga",
        "evidence_kind": "hardware-validated",
        "device": result.device,
        "cosim_bit_exact": result.cosim_bit_exact,
        "lut_used": result.lut_used,
        "ff_used": result.ff_used,
        "worst_negative_slack_ns": result.worst_negative_slack_ns,
        "clock_mhz": result.clock_mhz,
        "result_digest": result.result_digest,
        "artifacts": _sorted_artifacts(artifacts),
        "claim_status": _REFERENCE_VALIDATED if result.cosim_bit_exact else _VALIDATION_GAP,
    }


def _has_cosim_transcript(unit: Mapping[str, Any]) -> bool:
    """Whether a ``cosim-transcript`` artifact is present to back a bit-exactness claim."""
    artifacts = unit.get("artifacts", [])
    return isinstance(artifacts, list) and any(
        isinstance(a, Mapping) and a.get("role") == "cosim-transcript" for a in artifacts
    )


def regrade_fpga(unit: Mapping[str, Any]) -> str:
    """Recompute the FPGA grade from the signed unit.

    ``reference-validated`` only when the co-simulation is bit-exact **and** a
    ``cosim-transcript`` artifact backs that claim; otherwise ``validation-gap``. A
    bit-exact flag with no transcript to prove it does not earn the validated grade.
    """
    if unit.get("cosim_bit_exact") is True and _has_cosim_transcript(unit):
        return _REFERENCE_VALIDATED
    return _VALIDATION_GAP


def attest_fpga_deployment(
    result: FpgaDeploymentResult,
    artifacts: Iterable[FpgaArtifact],
    *,
    signer: Signer,
) -> HonestyEnvelope:
    """Seal an FPGA deployment in **attestation** mode with a signed result-pack.

    The FPGA run cannot be recomputed client-side, so the claim is verified by signature
    plus artifact-digest binding, not recompute. The result-pack reference content-
    addresses the real Vivado/cosim/bitstream artifacts and is studio-self-attested (the
    ``provider_sig`` is the studio's detached signature over the pack digest).

    Parameters
    ----------
    result
        The path-free synthesis/deployment result.
    artifacts
        The content-addressed artifacts the claim rests on; at least one is required, and
        a ``cosim-transcript`` must be present for the claim to grade as validated.
    signer
        The studio's :class:`~scpn_studio_platform.seal.keys.Signer`.

    Returns
    -------
    HonestyEnvelope
        An attestation-verifiable envelope on the ``fpga`` substrate.

    Raises
    ------
    ValueError
        If no artifacts are supplied.
    """
    materialised = list(artifacts)
    if not materialised:
        raise ValueError("attestation requires at least one content-addressed artifact")
    unit = _fpga_unit(result, materialised)
    pack_digest = content_digest({"artifacts": unit["artifacts"]})
    attestation = {
        "provider": f"vivado@{result.device}",
        "result_pack_digest": pack_digest,
        "provider_sig": base64.b64encode(signer.sign(pack_digest.encode("utf-8"))).decode("ascii"),
    }
    return seal(
        unit,
        signer=signer,
        grader=GRADER,
        verifiability_mode="attestation",
        exactness_class="bit-exact",
        attestation=attestation,
    )


def verify_envelope(
    envelope: Mapping[str, Any] | None,
    rendered_grade: str | None,
    *,
    keyring: Keyring,
) -> Verdict:
    """Verify a SC-NeuroCore honesty envelope, dispatching the regrade by schema.

    Parameters
    ----------
    envelope
        The :meth:`HonestyEnvelope.to_dict` wire form on the page, or ``None`` when the
        page carries no seal.
    rendered_grade
        The grade the page displays for the claim, or ``None``.
    keyring
        The trust anchor mapping ``key_id`` → public verifier.

    Returns
    -------
    Verdict
        :data:`~scpn_studio_platform.seal.verdict.Verdict.VERIFIED` only when the
        signature is valid and the rendered grade equals the grade recomputed from the
        signed unit; otherwise ``STRIPPED`` / ``FORGED`` / ``UNGRADED``.
    """
    schema: object = None
    if isinstance(envelope, Mapping):
        unit = envelope.get("unit")
        if isinstance(unit, Mapping):
            schema = unit.get("schema")
    regrade = regrade_fpga if schema == FPGA_DEPLOYMENT_SCHEMA else regrade_sc_inference
    return verify(envelope, rendered_grade, keyring=keyring, regrade=regrade)
