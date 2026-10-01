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
- :func:`attest_fpga_deployment` is **attestation-verifiable** — the producer signs
  a result pack binding its pre-silicon co-simulation and content-addressed
  artifacts. This is studio self-attestation of a software experiment, without
  hardware-rooted attestation or a claim that a physical FPGA executed it.

Freshness is signed and regraded. Numeric wire values use exact decimal strings
so a browser JSON roundtrip cannot invalidate an otherwise identical signature.

Importing this module requires the optional ``federation`` extra (the platform SDK).
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from scpn_studio_platform.evidence import Freshness
from scpn_studio_platform.seal import (
    HonestyEnvelope,
    Keyring,
    Signer,
    Verdict,
    content_digest,
    seal,
    verify,
)

from .evidence import (
    FpgaDeploymentResult,
    ScInferenceResult,
    _finite_number,
    _require_source_recheck,
    _valid_digest,
    _wire_count,
)
from .verbs import FPGA_DEPLOYMENT_SCHEMA, SC_INFERENCE_SCHEMA, STUDIO_ID

#: The grading code recorded in every envelope; its own error rate is WS-3's concern.
GRADER: Mapping[str, str] = {"name": "sc-neurocore.federation", "version": "2"}

_REFERENCE_VALIDATED = "reference-validated"
_BOUNDED_MODEL = "bounded-model"
_VALIDATION_GAP = "validation-gap"


@dataclass(frozen=True)
class FpgaArtifact:
    """A content-addressed artifact backing a pre-silicon co-simulation claim.

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
        If a field is blank or the digest is not a lowercase SHA-256 address.
    """

    role: str
    digest: str
    media_type: str

    def __post_init__(self) -> None:
        """Reject blank descriptions and malformed content addresses."""
        if not all(
            isinstance(value, str) and value.strip() for value in (self.role, self.media_type)
        ):
            raise ValueError("FpgaArtifact fields must be non-empty")
        if not _valid_digest(self.digest):
            raise ValueError("FpgaArtifact digest must be a lowercase SHA-256 content address")

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-serialisable mapping of the artifact."""
        return {"role": self.role, "digest": self.digest, "media_type": self.media_type}


def _sorted_artifacts(artifacts: Iterable[FpgaArtifact]) -> list[dict[str, str]]:
    """Return the artifacts as deterministically-ordered dicts (stable digest)."""
    return [a.to_dict() for a in sorted(artifacts, key=lambda a: (a.role, a.digest))]


def _sc_inference_unit(result: ScInferenceResult, freshness: Freshness | None) -> dict[str, Any]:
    """Build the signed unit for an sc-inference result (recompute-verifiable)."""
    bit_identical = result.max_abs_error == 0.0
    return {
        "schema": SC_INFERENCE_SCHEMA,
        "studio": STUDIO_ID,
        "evidence_kind": "measured",
        "freshness": freshness.value if freshness is not None else None,
        "active_backend": result.active_backend,
        "reference_backend": result.reference_backend,
        "max_abs_error": "0.0" if bit_identical else repr(float(result.max_abs_error)),
        "bitstream_length": result.bitstream_length,
        "input_digest": result.input_digest,
        "result_digest": result.result_digest,
        "claim_status": _REFERENCE_VALIDATED if bit_identical else _BOUNDED_MODEL,
    }


def regrade_sc_inference(unit: Mapping[str, Any]) -> str:
    """Recompute the sc-inference grade from the signed unit.

    ``reference-validated`` requires a complete measured unit, explicit source
    re-check and exact string zero error against NumPy. Missing, legacy or malformed
    evidence stays ``bounded-model``. The signed ``claim_status`` is not trusted.
    """
    length = unit.get("bitstream_length")
    valid = (
        unit.get("schema") == SC_INFERENCE_SCHEMA
        and unit.get("studio") == STUDIO_ID
        and unit.get("evidence_kind") == "measured"
        and unit.get("freshness") == Freshness.VERIFIED_AT_SOURCE.value
        and unit.get("active_backend") in ("numpy", "rust")
        and unit.get("reference_backend") == "numpy"
        and _wire_count(length)
        and length > 0
        and _valid_digest(unit.get("input_digest"))
        and _valid_digest(unit.get("result_digest"))
        and isinstance(unit.get("max_abs_error"), str)
        and unit.get("max_abs_error") in ("0", "0.0")
    )
    return _REFERENCE_VALIDATED if valid else _BOUNDED_MODEL


def seal_sc_inference(
    result: ScInferenceResult, *, signer: Signer, freshness: Freshness | None = None
) -> HonestyEnvelope:
    """Seal an sc-inference result in **recompute** mode (no attestation).

    Parameters
    ----------
    result
        The path-free stochastic-computing inference result.
    signer
        The studio's :class:`~scpn_studio_platform.seal.keys.Signer`.
    freshness
        Actual source re-check status. Bit-identical results require
        ``VERIFIED_AT_SOURCE``; the value is part of the signed unit.

    Returns
    -------
    HonestyEnvelope
        A recompute-verifiable envelope: the WASM/NumPy kernel re-derives the result and
        its digest must match.

    Raises
    ------
    ValueError
        If bit-identical evidence has not been re-checked at source.
    """
    _require_source_recheck(result.max_abs_error == 0.0, freshness)
    return seal(
        _sc_inference_unit(result, freshness),
        signer=signer,
        grader=GRADER,
        verifiability_mode="recompute",
        exactness_class="bit-exact",
    )


def _fpga_unit(
    result: FpgaDeploymentResult, artifacts: Iterable[FpgaArtifact], freshness: Freshness | None
) -> dict[str, Any]:
    """Build the signed unit for an FPGA deployment (attestation-verifiable)."""
    unit: dict[str, Any] = {
        "schema": FPGA_DEPLOYMENT_SCHEMA,
        "studio": STUDIO_ID,
        "substrate": "simulator",
        "evidence_kind": "measured",
        "freshness": freshness.value if freshness is not None else None,
        "device": result.device,
        "cosim_bit_exact": result.cosim_bit_exact,
        "lut_used": result.lut_used,
        "ff_used": result.ff_used,
        "worst_negative_slack_ns": repr(float(result.worst_negative_slack_ns)),
        "clock_mhz": repr(float(result.clock_mhz)),
        "result_digest": result.result_digest,
        "artifacts": _sorted_artifacts(artifacts),
    }
    unit["claim_status"] = regrade_fpga(unit)
    return unit


def _has_cosim_transcript(unit: Mapping[str, Any]) -> bool:
    """Whether a ``cosim-transcript`` artifact is present to back a bit-exactness claim."""
    artifacts = unit.get("artifacts", [])
    return (
        isinstance(artifacts, list)
        and all(
            isinstance(a, Mapping)
            and isinstance(a.get("role"), str)
            and bool(a["role"].strip())
            and _valid_digest(a.get("digest"))
            and isinstance(a.get("media_type"), str)
            and bool(a["media_type"].strip())
            for a in artifacts
        )
        and any(a.get("role") == "cosim-transcript" for a in artifacts)
    )


def _finite_wire_number(value: object, *, positive: bool = False) -> bool:
    """Recognise a bounded decimal-string scalar that survives browser transport."""
    if not isinstance(value, str) or len(value) > 64:
        return False
    try:
        number = float(value)
    except ValueError:
        return False
    return _finite_number(number) and (not positive or number > 0.0)


def regrade_fpga(unit: Mapping[str, Any]) -> str:
    """Recompute the FPGA grade from the signed unit.

    ``reference-validated`` requires source-rechecked, measured simulator evidence,
    valid report fields and an addressed co-simulation transcript. This grade is
    pre-silicon parity; no physical deployment or power claim is established.
    """
    valid = (
        unit.get("schema") == FPGA_DEPLOYMENT_SCHEMA
        and unit.get("studio") == STUDIO_ID
        and unit.get("substrate") == "simulator"
        and unit.get("evidence_kind") == "measured"
        and unit.get("freshness") == Freshness.VERIFIED_AT_SOURCE.value
        and isinstance(unit.get("device"), str)
        and bool(unit["device"].strip())
        and unit.get("cosim_bit_exact") is True
        and all(_wire_count(unit.get(key)) for key in ("lut_used", "ff_used"))
        and _finite_wire_number(unit.get("worst_negative_slack_ns"))
        and _finite_wire_number(unit.get("clock_mhz"), positive=True)
        and _valid_digest(unit.get("result_digest"))
        and _has_cosim_transcript(unit)
    )
    if valid:
        return _REFERENCE_VALIDATED
    return _VALIDATION_GAP


def attest_fpga_deployment(
    result: FpgaDeploymentResult,
    artifacts: Iterable[FpgaArtifact],
    *,
    signer: Signer,
    freshness: Freshness | None = None,
) -> HonestyEnvelope:
    """Seal pre-silicon co-simulation in **attestation** mode with a result pack.

    The producer signs the complete path-free unit, including source freshness and
    artifact addresses. Its detached signature vouches for the producer's report;
    physical device execution and hardware-rooted attestation remain separate.

    Parameters
    ----------
    result
        The path-free synthesis/deployment result.
    artifacts
        The content-addressed artifacts the claim rests on; at least one is required, and
        a ``cosim-transcript`` must be present for the claim to grade as validated.
    signer
        The studio's :class:`~scpn_studio_platform.seal.keys.Signer`.
    freshness
        Actual source re-check status; bit-exact reports require
        ``VERIFIED_AT_SOURCE`` before signing.

    Returns
    -------
    HonestyEnvelope
        A studio-self-attested measurement on the ``simulator`` substrate.

    Raises
    ------
    ValueError
        If artifacts are absent or bit-exact evidence lacks a source re-check.
    """
    materialised = list(artifacts)
    if not materialised:
        raise ValueError("attestation requires at least one content-addressed artifact")
    _require_source_recheck(result.cosim_bit_exact, freshness)
    unit = _fpga_unit(result, materialised, freshness)
    pack_digest = content_digest(unit)
    attestation = {
        "provider": f"{STUDIO_ID}@{result.device}",
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
