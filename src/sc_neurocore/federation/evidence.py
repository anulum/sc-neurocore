# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio federation evidence bundles

"""Emit ``studio.*.v1`` evidence bundles for SC-NeuroCore results.

Maps SC-NeuroCore's provenance-graded result surfaces onto the locked platform
:class:`scpn_studio_platform.evidence.EvidenceBundle`. Two mappers exercise the
contract's honesty invariant from opposite ends of the evidence ladder:

- :func:`sc_inference_evidence` is a ``measured`` software result. Its claim is
  ``reference-validated`` only when the accelerated backend is bit-identical to the
  NumPy floor for a fixed seed; otherwise the claim degrades to ``bounded-model``
  and is not admitted.
- :func:`fpga_deployment_evidence` records measured pre-silicon co-simulation
  against the Q8.8 fixed-point reference, on the ``simulator`` substrate. Its
  claim is ``reference-validated`` only when the co-simulation is bit-exact and
  re-checked at source; a mismatch degrades to ``validation-gap``. Synthesis and
  co-simulation do not establish that a bitstream ran on a physical board.

Neither outcome is an SC-NeuroCore-private rule; each falls out of the shared
:class:`scpn_studio_platform.evidence.ClaimBoundary` lattice and the orthogonal
:class:`scpn_studio_platform.evidence.EvidenceKind` and ``Substrate`` axes.
Timestamps and digests are passed in by the caller (no hidden clock), so a bundle
is reproducible.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TypeGuard

from scpn_studio_platform.evidence import (
    AdmissionDecision,
    CaseResult,
    ClaimBoundary,
    ClaimStatus,
    EvidenceBundle,
    EvidenceKind,
    EvidenceLevel,
    Freshness,
    NumericProvenance,
    PhysicalContract,
    ProvActivity,
    ProvAgent,
    ProvEntity,
    Substrate,
    ValidityDomain,
)

from .verbs import FPGA_DEPLOYMENT_SCHEMA, SC_INFERENCE_SCHEMA, STUDIO_ID


def _valid_digest(value: object) -> bool:
    """Recognise a complete, lowercase SHA-256 content address."""
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _wire_count(value: object) -> TypeGuard[int]:
    """Recognise a nonnegative integer preserved exactly by JavaScript JSON."""
    return type(value) is int and 0 <= value <= 2**53 - 1


def _finite_number(value: object) -> bool:
    """Recognise a finite numeric scalar without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _require_source_recheck(bit_exact: bool, freshness: Freshness | None) -> None:
    """Refuse source-unchecked parity and undeclared freshness values."""
    if freshness is not None and not isinstance(freshness, Freshness):
        raise ValueError("freshness must be a platform Freshness member")
    if bit_exact and freshness is not Freshness.VERIFIED_AT_SOURCE:
        raise ValueError("reference-validated parity requires evidence re-checked at source")


@dataclass(frozen=True)
class ScInferenceResult:
    """A path-free stochastic-computing inference result.

    Parameters
    ----------
    active_backend
        The accelerated backend that produced the result (e.g. ``"rust"``).
    reference_backend
        The bit-true reference backend (the NumPy floor).
    max_abs_error
        Maximum absolute difference between the active and reference outputs for
        the fixed seed; ``0.0`` means bit-identical.
    bitstream_length
        The stochastic bitstream length used.
    input_digest
        SHA-256 of the weights/inputs/seed.
    result_digest
        SHA-256 of the output firing rates.

    Raises
    ------
    ValueError
        If a count, error, backend or content address is invalid.
    """

    active_backend: str
    reference_backend: str
    max_abs_error: float
    bitstream_length: int
    input_digest: str
    result_digest: str

    def __post_init__(self) -> None:
        """Validate finite measured inputs and their SHA-256 identities."""
        if not _wire_count(self.bitstream_length) or self.bitstream_length == 0:
            raise ValueError("ScInferenceResult.bitstream_length must be a positive safe integer")
        if not _finite_number(self.max_abs_error) or self.max_abs_error < 0.0:
            raise ValueError("ScInferenceResult.max_abs_error must be finite and >= 0")
        if self.active_backend not in ("numpy", "rust") or self.reference_backend != "numpy":
            raise ValueError(
                "ScInferenceResult requires a supported backend and the NumPy reference"
            )
        if not _valid_digest(self.input_digest) or not _valid_digest(self.result_digest):
            raise ValueError(
                "ScInferenceResult digests must be lowercase SHA-256 content addresses"
            )


@dataclass(frozen=True)
class FpgaDeploymentResult:
    """A path-free pre-silicon FPGA synthesis/co-simulation result.

    Target device identity and synthesis reports do not attest physical execution.

    Parameters
    ----------
    device
        The target device (e.g. ``"xc7z020-1clg400"``).
    cosim_bit_exact
        Whether the synthesised RTL co-simulation matched the Q8.8 fixed-point
        reference bit-for-bit.
    lut_used, ff_used
        Look-up tables and flip-flops consumed by the synthesised design.
    worst_negative_slack_ns
        Worst negative slack at the target clock; ``>= 0`` means timing closed.
    clock_mhz
        The synthesis target clock frequency, in MHz.
    result_digest
        SHA-256 of the bitstream / synthesis report.

    Raises
    ------
    ValueError
        If a resource count, co-simulation flag, clock, timing or digest is invalid.
    """

    device: str
    cosim_bit_exact: bool
    lut_used: int
    ff_used: int
    worst_negative_slack_ns: float
    clock_mhz: float
    result_digest: str

    def __post_init__(self) -> None:
        """Validate finite report values and their SHA-256 identity."""
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("FpgaDeploymentResult.device must be non-empty")
        if type(self.cosim_bit_exact) is not bool:
            raise ValueError("FpgaDeploymentResult.cosim_bit_exact must be a boolean")
        if not all(_wire_count(value) for value in (self.lut_used, self.ff_used)):
            raise ValueError(
                "FpgaDeploymentResult resource counts must be nonnegative safe integers"
            )
        if not _finite_number(self.clock_mhz) or self.clock_mhz <= 0.0:
            raise ValueError("FpgaDeploymentResult.clock_mhz must be finite and > 0")
        if not _finite_number(self.worst_negative_slack_ns):
            raise ValueError("FpgaDeploymentResult timing must be finite")
        if not _valid_digest(self.result_digest):
            raise ValueError(
                "FpgaDeploymentResult digest must be a lowercase SHA-256 content address"
            )


def sc_inference_evidence(
    result: ScInferenceResult,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
    host: str | None = None,
    freshness: Freshness | None = None,
) -> EvidenceBundle:
    """Build the ``studio.sc-inference.v1`` bundle (measured software result).

    Parameters
    ----------
    result
        The path-free inference result.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the SC-NeuroCore studio that produced the result.
    started, ended
        ISO-8601 start/end timestamps (passed in; no hidden clock).
    host
        Optional host descriptor the run executed on.
    freshness
        Source re-check status supplied by the producer. A bit-identical result
        requires ``VERIFIED_AT_SOURCE``; parity alone never establishes freshness.

    Returns
    -------
    EvidenceBundle
        A ``measured`` bundle that renders as validated only when the accelerated
        backend is bit-identical to the NumPy floor.

    Raises
    ------
    ValueError
        If a bit-identical result lacks a verified-at-source re-check.
    """
    bit_identical = result.max_abs_error == 0.0
    _require_source_recheck(bit_identical, freshness)
    entity = ProvEntity(
        entity_id=f"{STUDIO_ID}/sc-inference/{result.result_digest}", digest=result.result_digest
    )
    activity = ProvActivity(
        verb="simulate", studio=STUDIO_ID, started=started, ended=ended, host=host
    )
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    numeric = NumericProvenance(
        active_backend=result.active_backend, reference_backend=result.reference_backend
    )
    claim = ClaimBoundary(
        status=ClaimStatus.REFERENCE_VALIDATED if bit_identical else ClaimStatus.BOUNDED_MODEL,
        admission=AdmissionDecision.ADMITTED if bit_identical else AdmissionDecision.REJECTED,
        validity_domain=ValidityDomain(
            note=(
                "Stochastic-computing forward pass; reference-validated only against the "
                "bit-true NumPy floor for a fixed seed, not against a third-party SNN simulator."
            )
        ),
    )
    return EvidenceBundle(
        schema=SC_INFERENCE_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.ENGINEERING_VERIFIED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=claim,
        freshness=freshness,
        numeric_provenance=numeric,
    )


def fpga_deployment_evidence(
    result: FpgaDeploymentResult,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
    host: str | None = None,
    freshness: Freshness | None = None,
) -> EvidenceBundle:
    """Build the ``studio.fpga-deployment.v1`` pre-silicon measurement bundle.

    RTL is co-simulated against the Q8.8 fixed-point reference on the
    ``simulator`` substrate. The claim is ``reference-validated`` only when that
    co-simulation is bit-exact and re-checked; a mismatch becomes ``validation-gap``.

    Parameters
    ----------
    result
        The path-free synthesis/deployment result.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the SC-NeuroCore studio that produced the result.
    started, ended
        ISO-8601 start/end timestamps (passed in; no hidden clock).
    host
        Optional host descriptor the synthesis ran on.
    freshness
        Source re-check status supplied by the producer. Bit-exact co-simulation
        requires ``VERIFIED_AT_SOURCE``; a retained report alone is not a re-check.

    Returns
    -------
    EvidenceBundle
        A ``measured`` bundle on the ``simulator`` substrate, without a board claim.

    Raises
    ------
    ValueError
        If a bit-exact co-simulation lacks a verified-at-source re-check.
    """
    _require_source_recheck(result.cosim_bit_exact, freshness)
    entity = ProvEntity(
        entity_id=f"{STUDIO_ID}/fpga-deployment/{result.result_digest}", digest=result.result_digest
    )
    activity = ProvActivity(
        verb="synthesise", studio=STUDIO_ID, started=started, ended=ended, host=host
    )
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    cosim = CaseResult(
        operation_family="cosim-bit-exact",
        dimension=1,
        status="pass" if result.cosim_bit_exact else "fail",
        error=0.0 if result.cosim_bit_exact else None,
    )
    physical = PhysicalContract(
        units={"lut": "count", "ff": "count", "wns": "ns", "clock": "MHz"},
        grid={
            "device": result.device,
            "lut_used": result.lut_used,
            "ff_used": result.ff_used,
        },
    )
    claim = ClaimBoundary(
        status=ClaimStatus.REFERENCE_VALIDATED
        if result.cosim_bit_exact
        else ClaimStatus.VALIDATION_GAP,
        admission=AdmissionDecision.ADMITTED
        if result.cosim_bit_exact
        else AdmissionDecision.REJECTED,
        validity_domain=ValidityDomain(
            note=(
                "Bit-exact Q8.8 reference vs RTL co-simulation for the named synthesis "
                "target. Pre-silicon measurement only; physical device execution, power "
                "and general floating-point accuracy are not established."
            )
        ),
    )
    return EvidenceBundle(
        schema=FPGA_DEPLOYMENT_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.ENGINEERING_VERIFIED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=claim,
        freshness=freshness,
        substrate=Substrate.SIMULATOR,
        cases=(cosim,),
        physical_contract=physical,
    )
