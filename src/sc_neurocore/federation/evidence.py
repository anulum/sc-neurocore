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
- :func:`fpga_deployment_evidence` is the ``hardware-validated`` silicon tier: the
  synthesised RTL is co-simulated against the Q8.8 fixed-point reference, on the
  ``fpga`` substrate. Its claim is ``reference-validated`` only when the
  co-simulation is bit-exact; a mismatch degrades to ``validation-gap``.

Neither outcome is an SC-NeuroCore-private rule; each falls out of the shared
:class:`scpn_studio_platform.evidence.ClaimBoundary` lattice and the orthogonal
:class:`scpn_studio_platform.evidence.EvidenceKind` and ``Substrate`` axes.
Timestamps and digests are passed in by the caller (no hidden clock), so a bundle
is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from scpn_studio_platform.evidence import (
    AdmissionDecision,
    CaseResult,
    ClaimBoundary,
    ClaimStatus,
    EvidenceBundle,
    EvidenceKind,
    EvidenceLevel,
    NumericProvenance,
    PhysicalContract,
    ProvActivity,
    ProvAgent,
    ProvEntity,
    Substrate,
    ValidityDomain,
)

from .verbs import FPGA_DEPLOYMENT_SCHEMA, SC_INFERENCE_SCHEMA, STUDIO_ID


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
        If a count is non-positive, an error is negative, or a digest is empty.
    """

    active_backend: str
    reference_backend: str
    max_abs_error: float
    bitstream_length: int
    input_digest: str
    result_digest: str

    def __post_init__(self) -> None:
        """Validate counts, error sign, and digests."""
        if self.bitstream_length <= 0:
            raise ValueError("ScInferenceResult.bitstream_length must be > 0")
        if self.max_abs_error < 0.0:
            raise ValueError("ScInferenceResult.max_abs_error must be >= 0")
        if not self.input_digest.strip() or not self.result_digest.strip():
            raise ValueError("ScInferenceResult digests must be non-empty")


@dataclass(frozen=True)
class FpgaDeploymentResult:
    """A path-free FPGA synthesis/deployment result.

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
        If a resource count is negative, the clock is non-positive, or the digest
        is empty.
    """

    device: str
    cosim_bit_exact: bool
    lut_used: int
    ff_used: int
    worst_negative_slack_ns: float
    clock_mhz: float
    result_digest: str

    def __post_init__(self) -> None:
        """Validate resource counts, clock, and digest."""
        if self.lut_used < 0 or self.ff_used < 0:
            raise ValueError("FpgaDeploymentResult resource counts must be >= 0")
        if self.clock_mhz <= 0.0:
            raise ValueError("FpgaDeploymentResult.clock_mhz must be > 0")
        if not self.result_digest.strip():
            raise ValueError("FpgaDeploymentResult.result_digest must be non-empty")


def sc_inference_evidence(
    result: ScInferenceResult,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
    host: str | None = None,
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

    Returns
    -------
    EvidenceBundle
        A ``measured`` bundle that renders as validated only when the accelerated
        backend is bit-identical to the NumPy floor.
    """
    bit_identical = result.max_abs_error == 0.0
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
) -> EvidenceBundle:
    """Build the ``studio.fpga-deployment.v1`` bundle (hardware-validated silicon).

    The synthesised RTL is co-simulated against the Q8.8 fixed-point reference on
    the ``fpga`` substrate. The claim is ``reference-validated`` only when that
    co-simulation is bit-exact; a mismatch degrades to ``validation-gap``.

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

    Returns
    -------
    EvidenceBundle
        A ``hardware-validated`` bundle on the ``fpga`` substrate.
    """
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
                "Bit-exact Q8.8 reference vs synthesised RTL co-simulation on the named "
                "device; hardware-validated for that fixed-point network, not a general "
                "floating-point accuracy guarantee."
            )
        ),
    )
    return EvidenceBundle(
        schema=FPGA_DEPLOYMENT_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.ENGINEERING_VERIFIED,
        evidence_kind=EvidenceKind.HARDWARE_VALIDATED,
        claim_boundary=claim,
        substrate=Substrate.FPGA,
        cases=(cosim,),
        physical_contract=physical,
    )
