# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio federation vertical tests

"""Contract tests for the SC-NeuroCore studio federation vertical.

Guarded by ``pytest.importorskip`` so the suite skips cleanly when the optional
``scpn-studio-platform`` SDK is absent (the CI base dev-lock), and runs in full
where the ``federation`` extra is installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scpn_studio_platform")

from scpn_studio_platform.evidence import (  # noqa: E402
    ClaimStatus,
    EvidenceKind,
    Substrate,
    validate_studio_bundle,
)
from scpn_studio_platform.verbs import CORE_VERBS  # noqa: E402

from sc_neurocore.federation import (  # noqa: E402
    NEUROCORE_VERBS,
    STUDIO_ID,
    FpgaDeploymentResult,
    ScInferenceResult,
    build_manifest,
    core_verbs,
    domain_verbs,
    evidence_schemas,
    fpga_deployment_evidence,
    sc_inference_evidence,
)

_TS1 = "2026-06-24T06:00:00Z"
_TS2 = "2026-06-24T06:05:00Z"


def test_manifest_is_well_formed_and_digest_is_reproducible() -> None:
    manifest = build_manifest(studio_version="3.15.34")

    assert manifest.studio == STUDIO_ID == "sc-neurocore"
    assert manifest.platform_sdk == ">=0.9,<0.10"
    assert len(manifest.verbs) == len(NEUROCORE_VERBS) == 8
    assert tuple(manifest.evidence_types) == evidence_schemas()
    # content digest is over the declared surface, not git state — reproducible.
    assert manifest.content_digest == build_manifest(studio_version="3.15.34").content_digest
    assert manifest.content_digest.startswith("sha256:")


def test_verbs_split_core_spine_from_domain_distinctive() -> None:
    core = {v.name for v in core_verbs()}
    domain = {v.name for v in domain_verbs()}
    assert core <= set(CORE_VERBS)  # every "core" verb is in the platform spine
    assert domain == {"encode", "deploy"}  # SC-NeuroCore's distinctive verbs
    assert core.isdisjoint(domain)
    assert core | domain == {v.name for v in NEUROCORE_VERBS}


def test_fpga_deployment_is_hardware_validated_and_federates() -> None:
    bundle = fpga_deployment_evidence(
        FpgaDeploymentResult(
            device="xc7z020-1clg400",
            cosim_bit_exact=True,
            lut_used=1317,
            ff_used=848,
            worst_negative_slack_ns=4.048,
            clock_mhz=100.0,
            result_digest="a" * 64,
        ),
        operator="opaque:tenant-1",
        studio_version="3.15.34",
        started=_TS1,
        ended=_TS2,
    )
    assert bundle.evidence_kind is EvidenceKind.HARDWARE_VALIDATED
    assert bundle.substrate is Substrate.FPGA
    assert bundle.claim_boundary.status is ClaimStatus.REFERENCE_VALIDATED

    verdict = validate_studio_bundle(bundle.to_dict())
    assert verdict.admitted is True
    assert verdict.mode == "validated"
    assert verdict.rejections == ()


def test_sc_inference_validated_only_when_bit_identical() -> None:
    identical = sc_inference_evidence(
        ScInferenceResult("rust", "numpy", 0.0, 1024, "b" * 64, "c" * 64),
        operator="o",
        studio_version="3.15.34",
        started=_TS1,
        ended=_TS2,
    )
    assert identical.evidence_kind is EvidenceKind.MEASURED
    assert identical.claim_boundary.status is ClaimStatus.REFERENCE_VALIDATED
    assert validate_studio_bundle(identical.to_dict()).admitted is True

    drifted = sc_inference_evidence(
        ScInferenceResult("rust", "numpy", 1e-3, 1024, "b" * 64, "c" * 64),
        operator="o",
        studio_version="3.15.34",
        started=_TS1,
        ended=_TS2,
    )
    assert drifted.claim_boundary.status is ClaimStatus.BOUNDED_MODEL


def test_cosim_mismatch_never_renders_validated() -> None:
    # The honesty invariant from SC-NeuroCore's own perspective: a hardware
    # co-simulation MISMATCH must never federate as validated silicon.
    mismatch = fpga_deployment_evidence(
        FpgaDeploymentResult(
            device="xc7z020-1clg400",
            cosim_bit_exact=False,
            lut_used=1,
            ff_used=1,
            worst_negative_slack_ns=-1.0,
            clock_mhz=100.0,
            result_digest="d" * 64,
        ),
        operator="o",
        studio_version="3.15.34",
        started=_TS1,
        ended=_TS2,
    )
    assert mismatch.claim_boundary.status is ClaimStatus.VALIDATION_GAP
    assert mismatch.claim_boundary.status is not ClaimStatus.REFERENCE_VALIDATED
    verdict = validate_studio_bundle(mismatch.to_dict())
    assert verdict.mode != "validated"


@pytest.mark.parametrize(
    ("length", "error"),
    [(0, 0.0), (-1, 0.0), (1024, -0.1)],
)
def test_sc_inference_result_rejects_invalid(length: int, error: float) -> None:
    with pytest.raises(ValueError):
        ScInferenceResult("rust", "numpy", error, length, "b" * 64, "c" * 64)


def test_fpga_result_rejects_empty_digest() -> None:
    with pytest.raises(ValueError):
        FpgaDeploymentResult("dev", True, 1, 1, 0.0, 100.0, "   ")
