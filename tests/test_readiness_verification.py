# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Readiness verification: declared versus bound, invalidation

"""Declared readiness never outranks its evidence; one change invalidates descendants only."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from sc_neurocore.neurons.descriptor_tiers import science_tier, silicon_tier
from sc_neurocore.neurons.facet_receipts import (
    FACET_BY_NAME,
    FacetReceipt,
    Subject,
    parse_receipt,
)
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_identity import identity_registry
from sc_neurocore.neurons.readiness import (
    FACET_STATUSES,
    ReadinessRecord,
    compiler_subjects,
    declared_facets,
    derive_subjects,
    readiness_report,
    summarise,
    verify_model,
    verify_receipt,
)


@pytest.fixture(scope="module")
def report() -> dict[str, ReadinessRecord]:
    """Verify the whole corpus once per module."""
    return dict(readiness_report())


def test_every_declared_evidence_reference_resolves(report: dict[str, ReadinessRecord]) -> None:
    """Tier 1 gate: no descriptor names a file or test node that does not exist."""
    unavailable = [
        (name, facet.facet, facet.problems)
        for name, record in report.items()
        for facet in record.facets
        if facet.status == "unavailable"
    ]
    assert unavailable == []
    dangling = [
        (name, reference.raw)
        for name, record in report.items()
        for facet in record.facets
        for reference in facet.evidence
        if reference.is_locatable and not reference.is_resolved
    ]
    assert dangling == []


def test_verified_never_outranks_declared_and_statuses_are_vocabulary(
    report: dict[str, ReadinessRecord],
) -> None:
    """Verified tiers are bounded by declared tiers; every status is a known word."""
    for record in report.values():
        assert record.verified_science <= record.declared_science
        declared_h = record.declared_silicon
        verified_h = record.verified_silicon
        assert verified_h is None or (declared_h is not None and verified_h <= declared_h)
        bound = {f.facet for f in record.facets if f.status == "bound"}
        for facet in record.facets:
            assert facet.status in FACET_STATUSES
            assert facet.declared or facet.status == "not-declared"
        if verified_h is not None:
            assert bound & {"rtl_compile", "cosim"}
        if record.verified_science >= 4:
            assert "dynamics_faithful" in bound
        if record.verified_science == 5:
            assert "class_validated" in bound


def test_declared_facets_mirror_the_tier_anchors() -> None:
    """A declared facet here is exactly a credited rung in descriptor_tiers."""
    for name, identity in identity_registry().items():
        if identity.kind == "api-alias":
            continue
        descriptor = load_descriptor(name)
        assert descriptor is not None
        declared = declared_facets(descriptor)
        science = science_tier(descriptor)
        silicon = silicon_tier(descriptor)
        if science >= 4:
            assert declared["dynamics_faithful"]
        if science == 5:
            assert declared["class_validated"]
        if silicon is not None:
            assert declared["rtl_compile"]
        if silicon is not None and silicon >= 1:
            assert declared["cosim"]
        if silicon is not None and silicon >= 2:
            assert declared["synthesis"]
        assert declared["backend:python"] == any(
            b.name == "python" and b.status == "implemented" for b in descriptor.backends
        )


def test_corrected_pointers_are_bound_and_withdrawn_claim_is_not_declared(
    report: dict[str, ReadinessRecord],
) -> None:
    """The four repaired pointers carry executed receipts; the withdrawn one is honest."""
    for name in ("AdaptiveThresholdIFNeuron", "BrunelWangNeuron", "ResonateAndFireNeuron"):
        cosim = report[name].facet("cosim")
        assert cosim.status == "bound", (name, cosim.problems, cosim.changed_subjects)
        assert cosim.receipt.startswith(f"{name}__cosim__")
    assert report["McCullochPittsNeuron"].facet("class_validated").status == "bound"
    lapicque = verify_model("LapicqueNeuron", profile="lapicque")
    assert lapicque.verified_science == 5
    assert lapicque.verified_silicon == 1
    assert lapicque.facet("synthesis").status == "located"
    withdrawn = report["SCInclusivePerfectIntegratorNeuron"]
    assert withdrawn.facet("cosim").status == "not-declared"
    assert withdrawn.declared_silicon == 0


def test_summary_counts_every_model_once(report: dict[str, ReadinessRecord]) -> None:
    """The summary is a partition of the corpus on every axis."""
    summary = summarise(report.values())
    assert summary["models"] == len(report)
    for key in (
        "declared_science_tiers",
        "verified_science_tiers",
        "declared_silicon_tiers",
        "verified_silicon_tiers",
    ):
        assert sum(summary[key].values()) == len(report)
    for statuses in summary["facet_statuses"].values():
        assert sum(statuses.values()) == len(report)


def test_derived_subjects_cover_every_required_kind() -> None:
    """The registry derives every required subject for a schema-bound model."""
    for facet in ("dynamics_faithful", "class_validated", "cosim", "rtl_compile"):
        kinds = {subject.kind for subject in derive_subjects("LapicqueNeuron", facet)}
        assert set(FACET_BY_NAME[facet].required_subjects) <= kinds, facet
    cosim = derive_subjects("LapicqueNeuron", "cosim")
    assert any(s.scope == "tree" and s.kind == "compiler" for s in cosim)
    assert "committed-rtl" not in {s.kind for s in cosim}
    rtl = Subject("committed-rtl", "hdl/formal/catalogue/sc_lapicque_1907.v", "0" * 64)
    with_rtl = derive_subjects("LapicqueNeuron", "cosim", extra_subjects=(rtl,))
    assert rtl in with_rtl
    assert rtl not in derive_subjects("LapicqueNeuron", "class_validated", extra_subjects=(rtl,))
    assert all(subject.kind == "compiler" for subject in compiler_subjects())


def test_unregistered_class_is_an_error() -> None:
    """Verification of an unknown class fails instead of returning an empty record."""
    with pytest.raises(KeyError):
        verify_model("GhostNeuron")


@pytest.fixture
def copied_receipt(tmp_path: Path) -> tuple[Path, FacetReceipt]:
    """Copy actual evidence inputs, never fabricate a scientifically passing model."""
    import shutil

    from sc_neurocore.neurons.facet_receipts import latest_receipts

    root = Path(__file__).resolve().parents[1]
    receipt = latest_receipts()[("AdaptiveThresholdIFNeuron", "cosim", "adaptive_threshold_if")][1]
    assert verify_receipt(receipt, class_name=receipt.class_name)[0] == "bound"
    for subject in receipt.subjects:
        source, target = root / subject.path, tmp_path / subject.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    return tmp_path, receipt


def test_one_changed_subject_invalidates_only_its_descendants(
    copied_receipt: tuple[Path, FacetReceipt],
) -> None:
    """Real committed cosim inputs stale on edits while documentation does not."""
    repo, receipt = copied_receipt
    assert verify_receipt(receipt, class_name=receipt.class_name, repo_root=repo)[0] == "bound"
    descriptor = next(s for s in receipt.subjects if s.kind == "descriptor-contract")
    path = repo / descriptor.path
    original = path.read_text()
    payload = tomllib.loads(original)
    payload["metadata"]["summary"] = "changed description"
    path.write_text(tomli_w.dumps(payload))
    assert verify_receipt(receipt, class_name=receipt.class_name, repo_root=repo)[0] == "bound"
    key = next(iter(payload["parameters"]))
    payload["parameters"][key]["default"] = 12345.0
    path.write_text(tomli_w.dumps(payload))
    assert verify_receipt(receipt, class_name=receipt.class_name, repo_root=repo)[0] == "stale"
    path.write_text(original)
    compiler = next(s for s in receipt.subjects if s.kind == "compiler" and s.scope == "tree")
    member = next((repo / compiler.path).rglob("*.py"))
    member.write_text(member.read_text() + "\n# changed compiler input\n")
    assert verify_receipt(receipt, class_name=receipt.class_name, repo_root=repo)[0] == "stale"


def test_current_manifest_rejects_omitted_inputs(
    copied_receipt: tuple[Path, FacetReceipt],
) -> None:
    """Resealing a partial set of a required kind does not make it complete."""
    from dataclasses import replace

    repo, receipt = copied_receipt
    compiler = [s for s in receipt.subjects if s.kind == "compiler"]
    assert len(compiler) > 1
    partial = replace(
        receipt, subjects=tuple(s for s in receipt.subjects if s != compiler[0])
    ).sealed()
    status, _changed, problems = verify_receipt(
        partial, class_name=receipt.class_name, repo_root=repo
    )
    assert status == "invalid"
    assert any("missing current subject" in p for p in problems)


def test_fabricated_and_tampered_receipts_are_invalid_not_stale(
    copied_receipt: tuple[Path, FacetReceipt],
) -> None:
    """A wrong identity or changed result cannot borrow an executed receipt."""
    repo, receipt = copied_receipt
    payload = receipt.to_payload()
    payload["outcome"] = "failed"
    status, changed, problems = verify_receipt(
        parse_receipt(payload), class_name=receipt.class_name, repo_root=repo
    )
    assert (status, changed) == ("invalid", ())
    assert problems
    assert verify_receipt(receipt, class_name="LapicqueNeuron", repo_root=repo)[0] == "invalid"


def test_profile_specific_readiness_does_not_promote_other_profiles() -> None:
    """Lapicque source evidence does not confer a class-wide LIF guarantee."""
    assert verify_model("LapicqueNeuron").verified_science < 4
    assert verify_model("LapicqueNeuron", profile="lapicque").verified_science == 5
    assert verify_model("LapicqueNeuron", profile="lif").verified_science < 4
    with pytest.raises(ValueError, match="unknown profile"):
        verify_model("LapicqueNeuron", profile="not-a-profile")
