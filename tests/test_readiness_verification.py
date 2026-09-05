# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Readiness verification: declared versus bound, invalidation

"""Declared readiness never outranks its evidence; one change invalidates descendants only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tomli_w

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from sc_neurocore.neurons.descriptor_tiers import science_tier, silicon_tier
from sc_neurocore.neurons.evidence_references import sha256_file, sha256_tree
from sc_neurocore.neurons.facet_receipts import (
    FACET_BY_NAME,
    FACETS,
    FacetReceipt,
    Subject,
    descriptor_contract_digest,
    parse_receipt,
    receipt_filename,
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
    lapicque = report["LapicqueNeuron"]
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


# --- invalidation matrix against a synthetic repository -----------------------------

_DESCRIPTOR = {
    "metadata": {"name": "Probe", "class_name": "Probe", "module": "probe", "summary": "s"},
    "state": {"v": {"init": 0.0}},
    "parameters": {"tau": {"default": 10.0}},
    "integration": {"dt": 0.1, "method": "euler"},
    "dynamics": {"v": "dv/dt=-v/tau"},
    "documentation": {"notes": "before"},
}


def _toml(payload: dict[str, object]) -> str:
    return tomli_w.dumps(payload)


def _synthetic_repo(tmp_path: Path) -> Path:
    (tmp_path / "desc").mkdir()
    (tmp_path / "desc" / "Probe.toml").write_text(_toml(_DESCRIPTOR), encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "probe.py").write_text("STATE = 1\n", encoding="utf-8")
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "probe.toml").write_text("[a]\nb = 1\n", encoding="utf-8")
    (tmp_path / "compiler").mkdir()
    (tmp_path / "compiler" / "emit.py").write_text("VERSION = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_probe.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    (tmp_path / "rtl").mkdir()
    (tmp_path / "rtl" / "probe.v").write_text("module probe; endmodule\n", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "synth.json").write_text('{"cells": 1}\n', encoding="utf-8")
    return tmp_path


def _subject(repo: Path, kind: str, relative: str, scope: str = "file") -> Subject:
    path = repo / relative
    if scope == "tree":
        digest = sha256_tree(sorted(path.rglob("*.py")), repo)
    elif scope == "contract-sections":
        digest = descriptor_contract_digest(tomllib.loads(path.read_text(encoding="utf-8")))
    else:
        digest = sha256_file(path)
    return Subject(kind, relative, digest, scope)  # type: ignore[arg-type]


def _receipt(repo: Path, facet: str, subjects: tuple[Subject, ...]) -> FacetReceipt:
    spec = FACET_BY_NAME[facet]
    return FacetReceipt(
        class_name="Probe",
        facet=facet,
        profile="probe",
        claim_scope=spec.claim_scope,
        subjects=subjects,
        evidence_refs=("tests/test_probe.py::test_x",),
        command=("python", "-m", "pytest", "tests/test_probe.py::test_x"),
        tool={"name": "pytest", "version": "8"},
        extra_tools={},
        runtime={"python": "3.12"},
        validator={"name": "tools/facet_receipt.py"},
        outcome="passed",
        exit_code=0,
        counts={"collected": 1, "passed": 1, "failed": 0, "errors": 0, "skipped": 0},
        recorded_at="2026-09-05T06:00:00Z",
    ).sealed()


def _receipts(repo: Path) -> dict[str, FacetReceipt]:
    descriptor = _subject(repo, "descriptor-contract", "desc/Probe.toml", "contract-sections")
    module = _subject(repo, "model-module", "models/probe.py")
    schema = _subject(repo, "schema-profile", "schemas/probe.toml")
    compiler = _subject(repo, "compiler", "compiler", "tree")
    validator = _subject(repo, "validator", "tests/test_probe.py")
    rtl = _subject(repo, "committed-rtl", "rtl/probe.v")
    synth = _subject(repo, "report", "reports/synth.json")
    return {
        "class_validated": _receipt(
            repo, "class_validated", (descriptor, module, schema, validator)
        ),
        "cosim": _receipt(repo, "cosim", (descriptor, module, schema, compiler, validator, rtl)),
        "synthesis": _receipt(repo, "synthesis", (rtl, synth, validator)),
    }


def _statuses(
    repo: Path, receipts: dict[str, FacetReceipt]
) -> dict[str, tuple[str, tuple[str, ...]]]:
    return {
        facet: verify_receipt(receipt, class_name="Probe", repo_root=repo)[:2]
        for facet, receipt in receipts.items()
    }


def test_one_changed_subject_invalidates_only_its_descendants(tmp_path: Path) -> None:
    """Compiler, contract, RTL and report edits each stale exactly their consumers."""
    repo = _synthetic_repo(tmp_path)
    receipts = _receipts(repo)
    assert {facet: status for facet, (status, _c) in _statuses(repo, receipts).items()} == {
        "class_validated": "bound",
        "cosim": "bound",
        "synthesis": "bound",
    }

    (repo / "compiler" / "emit.py").write_text("VERSION = 2\n", encoding="utf-8")
    statuses = _statuses(repo, receipts)
    assert statuses["cosim"] == ("stale", ("compiler:compiler",))
    assert statuses["class_validated"][0] == "bound"
    assert statuses["synthesis"][0] == "bound"
    (repo / "compiler" / "emit.py").write_text("VERSION = 1\n", encoding="utf-8")

    edited = json.loads(json.dumps(_DESCRIPTOR))
    edited["documentation"]["notes"] = "after"
    (repo / "desc" / "Probe.toml").write_text(_toml(edited), encoding="utf-8")
    assert {f: s for f, (s, _c) in _statuses(repo, receipts).items()} == {
        "class_validated": "bound",
        "cosim": "bound",
        "synthesis": "bound",
    }
    edited["parameters"]["tau"]["default"] = 20.0
    (repo / "desc" / "Probe.toml").write_text(_toml(edited), encoding="utf-8")
    statuses = _statuses(repo, receipts)
    assert statuses["class_validated"] == ("stale", ("descriptor-contract:desc/Probe.toml",))
    assert statuses["cosim"][0] == "stale"
    assert statuses["synthesis"][0] == "bound"
    (repo / "desc" / "Probe.toml").write_text(_toml(_DESCRIPTOR), encoding="utf-8")

    (repo / "rtl" / "probe.v").write_text("module probe2; endmodule\n", encoding="utf-8")
    statuses = _statuses(repo, receipts)
    assert statuses["synthesis"] == ("stale", ("committed-rtl:rtl/probe.v",))
    assert statuses["cosim"] == ("stale", ("committed-rtl:rtl/probe.v",))
    assert statuses["class_validated"][0] == "bound"
    (repo / "rtl" / "probe.v").write_text("module probe; endmodule\n", encoding="utf-8")

    (repo / "reports" / "synth.json").unlink()
    statuses = _statuses(repo, receipts)
    assert statuses["synthesis"] == ("stale", ("report:reports/synth.json (missing)",))
    assert statuses["cosim"][0] == "bound"
    assert statuses["class_validated"][0] == "bound"


def test_fabricated_and_tampered_receipts_are_invalid_not_stale(tmp_path: Path) -> None:
    """A receipt that fails the credit rules never reaches the freshness check."""
    repo = _synthetic_repo(tmp_path)
    receipts = _receipts(repo)
    payload = receipts["cosim"].to_payload()
    payload["outcome"] = "failed"
    tampered = parse_receipt(payload)
    status, changed, problems = verify_receipt(tampered, class_name="Probe", repo_root=repo)
    assert status == "invalid"
    assert changed == ()
    assert problems
    wrong_class = verify_receipt(receipts["cosim"], class_name="Other", repo_root=repo)
    assert wrong_class[0] == "invalid"
    name = receipt_filename("Probe", "cosim", "2026-09-05T06:00:00Z")
    assert name == "Probe__cosim__20260905T060000Z.json"
    assert len(FACETS) == len({spec.name for spec in FACETS})
