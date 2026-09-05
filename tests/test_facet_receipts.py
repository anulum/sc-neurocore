# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Facet receipt contract: sealing, credit rules and recording

"""A facet receipt credits a facet only when every rule of the contract holds."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sc_neurocore.neurons.facet_receipts import (
    FACET_BY_NAME,
    FACET_RECEIPT_SCHEMA,
    FACETS,
    INVALIDATION_MATRIX,
    FacetReceipt,
    FacetReceiptError,
    Subject,
    credit_problems,
    descriptor_contract_digest,
    facets_invalidated_by,
    latest_receipts,
    load_receipt,
    parse_receipt,
    receipt_filename,
    seal_digest,
)

_DIGEST = "0" * 64


def _subjects(facet: str) -> tuple[Subject, ...]:
    spec = FACET_BY_NAME[facet]
    return tuple(Subject(kind, f"path/{kind}.txt", _DIGEST) for kind in spec.required_subjects)


def _receipt(facet: str = "cosim", **overrides: Any) -> FacetReceipt:
    spec = FACET_BY_NAME[facet]
    fields: dict[str, Any] = {
        "class_name": "LapicqueNeuron",
        "facet": facet,
        "profile": "lapicque",
        "claim_scope": spec.claim_scope,
        "subjects": _subjects(facet),
        "evidence_refs": ("tests/test_cosim_lapicque.py::test_x",),
        "command": ("python", "-m", "pytest", "tests/test_cosim_lapicque.py::test_x"),
        "tool": {"name": "pytest", "version": "8.0"},
        "extra_tools": {"iverilog": "12.0"},
        "runtime": {"python": "3.12"},
        "validator": {"name": "tools/facet_receipt.py"},
        "outcome": "passed",
        "exit_code": 0,
        "counts": {"collected": 1, "passed": 1, "failed": 0, "errors": 0, "skipped": 0},
        "recorded_at": "2026-09-05T06:00:00Z",
    }
    fields.update(overrides)
    return FacetReceipt(**fields).sealed()


def test_sealed_receipt_round_trips_and_credits() -> None:
    """A sealed, passed receipt with every required subject is creditable."""
    receipt = _receipt()
    payload = receipt.to_payload()
    assert payload["schema"] == FACET_RECEIPT_SCHEMA
    assert payload["receipt_sha256"] == seal_digest(receipt.to_payload(sealed=False))
    parsed = parse_receipt(json.loads(json.dumps(payload)))
    assert parsed == receipt
    assert credit_problems(parsed, class_name="LapicqueNeuron") == ()


def test_tampered_receipt_breaks_the_seal() -> None:
    """Any edit after sealing is detected."""
    payload = _receipt().to_payload()
    payload["counts"]["passed"] = 2
    problems = credit_problems(parse_receipt(payload))
    assert any("seal" in problem for problem in problems)
    unsealed = _receipt().to_payload(sealed=False)
    assert any("not sealed" in problem for problem in credit_problems(parse_receipt(unsealed)))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"schema": "sc-neurocore.facet-receipt.v0"}, "unsupported receipt schema"),
        ({"facet": "psychic"}, "not a readiness facet"),
        ({"outcome": "maybe"}, "outcome"),
        ({"recorded_at": "2026-09-05 06:00"}, "recorded_at"),
        ({"command": []}, "command"),
        ({"class_name": "not a class"}, "identifier"),
        ({"receipt_sha256": "zz"}, "receipt_sha256"),
    ],
)
def test_parse_rejects_malformed_payloads(override: dict[str, Any], message: str) -> None:
    """Structural violations raise instead of producing a half-valid receipt."""
    payload = _receipt().to_payload()
    payload.update(override)
    with pytest.raises(FacetReceiptError, match=message):
        parse_receipt(payload)


@pytest.mark.parametrize(
    "subject",
    [
        {"kind": "descriptor-contract", "path": "/abs/path", "sha256": _DIGEST},
        {"kind": "descriptor-contract", "path": "../escape", "sha256": _DIGEST},
        {"kind": "descriptor-contract", "path": "x", "sha256": "abc"},
        {"kind": "mystery", "path": "x", "sha256": _DIGEST},
        {"kind": "descriptor-contract", "path": "x", "sha256": _DIGEST, "scope": "vibes"},
    ],
)
def test_parse_rejects_malformed_subjects(subject: dict[str, str]) -> None:
    """A fabricated, escaping or undigested subject cannot enter a receipt."""
    payload = _receipt().to_payload()
    payload["subjects"] = [subject]
    with pytest.raises(FacetReceiptError):
        parse_receipt(payload)


def test_failed_skipped_and_nonterminal_runs_cannot_credit() -> None:
    """Outcome, exit code and counts each gate the credit independently."""
    assert any("outcome" in p for p in credit_problems(_receipt(outcome="failed", exit_code=1)))
    assert any("exit code" in p for p in credit_problems(_receipt(exit_code=3)))
    assert any("timeout" in p for p in credit_problems(_receipt(outcome="timeout", exit_code=-1)))
    skipped = _receipt(counts={"collected": 2, "passed": 1, "failed": 0, "errors": 0, "skipped": 1})
    assert any("skipped" in p for p in credit_problems(skipped))
    nothing = _receipt(counts={"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0})
    assert any("no passed check" in p for p in credit_problems(nothing))


def test_wrong_class_and_missing_or_foreign_subjects_cannot_credit() -> None:
    """A receipt is read for one class and must carry exactly its facet's inputs."""
    assert any("not AdExNeuron" in p for p in credit_problems(_receipt(), class_name="AdExNeuron"))
    partial = _receipt(subjects=_subjects("cosim")[:-1])
    assert any("required subject kind" in p for p in credit_problems(partial))
    foreign = _receipt(subjects=(*_subjects("cosim"), Subject("report", "hdl/x.json", _DIGEST)))
    assert any("not an input of cosim" in p for p in credit_problems(foreign))


def test_bounded_safety_proof_cannot_credit_formal_equivalence() -> None:
    """Claim scopes keep a BMC safety run out of the H4 equivalence rung."""
    bmc_as_equivalence = _receipt("formal_equivalence", claim_scope="bounded-safety")
    assert any("claim scope" in p for p in credit_problems(bmc_as_equivalence))
    assert credit_problems(_receipt("formal_equivalence")) == ()
    assert credit_problems(_receipt("formal_safety")) == ()
    assert FACET_BY_NAME["formal_safety"].rung is None
    assert FACET_BY_NAME["formal_equivalence"].rung == 4


def test_invalidation_matrix_names_descendants_only() -> None:
    """Each subject kind invalidates exactly the facets that consume it."""
    assert set(INVALIDATION_MATRIX) == {spec.name for spec in FACETS}
    assert all("validator" in kinds for kinds in INVALIDATION_MATRIX.values())
    compiler = set(facets_invalidated_by("compiler"))
    assert {"rtl_compile", "cosim", "synthesis", "formal_equivalence"} <= compiler
    assert compiler.isdisjoint({"dynamics_faithful", "class_validated", "timing", "ppa"})
    assert not any(name.startswith("backend:") for name in compiler)
    assert set(facets_invalidated_by("report")) == {
        "synthesis",
        "timing",
        "formal_equivalence",
        "formal_safety",
        "ppa",
        "physical",
    }
    assert set(facets_invalidated_by("native-backend")) == {
        f"backend:{name}" for name in ("python", "rust", "julia", "go", "mojo")
    }
    assert "model-module" in INVALIDATION_MATRIX["cosim"]
    assert "model-module" not in INVALIDATION_MATRIX["synthesis"]


def test_descriptor_contract_digest_ignores_non_contract_sections() -> None:
    """Documentation and evidence edits keep the digest; contract edits change it."""
    payload: dict[str, Any] = {
        "metadata": {"name": "X", "class_name": "X", "module": "x", "summary": "one"},
        "state": {"v": {"init": 0.0}},
        "parameters": {"tau": {"default": 10.0}},
        "integration": {"dt": 0.1, "method": "euler"},
        "dynamics": {"v": "dv/dt=-v/tau"},
        "documentation": {"notes": "a"},
        "validation": {"evidence": "tests/a.py"},
    }
    base = descriptor_contract_digest(payload)
    edited = json.loads(json.dumps(payload))
    edited["documentation"]["notes"] = "b"
    edited["validation"]["evidence"] = "tests/b.py"
    edited["metadata"]["summary"] = "two"
    assert descriptor_contract_digest(edited) == base
    edited["parameters"]["tau"]["default"] = 20.0
    assert descriptor_contract_digest(edited) != base
    edited["parameters"]["tau"]["default"] = 10.0
    edited["integration"]["dt"] = 0.05
    assert descriptor_contract_digest(edited) != base


def test_latest_receipt_is_the_append_only_successor(tmp_path: Path) -> None:
    """The newest receipt per (class, facet) supersedes; older files stay."""
    older = _receipt(recorded_at="2026-09-05T06:00:00Z")
    newer = _receipt(recorded_at="2026-09-05T07:00:00Z", outcome="failed", exit_code=1)
    other = _receipt("class_validated")
    for receipt in (older, newer, other):
        name = receipt_filename(receipt.class_name, receipt.facet, receipt.recorded_at)
        (tmp_path / name).write_text(json.dumps(receipt.to_payload()), encoding="utf-8")
    latest = latest_receipts(tmp_path)
    assert latest[("LapicqueNeuron", "cosim")][1] == newer
    assert latest[("LapicqueNeuron", "class_validated")][1] == other
    assert len(list(tmp_path.glob("*.json"))) == 3
    assert receipt_filename("A", "backend:rust", "2026-09-05T06:00:00Z") == (
        "A__backend-rust__20260905T060000Z.json"
    )
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    with pytest.raises(FacetReceiptError):
        latest_receipts(tmp_path)


def _load_recorder() -> ModuleType:
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "facet_receipt.py"
    spec = importlib.util.spec_from_file_location("facet_receipt_tool", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recorder_counts_come_from_the_run_not_the_exit_code(tmp_path: Path) -> None:
    """A pytest run records collected/passed/skipped from its JUnit report."""
    recorder = _load_recorder()
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import pytest\n"
        "def test_ok():\n    pass\n"
        "@pytest.mark.skip(reason='x')\ndef test_skip():\n    pass\n",
        encoding="utf-8",
    )
    exit_code, counts, outcome, tool = recorder.run_command(
        [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:nengo"], timeout=300
    )
    assert exit_code == 0
    assert tool["name"] == "pytest"
    assert counts == {"collected": 2, "passed": 1, "failed": 0, "errors": 0, "skipped": 1}
    assert outcome == "passed"
    probe.write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    exit_code, counts, outcome, _tool = recorder.run_command(
        [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:nengo"], timeout=300
    )
    assert exit_code != 0
    assert counts["failed"] == 1
    assert outcome == "failed"
    exit_code, counts, outcome, tool = recorder.run_command(
        [sys.executable, "-c", "raise SystemExit(0)"], timeout=60
    )
    assert (exit_code, outcome, counts["passed"]) == (0, "passed", 1)
    assert tool["name"] == Path(sys.executable).name


def test_recorder_writes_an_immutable_creditable_receipt(tmp_path: Path) -> None:
    """Recording a real model against a passing command yields a sealed receipt."""
    recorder = _load_recorder()
    probe = tmp_path / "test_probe.py"
    probe.write_text("def test_ok():\n    pass\n", encoding="utf-8")
    command = [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:nengo"]
    path, receipt = recorder.record_receipt(
        model="LapicqueNeuron",
        facet="class_validated",
        command=command,
        receipt_dir=tmp_path / "receipts",
        recorded_at="2026-09-05T06:00:00Z",
        timeout=300,
    )
    assert path.name == "LapicqueNeuron__class_validated__20260905T060000Z.json"
    assert credit_problems(load_receipt(path), class_name="LapicqueNeuron") == ()
    assert receipt.profile == "lapicque"
    assert {subject.kind for subject in receipt.subjects} >= {
        "descriptor-contract",
        "model-module",
        "validator",
    }
    assert receipt.runtime["sc_neurocore"]
    assert recorder.committed_rtl_path("LapicqueNeuron") == (
        "hdl/formal/catalogue/sc_lapicque_1907.v"
    )
    assert recorder.committed_rtl_path("AdExNeuron") == "hdl/formal/catalogue/sc_adex.v"
    assert recorder.committed_rtl_path("SCInclusivePerfectIntegratorNeuron") == (
        "hdl/formal/catalogue/sc_perfect_integrator.v"
    )
    assert recorder.committed_rtl_path("AkidaNeuron") is None
    with pytest.raises(recorder.RecordError, match="immutable"):
        recorder.record_receipt(
            model="LapicqueNeuron",
            facet="class_validated",
            command=command,
            receipt_dir=tmp_path / "receipts",
            recorded_at="2026-09-05T06:00:00Z",
            timeout=300,
        )
    with pytest.raises(recorder.RecordError, match="not a registered class"):
        recorder.record_receipt(
            model="GhostNeuron", facet="cosim", command=command, receipt_dir=tmp_path
        )
    with pytest.raises(recorder.RecordError, match="does not resolve"):
        recorder.record_receipt(
            model="LapicqueNeuron",
            facet="cosim",
            command=command,
            evidence=["tests/test_cosim_lapicque.py::test_never_written"],
            receipt_dir=tmp_path,
        )
