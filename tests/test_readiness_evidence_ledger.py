# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated readiness ledger freshness and committed receipt custody

"""The tracked readiness ledger and the committed receipts mirror the live verifier."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sc_neurocore.neurons.facet_receipts import (
    FACETS,
    INVALIDATION_MATRIX,
    RECEIPT_DIR,
    credit_problems,
    iter_receipts,
    latest_receipts,
    receipt_filename,
)
from sc_neurocore.neurons.model_identity import identity_registry
from sc_neurocore.neurons.readiness import (
    FACET_STATUSES,
    readiness_report,
    summarise,
    verify_receipt,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_tool() -> ModuleType:
    tool_path = _repo_root() / "tools" / "readiness_evidence_ledger.py"
    spec = importlib.util.spec_from_file_location("readiness_evidence_ledger", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ledger() -> dict[str, Any]:
    path = _repo_root() / "docs/_generated/readiness_evidence_ledger.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def test_tracked_ledger_is_current() -> None:
    """The generated ledger equals a fresh render of the live verifier."""
    tool = _load_tool()
    assert tool.ledger_problems(_repo_root()) == []


def test_ledger_payload_mirrors_the_verifier() -> None:
    """Every model, facet definition and summary in the ledger is the live one."""
    payload = _ledger()
    assert payload["schema"] == "sc-neurocore.readiness-evidence-ledger.v1"
    assert set(payload["status_definition"]) == set(FACET_STATUSES)
    assert [facet["name"] for facet in payload["facets"]] == [spec.name for spec in FACETS]
    assert payload["invalidation_matrix"] == {
        name: list(kinds) for name, kinds in INVALIDATION_MATRIX.items()
    }
    records = readiness_report()
    assert payload["summary"] == summarise(records.values())
    models = {row["class_name"]: row for row in payload["models"]}
    expected = {
        name for name, identity in identity_registry().items() if identity.kind != "api-alias"
    }
    assert set(models) == expected
    for name, record in records.items():
        assert models[name] == record.to_public_dict()
    assert "timestamp" not in payload
    assert "head" not in payload


def test_committed_receipts_are_creditable_named_and_fresh() -> None:
    """Every receipt is sealed and correctly named; the newest per facet is still bound.

    Superseded receipts stay in the store as history and may be stale; only the
    newest receipt of each (class, facet) must match the current subjects.
    """
    receipts = list(iter_receipts(RECEIPT_DIR))
    assert receipts, "the receipt store must hold the representative receipts"
    for path, receipt in receipts:
        assert path.name == receipt_filename(
            receipt.class_name, receipt.facet, receipt.recorded_at
        )
        assert credit_problems(receipt, class_name=receipt.class_name) == ()
        assert receipt.runtime["git_head"]
        assert receipt.counts["passed"] >= 1
        assert receipt.counts["skipped"] == 0
    for path, receipt in latest_receipts(RECEIPT_DIR).values():
        status, changed, problems = verify_receipt(receipt, class_name=receipt.class_name)
        assert (status, changed, problems) == ("bound", (), ()), path.name


def test_summary_cli_prints_the_partition(capsys: pytest.CaptureFixture[str]) -> None:
    """The --summary mode prints the same partition the ledger carries."""
    tool = _load_tool()
    assert tool.main(["--summary"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == _ledger()["summary"]
