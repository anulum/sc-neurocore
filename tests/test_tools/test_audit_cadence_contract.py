# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Audit cadence workflow contract

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def _repo_root() -> Path:
    """Return the repository root containing the audit cadence workflow."""

    return Path(__file__).resolve().parents[2]


def _load_workflow() -> dict[str, Any]:
    """Load the audit cadence workflow through the same YAML surface CI uses."""

    workflow = _repo_root() / ".github" / "workflows" / "audit-cadence.yml"
    return cast(dict[str, Any], yaml.safe_load(workflow.read_text(encoding="utf-8")))


def _workflow_events(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return GitHub event configuration while preserving YAML's boolean key quirk."""

    raw_events = workflow.get("on")
    if raw_events is None:
        raw_events = cast(dict[Any, Any], workflow).get(True, {})
    return cast(dict[str, Any], raw_events)


def _run_text(workflow: dict[str, Any]) -> str:
    """Return all shell snippets from the test-inventory job."""

    steps = workflow["jobs"]["test-inventory"]["steps"]
    return "\n".join(step["run"] for step in steps if isinstance(step, dict) and "run" in step)


def test_audit_cadence_workflow_is_monthly_and_manual() -> None:
    """Keep the recurring audit cadence explicit and low-privilege."""

    workflow = _load_workflow()
    events = _workflow_events(workflow)

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["test-inventory"]["permissions"] == {"contents": "read"}
    assert "workflow_dispatch" in events
    assert events["schedule"] == [{"cron": "17 4 1 * *"}]
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_audit_cadence_collects_inventory_without_running_full_suite() -> None:
    """Ensure the monthly lane stays an inventory audit, not a full test run."""

    run_text = _run_text(_load_workflow())

    assert "python -m pytest tests/ --collect-only -q" in run_text
    assert "tools/test_inventory_audit.py" in run_text
    assert "--output audit-inventory.json" in run_text
    assert "python -m pytest tests/ -q" not in run_text
    assert "--cov" not in run_text


def test_audit_cadence_builds_collection_gating_backends() -> None:
    """Keep native parity modules collectable instead of module-skipped."""

    workflow = _load_workflow()
    steps = workflow["jobs"]["test-inventory"]["steps"]
    uses = [str(step.get("uses", "")) for step in steps if isinstance(step, dict)]
    run_text = _run_text(workflow)

    assert any(item.startswith("actions/setup-go@") for item in uses)
    assert any(item.startswith("prefix-dev/setup-pixi@") for item in uses)
    assert "for model in rk4_neurons wilson_cowan" in run_text
    assert "mojo build --emit shared-lib --target-cpu x86-64-v3" in run_text


def test_audit_cadence_is_documented_and_in_navigation() -> None:
    """Keep the workflow, public guide, and MkDocs navigation connected."""

    repo = _repo_root()
    docs = (repo / "docs" / "development" / "audit_cadence.md").read_text(encoding="utf-8")
    mkdocs = (repo / "mkdocs.yml").read_text(encoding="utf-8")

    assert "Audit Cadence" in docs
    assert "tools/test_inventory_audit.py" in docs
    assert "audit-inventory.json" in docs
    assert "development/audit_cadence.md" in mkdocs
