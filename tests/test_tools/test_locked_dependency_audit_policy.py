# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Blocking dependency audit policy contracts

"""Pin unconditional workflow dispatch, scanner commands and aggregate refusal."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_every_head_requires_all_dependency_ecosystems() -> None:
    """Require six unconditional audit lanes and a failure-propagating aggregate."""
    workflow = cast(
        dict[str, Any],
        yaml.load(
            (ROOT / ".github/workflows/dependency-audit.yml").read_text(), Loader=yaml.BaseLoader
        ),
    )
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["on"]["push"] == ""
    assert workflow["on"]["pull_request"] == {"branches": ["main"]}
    jobs = workflow["jobs"]
    lane = jobs["audit"]
    assert "if" not in lane and "continue-on-error" not in lane
    assert lane["strategy"]["fail-fast"] == "false"
    assert set(lane["strategy"]["matrix"]["ecosystem"]) == {
        "python",
        "rust",
        "npm",
        "go",
        "julia",
        "pixi",
    }
    steps = lane["steps"]
    for step in steps:
        assert "continue-on-error" not in step
        if "uses" in step:
            revision = step["uses"].split("@", 1)[1]
            assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    commands = [step["run"] for step in steps if "run" in step]
    assert not any("||" in command or "set +e" in command for command in commands)
    audit = next(
        step for step in steps if step.get("name") == "Audit every committed lock in this ecosystem"
    )
    assert "if" not in audit
    assert audit["run"].strip() == (
        'python -m tools.security_scan.locked_dependency_audit \\\n  --ecosystem "$AUDIT_ECOSYSTEM" \\\n  --output-dir "$RUNNER_TEMP/dependency-audit/$AUDIT_ECOSYSTEM"'
    )
    gate = jobs["dependency-lock-audit"]
    assert gate["name"] == "dependency-lock-audit" and gate["needs"] == ["audit"]
    assert gate["if"] == "always()" and "continue-on-error" not in gate
    assert gate["steps"] == [
        {
            "name": "Require all ecosystem audits to succeed",
            "env": {"AUDIT_RESULT": "${{ needs.audit.result }}"},
            "run": 'test "$AUDIT_RESULT" = success',
        }
    ]


def test_advisory_commands_cannot_silently_filter_findings() -> None:
    """Pin committed-lock commands and forbid audit suppression or stale databases."""
    source = (ROOT / "tools/security_scan/locked_dependency_audit.py").read_text()
    literals = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "cargo",
        "audit",
        "--file",
        "--format",
        "json",
        "--deny",
        "warnings",
        "--url",
        "https://github.com/RustSec/advisory-db",
        "npm",
        "--package-lock-only",
        "--include=dev",
        "--include=optional",
        "--include=peer",
        "--registry=https://registry.npmjs.org",
        "--json",
        "pixi-audit==0.1.1=ha35fb5c_30",
    } <= literals
    assert not literals.intersection(
        {
            "--severity",
            "--ignore",
            "--ignore-unfixed",
            "--fix",
            "--no-fetch",
            "--stale",
            "--offline",
            "--prod",
            "--omit",
            "--audit-level",
            "--allow-no-lockfiles",
        }
    )
    assert "inventory_locks(repo_root) != inventory" in source
    assert "run_dependency_audit(repo_root=repo_root, output_dir=output_dir)" in source
    report_source = (ROOT / "tools/security_scan/locked_dependency_reports.py").read_text()
    assert 'OSV_ENDPOINT = "https://api.osv.dev/v1/querybatch"' in report_source
    assert '"ignored", "unchecked", "unmatched_ignores"' in report_source
