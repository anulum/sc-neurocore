# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLAN_TOOL_PATH = _REPO_ROOT / "tools" / "security_scan" / "rust_supply_chain_scanner_plan.py"
_MANIFEST_TOOL_PATH = _REPO_ROOT / "tools" / "security_scanner_manifest.py"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_plan_tool() -> Any:
    return _load_module(_PLAN_TOOL_PATH, "rust_supply_chain_scanner_plan")


def _load_manifest_module() -> Any:
    return _load_module(_MANIFEST_TOOL_PATH, "security_scanner_manifest")


def _fake_manifest(scanner: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "sc-neurocore.security-scanner-manifest.v1",
        "scanners": scanner,
    }


def _has_executable_factory(executables: dict[str, str | None]) -> Callable[[str], str | None]:
    def _fake_which(name: str) -> str | None:
        return executables.get(name)

    return _fake_which


def _scanner_entry(plan: dict[str, Any], name: str) -> dict[str, Any]:
    entry = next((entry for entry in plan["scanners"] if entry["name"] == name), None)
    assert entry is not None
    return entry


def test_plan_includes_expected_scanner_names_sorted_deterministically() -> None:
    manifest_tool = _load_manifest_module()
    plan_tool = _load_plan_tool()

    manifest = manifest_tool.build_scanner_manifest()
    plan = plan_tool.build_rust_supply_chain_plan(
        manifest,
        repo_root=_REPO_ROOT,
        include_heavy=True,
    )
    scanner_names = [scanner["name"] for scanner in plan["scanners"]]

    assert scanner_names == [
        "actionlint",
        "cargo-audit",
        "cargo-deny",
        "cargo-fuzz-nightly",
        "osv-scanner",
        "reuse",
        "syft-cyclonedx",
        "trivy fs",
    ]
    assert plan["scanner_count"] == len(scanner_names)

    second = plan_tool.build_rust_supply_chain_plan(
        manifest,
        repo_root=_REPO_ROOT,
        include_heavy=True,
    )
    assert json.dumps(plan, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_heavy_scanners_default_to_deferred_without_include() -> None:
    plan_tool = _load_plan_tool()
    manifest = _fake_manifest(
        [
            {
                "name": "trivy fs",
                "command": "trivy fs . --format json --output security/trivy_fs.json",
                "inputs": [{"path": ".", "required": True, "purpose": "repo", "name": "repo"}],
            },
            {
                "name": "cargo-fuzz-nightly",
                "command": "cargo fuzz run --dev --fuzz-dir fuzz -- -dict=/dev/null",
                "inputs": [
                    {"path": "fuzz", "required": True, "purpose": "fuzz", "name": "fuzz"},
                    {"path": "src/sc_neurocore", "required": True, "purpose": "src", "name": "src"},
                ],
            },
        ]
    )

    plan = plan_tool.build_rust_supply_chain_plan(
        manifest,
        repo_root=_REPO_ROOT,
        include_heavy=False,
        has_executable=_has_executable_factory(
            {"cargo": "/usr/bin/cargo", "trivy": "/usr/bin/trivy"}
        ),
    )
    classes = {entry["name"]: entry["run_class"] for entry in plan["scanners"]}
    assert classes["trivy fs"] == "deferred_heavy"
    assert classes["cargo-fuzz-nightly"] == "deferred_heavy"


def test_missing_required_input_marks_missing_required_input(tmp_path: Path) -> None:
    plan_tool = _load_plan_tool()
    manifest = _fake_manifest(
        [
            {
                "name": "cargo-audit",
                "command": "cargo audit --format json --file security/cargo_audit.json",
                "inputs": [
                    {
                        "path": "missing-lock-file",
                        "required": True,
                        "purpose": "lock",
                        "name": "lock",
                    },
                    {
                        "path": "Cargo.toml",
                        "required": True,
                        "purpose": "manifest",
                        "name": "manifest",
                    },
                ],
            }
        ]
    )

    repo_root = tmp_path / "repo-missing-input"
    repo_root.mkdir(exist_ok=True)
    (repo_root / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")

    plan = plan_tool.build_rust_supply_chain_plan(
        manifest,
        repo_root=repo_root,
        include_heavy=True,
        has_executable=_has_executable_factory({"cargo": "/usr/bin/cargo"}),
    )
    entry = _scanner_entry(plan, "cargo-audit")
    assert entry["run_class"] == "missing_required_input"
    assert "missing-lock-file" in entry["missing_required_inputs"]


def test_missing_tool_marks_missing_tool(tmp_path: Path) -> None:
    plan_tool = _load_plan_tool()
    manifest = _fake_manifest(
        [
            {
                "name": "actionlint",
                "command": "actionlint -format json -out security/actionlint.json .github/workflows",
                "inputs": [
                    {
                        "path": ".github/workflows",
                        "required": True,
                        "purpose": "workflows",
                        "name": "workflows",
                    }
                ],
            }
        ]
    )

    repo_root = tmp_path / "repo-missing-tool"
    workflows_root = repo_root / ".github" / "workflows"
    workflows_root.mkdir(parents=True, exist_ok=True)

    plan = plan_tool.build_rust_supply_chain_plan(
        manifest,
        repo_root=repo_root,
        include_heavy=True,
        has_executable=_has_executable_factory({"actionlint": None}),
    )
    assert plan["scanners"][0]["run_class"] == "missing_tool"
    assert plan["scanners"][0]["executable"] == "actionlint"


def test_include_heavy_checks_required_inputs_and_executables(tmp_path: Path) -> None:
    plan_tool = _load_plan_tool()
    manifest = _fake_manifest(
        [
            {
                "name": "cargo-fuzz-nightly",
                "command": "cargo fuzz run --dev --fuzz-dir fuzz -- -dict=/dev/null",
                "inputs": [{"path": "fuzz", "required": True, "purpose": "fuzz", "name": "fuzz"}],
            }
        ]
    )

    repo_root = tmp_path / "repo-heavy-checks"
    repo_root.mkdir(exist_ok=True)

    plan = plan_tool.build_rust_supply_chain_plan(
        manifest,
        repo_root=repo_root,
        include_heavy=True,
        has_executable=_has_executable_factory({"cargo": "/usr/bin/cargo"}),
    )
    entry = _scanner_entry(plan, "cargo-fuzz-nightly")
    assert entry["run_class"] == "missing_required_input"

    (repo_root / "fuzz").mkdir(exist_ok=True)
    plan_tool_with_no_tool = plan_tool.build_rust_supply_chain_plan(
        manifest,
        repo_root=repo_root,
        include_heavy=True,
        has_executable=_has_executable_factory({"cargo": None}),
    )
    missing_tool_entry = _scanner_entry(plan_tool_with_no_tool, "cargo-fuzz-nightly")
    assert missing_tool_entry["run_class"] == "missing_tool"


def test_cli_output_and_fail_on_missing_required_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_tool = _load_plan_tool()
    manifest = _fake_manifest(
        [
            {
                "name": "reuse",
                "command": "reuse lint --root .",
                "inputs": [
                    {"path": "REUSE.toml", "required": True, "purpose": "reuse", "name": "reuse"}
                ],
            }
        ]
    )

    repo_root = tmp_path

    monkeypatch.setattr(plan_tool, "_load_manifest", lambda: manifest)
    monkeypatch.setattr(plan_tool, "_project_root", lambda: repo_root)
    monkeypatch.setattr(
        plan_tool,
        "_has_executable",
        _has_executable_factory({"reuse": "/usr/bin/reuse"}),
    )

    output = repo_root / "plan.json"
    assert plan_tool.main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert _scanner_entry(payload, "reuse")["run_class"] == "missing_required_input"

    assert plan_tool.main(["--output", str(output), "--fail-on-missing-required-inputs"]) == 1
