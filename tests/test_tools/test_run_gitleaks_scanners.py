# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Gitleaks Scanner Runner Tests

"""Tests of the Gitleaks scanner lane and of the repository's Gitleaks allowlist."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "security_scan" / "run_gitleaks_scanners.py"
CONFIG_PATH = REPO_ROOT / ".gitleaks.toml"
RULE = "generic-api-key"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("run_gitleaks_scanners", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _allowlists() -> list[dict[str, Any]]:
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return list(config["allowlists"])


def _allowlisted(rule: str, path: str, secret: str) -> bool:
    """Apply the configured allowlists as Gitleaks does for AND allowlists on the secret."""
    for allowlist in _allowlists():
        if rule not in allowlist["targetRules"]:
            continue
        if not any(re.search(pattern, path) for pattern in allowlist["paths"]):
            continue
        if any(re.search(pattern, secret) for pattern in allowlist["regexes"]):
            return True
    return False


def _hex64(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _base62(label: str, length: int = 32) -> str:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return "".join(alphabet[byte % 62] for byte in hashlib.sha256(label.encode()).digest())[:length]


def test_gitleaks_scanner_writes_low_noise_release_artifacts(tmp_path: Path) -> None:
    """The lane runs the repository's configuration, records its digest and writes both artefacts."""
    tool = _load_tool()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output_path = Path(command[command.index("--report-path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="clean\n", stderr="")

    summary = tool.run_gitleaks_scanner(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        run_command=fake_run,
    )

    assert calls == [
        [
            "gitleaks",
            "detect",
            "--source",
            str(REPO_ROOT),
            "--report-format",
            "json",
            "--report-path",
            str(tmp_path / "security" / "gitleaks.json"),
            "--no-banner",
            "--redact",
            "--config",
            str(CONFIG_PATH),
        ]
    ]
    assert summary["schema_version"] == tool.GITLEAKS_SCANNER_SCHEMA_VERSION
    assert summary["passed"] is True
    assert summary["finding_count"] == 0
    assert summary["allowed_to_fail"] is True
    assert summary["config"] == str(CONFIG_PATH)
    assert summary["config_sha256"] == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    written = json.loads((tmp_path / "security" / "gitleaks_summary.json").read_text())
    assert written == summary


def test_gitleaks_scanner_runs_the_default_rules_without_a_configuration(tmp_path: Path) -> None:
    """A tree without ``.gitleaks.toml`` runs Gitleaks without ``--config`` and records no digest."""
    tool = _load_tool()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_gitleaks_scanner(
        repo_root=tmp_path / "tree",
        output_dir=tmp_path / "out",
        run_command=fake_run,
    )

    assert "--config" not in calls[0]
    assert summary["config"] is None
    assert summary["config_sha256"] is None
    assert summary["passed"] is True


def test_gitleaks_scanner_records_findings_without_blocking_packet(tmp_path: Path) -> None:
    """Findings are counted and reported, and the lane stays non-blocking."""
    tool = _load_tool()

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--report-path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([{"RuleID": RULE}]) + "\n")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="1 leak found\n")

    summary = tool.run_gitleaks_scanner(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        run_command=fake_run,
    )

    assert summary["passed"] is True
    assert summary["finding_count"] == 1
    assert summary["leak_detected"] is True
    assert summary["returncode"] == 1


def test_gitleaks_scanner_cli_returns_nonzero_for_execution_failure(tmp_path: Path) -> None:
    """The command line returns 1 when the lane could not produce a valid report."""
    tool = _load_tool()

    def fake_runner(**_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": tool.GITLEAKS_SCANNER_SCHEMA_VERSION,
            "passed": False,
            "finding_count": 0,
            "validation_errors": ["missing Gitleaks report artifact"],
        }

    assert (
        tool.main(
            ["--repo-root", str(REPO_ROOT), "--output-dir", str(tmp_path)], runner=fake_runner
        )
        == 1
    )


def test_gitleaks_configuration_keeps_every_default_rule_and_narrow_allowlists() -> None:
    """Every allowlist names one rule, existing files and anchored value shapes, all required together."""
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["extend"] == {"useDefault": True}
    assert "allowlist" not in config
    assert "rules" not in config
    allowlists = config["allowlists"]
    assert len(allowlists) == 4
    for allowlist in allowlists:
        assert set(allowlist) == {
            "description",
            "targetRules",
            "condition",
            "paths",
            "regexTarget",
            "regexes",
        }
        assert allowlist["targetRules"] == [RULE]
        assert allowlist["condition"] == "AND"
        assert allowlist["regexTarget"] == "secret"
        assert allowlist["paths"] and allowlist["regexes"]
        for pattern in [*allowlist["paths"], *allowlist["regexes"]]:
            assert pattern.startswith("^") and pattern.endswith("$"), pattern
        for pattern in allowlist["paths"]:
            relative = pattern[1:-1].replace("\\.", ".")
            assert (REPO_ROOT / relative).is_file(), relative


def test_gitleaks_allowlists_cover_the_reviewed_values_and_no_planted_key() -> None:
    """The reviewed digests and names are allowed; a key of another shape, file or rule is not."""
    lapicque_bench = "benchmarks/results/bench_lapicque.json"
    vip = "engine/src/neurons/interneurons/vip_neuron.rs"
    reviewed = [
        (RULE, lapicque_bench, _hex64("a reviewed content digest")),
        (RULE, "benchmarks/results/local_python_2026-06-17_lapicque_exact_flow.json", _hex64("b")),
        (RULE, "tests/fixtures/engine_abi_default.json", _hex64("c")),
        (RULE, vip, "irregular/accommodating"),
        (RULE, "tests/test_rust_python_neuron_parity.py", "py_izhikevich2007_simulate"),
        (RULE, "tools/emit_catalogue_formal.py", "sc_lapicque_1907"),
    ]
    planted = [
        (RULE, lapicque_bench, _base62("planted in an allowlisted JSON file")),
        (RULE, vip, _base62("planted in the allowlisted Rust file")),
        (RULE, "tools/planted_settings.py", _hex64("a digest-shaped key in an unnamed file")),
        (RULE, lapicque_bench, _hex64("x").upper()),
        (RULE, vip, "irregular/accommodating-and-more"),
        ("aws-access-token", "tests/fixtures/engine_abi_default.json", _hex64("another rule")),
    ]
    assert all(_allowlisted(*case) for case in reviewed)
    assert not any(_allowlisted(*case) for case in planted)
