# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real typing scanner entrypoints

"""Exercise real scanners on the maintained runner and actual failure cases."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tools.security_scan.run_typing_scanners import (
    main,
    run_typing_scanners,
    validate_typing_output,
)
from tools.security_scanner_manifest import build_scanner_manifest

REPO = Path(__file__).resolve().parents[2]
SOURCE = "tools/security_scan/run_typing_scanners.py"


@pytest.fixture(autouse=True)
def active_python_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the current interpreter's tools before unrelated user installations."""
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])


def test_manifest_typing_commands_are_executable_and_pinned() -> None:
    """Bind the public scanner plan to the executable typing adapter."""
    manifest = build_scanner_manifest()
    scanners = {scanner["name"]: scanner for scanner in manifest["scanners"]}
    assert scanners["pyright"]["pinned_version"] == "pyright==1.1.414"
    assert scanners["pyright"]["command"] == "pyright --project pyrightconfig.json --outputjson"
    assert scanners["mypy"]["pinned_version"] == "mypy==2.3.1"
    assert scanners["mypy"]["command"] == (
        "python tools/security_scan/run_typing_scanners.py --output-dir ."
    )


def test_runner_writes_pyright_json_and_mypy_report_summary(tmp_path: Path) -> None:
    """Write current diagnostics for the maintained runner without scanning the repository."""
    packet = tmp_path / "packet"
    summary = run_typing_scanners(repo_root=REPO, output_dir=packet, paths=(SOURCE,))
    report = json.loads((packet / "security/mypy/index.json").read_text())
    assert report["execution"]["returncode"] == 0
    assert report["diagnostics"] == []
    assert "--output=json" in report["execution"]["command"]
    assert "--strict" in report["execution"]["command"]
    assert summary["scope"] == {"kind": "explicit-paths", "paths": [SOURCE]}
    stored = json.loads((packet / "security/typing_scanner_summary.json").read_text())
    assert stored == summary
    mypy = next(row for row in stored["scanners"] if row["name"] == "mypy")
    assert mypy["passed"] and mypy["report_valid"] and mypy["version_valid"]
    pyright = next(row for row in stored["scanners"] if row["name"] == "pyright")
    assert stored["passed"] == pyright["passed"]
    if pyright["passed"]:
        actual = json.loads((packet / "security/pyright.json").read_text())
        assert actual["summary"]["filesAnalyzed"] >= 1
        assert actual["summary"]["errorCount"] == 0
    else:
        assert "pyright" in stored["failed_scanners"]


def test_runner_records_invalid_pyright_json_as_raw_output(tmp_path: Path) -> None:
    """Replace stale reports when the actual process cannot enter its working directory."""
    packet = tmp_path / "packet"
    security = packet / "security"
    (security / "mypy").mkdir(parents=True)
    (security / "mypy/index.json").write_text('{"old": true}')
    (security / "pyright.json").write_text('{"summary":{"errorCount":0}}')
    missing = tmp_path / "absent-repository"
    summary = run_typing_scanners(repo_root=missing, output_dir=packet, paths=(SOURCE,))
    assert summary["passed"] is False
    assert summary["failed_scanners"] == ["pyright", "mypy"]
    pyright = json.loads((security / "pyright.json").read_text())
    assert pyright == {"raw_stdout": ""}
    mypy = json.loads((security / "mypy/index.json").read_text())
    assert "old" not in mypy
    assert mypy["execution"]["execution_error"] == "FileNotFoundError"
    assert mypy["execution"]["returncode"] is None


def test_runner_resolves_tools_next_to_active_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the real interpreter-sibling executable when PATH supplies no tools."""
    monkeypatch.setenv("PATH", "")
    """Write current diagnostics for the maintained runner without scanning the repository."""
    packet = tmp_path / "packet"
    summary = run_typing_scanners(repo_root=REPO, output_dir=packet, paths=(SOURCE,))
    stored = json.loads((packet / "security/typing_scanner_summary.json").read_text())
    mypy = next(row for row in stored["scanners"] if row["name"] == "mypy")
    assert Path(mypy["command"][0]).parent == Path(sys.executable).parent
    assert mypy["passed"]
    assert summary["scanner_count"] == 2


def test_cli_retains_real_type_errors_and_returns_failure(tmp_path: Path) -> None:
    """Retain real checker diagnostics for a corrupted copy of the production module."""
    source = (REPO / SOURCE).read_text()
    original = 'TYPING_SCANNER_SCHEMA_VERSION = "sc-neurocore.typing-scanners.v2"'
    assert original in source
    defective = tmp_path / "run_typing_scanners.py"
    defective.write_text(source.replace(original, original.replace(" = ", ": int = ")))
    packet = tmp_path / "packet"
    assert (
        main(["--repo-root", str(REPO), "--output-dir", str(packet), "--paths", str(defective)])
        == 1
    )
    report = json.loads((packet / "security/mypy/index.json").read_text())
    assert report["execution"]["returncode"] == 1
    assert any(row["code"] == "assignment" for row in report["diagnostics"])
    stored = json.loads((packet / "security/typing_scanner_summary.json").read_text())
    assert "mypy" in stored["failed_scanners"]
    mypy = next(row for row in stored["scanners"] if row["name"] == "mypy")
    assert mypy["report_valid"]


def test_runner_records_actual_process_timeouts(tmp_path: Path) -> None:
    """Record an actual bounded subprocess timeout as failed execution."""
    packet = tmp_path / "packet"
    summary = run_typing_scanners(
        repo_root=REPO,
        output_dir=packet,
        paths=(SOURCE,),
        pyright_timeout=0.000001,
        mypy_timeout=0.000001,
    )
    assert summary["passed"] is False
    stored = json.loads((packet / "security/typing_scanner_summary.json").read_text())
    mypy = next(row for row in stored["scanners"] if row["name"] == "mypy")
    assert mypy["execution_error"] == "TimeoutExpired"
    assert mypy["returncode"] is None


@pytest.mark.parametrize("budget", [0.0, -1.0, float("inf"), float("nan")])
def test_runner_rejects_invalid_process_budgets(tmp_path: Path, budget: float) -> None:
    """Reject budgets that cannot bound a real checker process."""
    with pytest.raises(ValueError, match="positive and finite"):
        run_typing_scanners(repo_root=REPO, output_dir=tmp_path, mypy_timeout=budget)


@pytest.fixture(scope="module")
def captured_typing_reports(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Capture diagnostic bytes through the real runner on its production source."""
    packet = tmp_path_factory.mktemp("typing-reports")
    run_typing_scanners(repo_root=REPO, output_dir=packet, paths=(SOURCE,))
    return {
        name: (packet / "security" / f"{name}.stdout.log").read_text()
        for name in ("pyright", "mypy")
    }


@pytest.mark.parametrize(
    "damage",
    [
        "invalid-json",
        "not-object",
        "no-summary",
        "no-diagnostics",
        "bad-counter",
        "no-files",
        "wrong-version",
        "nonobject-diagnostic",
        "bad-severity",
        "bad-message",
        "inconsistent-count",
        "wrong-exit",
        "boolean-exit",
    ],
)
def test_offline_pyright_validation_rejects_damaged_actual_output(
    captured_typing_reports: dict[str, str], damage: str
) -> None:
    """Refuse corruptions of captured tool evidence through the public validator."""
    raw = captured_typing_reports["pyright"]
    report = json.loads(raw or "{}")
    status: object = 0
    if damage == "invalid-json":
        raw = raw + "["
    elif damage == "not-object":
        raw = json.dumps([report])
    else:
        if damage == "no-summary":
            report.pop("summary", None)
        elif damage == "no-diagnostics":
            report.pop("generalDiagnostics", None)
        elif damage == "bad-counter":
            report.setdefault("summary", {})["errorCount"] = True
        elif damage == "no-files":
            report.setdefault("summary", {})["filesAnalyzed"] = 0
        elif damage == "wrong-version":
            report["version"] = "unqualified"
        elif damage == "nonobject-diagnostic":
            report["generalDiagnostics"] = [None]
        elif damage == "bad-severity":
            report["generalDiagnostics"] = [{"severity": [], "message": "damaged"}]
        elif damage == "bad-message":
            report["generalDiagnostics"] = [{"severity": "error", "message": None}]
        elif damage == "inconsistent-count":
            report.setdefault("summary", {})["errorCount"] = 1
        elif damage == "wrong-exit":
            status = 2
        elif damage == "boolean-exit":
            status = False
        raw = json.dumps(report)
    _, valid = validate_typing_output("pyright", raw, status)
    assert not valid


@pytest.mark.parametrize(
    "damage", ["bad-json", "nonobject", "bad-file", "bad-line", "bad-severity"]
)
def test_offline_mypy_validation_rejects_invalid_diagnostics(tmp_path: Path, damage: str) -> None:
    """Reject malformed NDJSON derived from a real checker diagnostic."""
    source = (REPO / SOURCE).read_text()
    original = 'TYPING_SCANNER_SCHEMA_VERSION = "sc-neurocore.typing-scanners.v2"'
    defective = tmp_path / "run_typing_scanners.py"
    defective.write_text(source.replace(original, original.replace(" = ", ": int = ")))
    packet = tmp_path / "packet"
    run_typing_scanners(repo_root=REPO, output_dir=packet, paths=(str(defective),))
    raw = (packet / "security/mypy.stdout.log").read_text()
    diagnostic = json.loads(raw.splitlines()[0])
    if damage == "bad-json":
        raw += "["
    elif damage == "nonobject":
        raw = json.dumps([diagnostic])
    else:
        if damage == "bad-file":
            diagnostic["file"] = None
        elif damage == "bad-line":
            diagnostic["line"] = True
        else:
            diagnostic["severity"] = "unqualified"
        raw = json.dumps(diagnostic)
    _, valid = validate_typing_output("mypy", raw, 1)
    assert not valid


def test_offline_validation_rejects_unknown_scanner() -> None:
    """Require a declared format rather than accepting arbitrary output as evidence."""
    with pytest.raises(ValueError, match="Unknown typing scanner"):
        validate_typing_output("other", "", 0)


def test_cli_refuses_invalid_budget_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Return an authored argument refusal before attempting a checker process."""
    with pytest.raises(SystemExit) as stopped:
        main(["--output-dir", str(tmp_path), "--mypy-timeout", "0"])
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "Typing process budgets must be positive and finite." in error
    assert "Traceback" not in error
