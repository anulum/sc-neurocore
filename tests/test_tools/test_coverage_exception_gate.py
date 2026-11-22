# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The coverage union gate with its platform exception register

"""Hold the coverage exception gate to its register, its evidence record and its refusals.

Each test writes a checkout with a register, an evidence record and the sources they name,
then runs the gate's command line on a ``coverage json`` report of the shape coverage.py writes.
One test runs the gate on this repository's own register and evidence.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import tools.coverage_exception_gate as gate

#: The excepted line in the scratch checkout.
LINE_TEXT = '    return coder, (coder.device_name, coder.device_api), "gpu"'

#: The sources the scratch register digests.
SOURCES = ("engine/gpu.rs", "src/pkg/_gpu.py")


def _sha(text: str) -> str:
    """Return the hex sha256 of a text's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checkout(root: Path, **overrides: Any) -> None:
    """Write a checkout whose register, evidence and sources agree, unless overridden.

    Parameters
    ----------
    root : Path
        The checkout.
    **overrides : Any
        ``register`` or ``evidence`` fields to replace.
    """
    (root / "engine").mkdir(parents=True)
    (root / "src/pkg").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "engine/gpu.rs").write_text("fn gpu() {}\n", encoding="utf-8")
    (root / "src/pkg/_gpu.py").write_text(f'"""GPU."""\n\n\n{LINE_TEXT}\n', encoding="utf-8")
    register = {
        "schema": gate.REGISTER_SCHEMA,
        "decisions": ["523df20b", "fd2fcd90"],
        "evidence": "tools/evidence.json",
        "source_set": list(SOURCES),
        "entries": [
            {
                "path": "src/pkg/_gpu.py",
                "line": 4,
                "source_sha256": _sha(LINE_TEXT),
                "reason": "GPU",
            }
        ],
    }
    register.update(overrides.get("register", {}))
    evidence = {
        "schema": gate.EVIDENCE_SCHEMA,
        "decisions": ["523df20b", "fd2fcd90"],
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in SOURCES
        },
        "excepted_lines_covered": [{"path": "src/pkg/_gpu.py", "line": 4}],
    }
    evidence.update(overrides.get("evidence", {}))
    (root / gate.REGISTER).write_text(json.dumps(register), encoding="utf-8")
    (root / "tools/evidence.json").write_text(json.dumps(evidence), encoding="utf-8")


def _report(path: Path, files: dict[str, dict[str, Any]]) -> Path:
    """Write a coverage json report with the given per-file entries and return its path."""
    report = path / "coverage.json"
    report.write_text(json.dumps({"meta": {"version": "7.17.3"}, "files": files}), encoding="utf-8")
    return report


def _file(
    statements: int, missing: list[int], branches: list[list[int]] | None = None
) -> dict[str, Any]:
    """Return one file's entry in a coverage json report."""
    entry: dict[str, Any] = {"summary": {"num_statements": statements}, "missing_lines": missing}
    if branches is not None:
        entry["missing_branches"] = branches
    return entry


def _run(root: Path, report: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    """Run the gate's command line and return its exit status and its output."""
    status = gate.main(["--coverage-json", str(report), "--root", str(root)])
    captured = capsys.readouterr()
    return status, captured.out + captured.err


def test_a_union_missing_only_the_registered_line_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The registered line may be missing; everything else is covered, so the gate passes.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the verdict.
    """
    _checkout(tmp_path)
    report = _report(tmp_path, {"src/pkg/_gpu.py": _file(3, [4]), "src/pkg/other.py": _file(5, [])})
    status, out = _run(tmp_path, report, capsys)
    assert status == 0
    assert "7 of 8 statements covered; 1 of 1 missing excepted by the register; pass" in out


def test_a_run_on_a_gpu_host_that_covers_the_line_also_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A union that reaches the registered line needs no exception and still passes.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the verdict.
    """
    _checkout(tmp_path)
    status, out = _run(tmp_path, _report(tmp_path, {"src/pkg/_gpu.py": _file(3, [])}), capsys)
    assert status == 0
    assert "0 of 0 missing excepted" in out


@pytest.mark.parametrize(
    ("files", "finding"),
    [
        (
            {"src/pkg/_gpu.py": _file(3, [3, 4])},
            "src/pkg/_gpu.py:3 is not covered and not in the register",
        ),
        ({"src/pkg/x.py": _file(2, [], [[1, 2]])}, "src/pkg/x.py: branch [1, 2] is not covered"),
    ],
)
def test_a_missing_line_or_branch_the_register_does_not_name_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], files: dict[str, Any], finding: str
) -> None:
    """An unregistered missing line fails the union, and a missing branch can never be excepted.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the findings.
    files : dict
        The report's per-file entries.
    finding : str
        The finding the gate must print.
    """
    _checkout(tmp_path)
    status, out = _run(tmp_path, _report(tmp_path, files), capsys)
    assert status == 1
    assert finding in out


def test_a_registered_line_whose_text_changed_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Editing the excepted line invalidates its registration, and the GPU evidence with it.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the findings.
    """
    _checkout(tmp_path)
    (tmp_path / "src/pkg/_gpu.py").write_text(
        '"""GPU."""\n\n\n    return coder\n', encoding="utf-8"
    )
    status, out = _run(tmp_path, _report(tmp_path, {"src/pkg/_gpu.py": _file(3, [4])}), capsys)
    assert status == 1
    assert "src/pkg/_gpu.py:4 changed since it was registered" in out
    assert "src/pkg/_gpu.py differs from the physical-GPU evidence; record it again" in out


def test_a_registered_line_past_the_end_of_its_file_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A register that points beyond the file, or at a file that is gone, is a changed line.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the findings.
    """
    _checkout(
        tmp_path,
        register={
            "entries": [
                {"path": "src/pkg/_gpu.py", "line": 9, "source_sha256": _sha(LINE_TEXT)},
                {"path": "src/pkg/gone.py", "line": 1, "source_sha256": _sha(LINE_TEXT)},
            ]
        },
    )
    status, out = _run(tmp_path, _report(tmp_path, {"src/pkg/_gpu.py": _file(3, [])}), capsys)
    assert status == 1
    assert "src/pkg/_gpu.py:9 changed since it was registered (sha256 None" in out
    assert "src/pkg/gone.py:1 changed since it was registered (sha256 None" in out


@pytest.mark.parametrize(
    ("evidence", "finding"),
    [
        ({"schema": "other.v1"}, "evidence record declares 'other.v1'"),
        ({"decisions": ["523df20b"]}, "evidence record does not cite the register's decisions"),
        ({"source_sha256": {}}, "engine/gpu.rs differs from the physical-GPU evidence"),
        ({"excepted_lines_covered": []}, "src/pkg/_gpu.py:4 is not shown run on a physical GPU"),
    ],
)
def test_evidence_that_does_not_describe_the_code_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], evidence: dict[str, Any], finding: str
) -> None:
    """The physical-GPU evidence must match the register's decisions, sources and lines.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the findings.
    evidence : dict
        The evidence fields to replace.
    finding : str
        The finding the gate must print.
    """
    _checkout(tmp_path, evidence=evidence)
    status, out = _run(tmp_path, _report(tmp_path, {"src/pkg/_gpu.py": _file(3, [4])}), capsys)
    assert status == 1
    assert finding in out


def test_an_unreadable_source_of_the_excepted_path_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A source the register names but the checkout lacks is reported, not skipped.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the findings.
    """
    _checkout(tmp_path, register={"source_set": [*SOURCES, "engine/missing.wgsl"]})
    status, out = _run(tmp_path, _report(tmp_path, {"src/pkg/_gpu.py": _file(3, [4])}), capsys)
    assert status == 1
    assert "engine/missing.wgsl of the excepted path's sources cannot be read" in out


@pytest.mark.parametrize(
    ("register", "report_text", "message"),
    [
        ({"schema": "other.v1"}, None, "register declares 'other.v1'"),
        ({"entries": [{"path": "x.py"}]}, None, "register entry {'path': 'x.py'} is malformed"),
        ({"source_set": []}, None, "register lacks 'source_set'"),
        ({}, '{"files": {}}', "coverage report has no measured files"),
        ({}, "[]", "is not a JSON object"),
        ({}, "{", "cannot be read"),
    ],
)
def test_a_malformed_input_exits_2_with_the_reason(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    register: dict[str, Any],
    report_text: str | None,
    message: str,
) -> None:
    """A malformed register or report is refused with exit status 2, never read as a pass.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the refusal.
    register : dict
        The register fields to replace.
    report_text : str or None
        The report's text, or None for a valid report.
    message : str
        The refusal the gate must print.
    """
    _checkout(tmp_path, register=register)
    report = _report(tmp_path, {"src/pkg/_gpu.py": _file(3, [4])})
    if report_text is not None:
        report.write_text(report_text, encoding="utf-8")
    status, out = _run(tmp_path, report, capsys)
    assert status == 2
    assert message in out


def test_a_missing_report_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A report path that does not exist is refused, not read as an empty union.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the refusal.
    """
    _checkout(tmp_path)
    status, out = _run(tmp_path, tmp_path / "absent.json", capsys)
    assert status == 2
    assert "coverage report" in out and "cannot be read" in out


def test_this_repository_register_and_evidence_describe_its_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real register and physical-GPU evidence agree with this checkout's GPU sources.

    A union that misses only the registered line passes the real gate, so the register's line
    hash, the evidence's source digests and its decisions all hold for the tracked code.

    Parameters
    ----------
    tmp_path : Path
        Holds the report.
    capsys : pytest.CaptureFixture[str]
        Captures the verdict.
    """
    register, entries = gate.load_register(gate.ROOT)
    files = {entry.path: _file(7, [entry.line]) for entry in entries}
    status, out = _run(gate.ROOT, _report(tmp_path, files), capsys)
    assert status == 0, out
    assert register["evidence"] == "tools/gpu_waveform_encoder_evidence.json"
    assert [entry.path for entry in entries] == ["src/sc_neurocore/spike_codec/_waveform_gpu.py"]


def test_the_gate_runs_as_a_script(tmp_path: Path) -> None:
    """``python tools/coverage_exception_gate.py`` exits with the gate's status.

    Parameters
    ----------
    tmp_path : Path
        Holds the report.
    """
    _, entries = gate.load_register(gate.ROOT)
    report = _report(tmp_path, {entry.path: _file(7, [entry.line]) for entry in entries})
    done = subprocess.run(
        [
            sys.executable,
            str(gate.ROOT / "tools/coverage_exception_gate.py"),
            "--coverage-json",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "pass" in done.stdout
