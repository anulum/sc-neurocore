# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual function measurement CLI contracts

"""Exercise command outputs and exit status using the installed native synthesis tools."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.function_synthesis import main
from tools.function_synthesis_cache import FunctionSynthesisCache
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain


@pytest.fixture(scope="module")
def command_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Populate a diagnostic cache through the actual command's complete native path."""
    root = tmp_path_factory.mktemp("function-command")
    output = root / "original.json"
    cache = root / "cache"
    assert (
        main(
            [
                "--cache",
                str(cache),
                "--output",
                str(output),
                "measure",
                "floor",
                "--data-width",
                "8",
                "--fraction",
                "0",
            ]
        )
        == 0
    )
    return cache


def _projection(text: str) -> dict[str, object]:
    payload: object = json.loads(text)
    assert isinstance(payload, dict)
    return dict(payload)


@pytest.mark.parametrize(
    "error_name", ["RecursionError", "ArithmeticError", "KeyError", "TypeError"]
)
def test_actual_io_audit_failure_has_distinct_internal_error_exit(
    tmp_path: Path, error_name: str
) -> None:
    """Keep unexpected real directory-audit failures distinct from incomplete JSON reports."""
    root = Path(__file__).resolve().parents[2]
    cache, output = tmp_path / "cache", tmp_path / "report.json"
    script = """import builtins
from pathlib import Path
import sys
from tools.function_synthesis import main

cache, output = Path(sys.argv[1]), Path(sys.argv[2])
error = getattr(builtins, sys.argv[3])
def reject_inspection(event, args):
    if event == "os.mkdir" and str(args[0]).startswith(str(cache / "inspections")):
        raise error("private audit failure details")
sys.addaudithook(reject_inspection)
raise SystemExit(main(["--cache", str(cache), "--output", str(output), "report"]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(cache), str(output), error_name],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": f"{root / 'src'}:{root}:{os.environ.get('PYTHONPATH', '')}",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 3 and not result.stdout
    assert "failed unexpectedly" in result.stderr
    assert "private audit failure details" not in result.stderr
    assert "Traceback" not in result.stderr and not output.exists()
    assert not (cache / "attempts").exists()


def test_command_reuses_success_and_preserves_original_time(
    command_cache: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Read the exact earlier success without modifying any native source/proof receipt."""
    before = {
        str(path.relative_to(command_cache)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in (command_cache / "successful").rglob("*")
        if path.is_file()
    }
    assert (
        main(
            [
                "--cache",
                str(command_cache),
                "measure",
                "floor",
                "--data-width",
                "8",
                "--fraction",
                "0",
            ]
        )
        == 0
    )
    report = _projection(capsys.readouterr().out)
    original = _projection((command_cache.parent / "original.json").read_text())
    rows = report["cases"]
    first_rows = original["cases"]
    assert isinstance(rows, list) and isinstance(first_rows, list)
    assert rows[0]["recorded_at"] == first_rows[0]["recorded_at"]
    assert rows[0]["reused"] is True and report["matrix_complete"] is False
    assert (
        report["command_source_sha256"]
        == hashlib.sha256(
            (Path(__file__).resolve().parents[2] / "tools/function_synthesis.py").read_bytes()
        ).hexdigest()
    )
    assert before == {
        str(path.relative_to(command_cache)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in (command_cache / "successful").rglob("*")
        if path.is_file()
    }


def test_module_entry_from_outside_checkout_uses_the_real_cache(
    command_cache: Path, tmp_path: Path
) -> None:
    """Invoke python -m from another directory with the actual owning compiler/tooling import path."""
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.function_synthesis",
            "--cache",
            str(command_cache),
            "--output",
            str(output),
            "measure",
            "floor",
            "--data-width",
            "8",
            "--fraction",
            "0",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": f"{root / 'src'}:{root}:{os.environ.get('PYTHONPATH', '')}",
        },
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout
    report = _projection(output.read_text())
    assert report["status"] == "complete" and report["scope"] == "diagnostic"


def test_incomplete_matrix_returns_failure_with_every_disposition(
    command_cache: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse full-matrix success for a valid diagnostic outside the six official formats."""
    attempts = sorted(path.name for path in (command_cache / "attempts").iterdir())
    assert main(["--cache", str(command_cache), "report"]) == 1
    report = _projection(capsys.readouterr().out)
    assert report["status"] == "incomplete" and report["matrix_complete"] is False
    assert report["counts"] == {
        "measured": 0,
        "unsupported": 102,
        "missing": 270,
        "timeout": 0,
        "tool_failure": 0,
        "writer_timeout": 0,
        "evidence_rejected": 0,
    }
    rows = report["cases"]
    assert isinstance(rows, list) and len(rows) == 372
    assert attempts == sorted(path.name for path in (command_cache / "attempts").iterdir())


def test_unsupported_legacy_diagnostic_never_returns_measurement_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep an unsupported legacy function explicit and return nonzero without mapping it."""
    cache = tmp_path / "cache"
    assert main(["--cache", str(cache), "measure", "tan", "--lowering", "lut-v1"]) == 1
    report = _projection(capsys.readouterr().out)
    rows = report["cases"]
    assert isinstance(rows, list) and rows[0]["status"] == "unsupported"
    assert rows[0]["hardware"] is None
    assert not (cache / "attempts").exists()


def test_native_deadline_returns_nonzero_and_retains_attempt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Emit a real inconclusive diagnostic and keep its original GNU deadline receipts."""
    cache = tmp_path / "cache"
    assert (
        main(
            [
                "--cache",
                str(cache),
                "--synthesis-timeout",
                "1e-9",
                "measure",
                "floor",
                "--data-width",
                "8",
                "--fraction",
                "0",
            ]
        )
        == 1
    )
    report = _projection(capsys.readouterr().out)
    rows = report["cases"]
    assert isinstance(rows, list) and rows[0]["status"] == "timeout"
    assert (
        "timeout: sending signal KILL"
        in next((cache / "attempts").glob("*/measurement/yosys.log")).read_text()
    )


@pytest.mark.parametrize("option", ["--synthesis-timeout", "--proof-timeout", "--lock-timeout"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_invalid_deadlines_are_rejected_before_creating_evidence(
    tmp_path: Path, option: str, value: str
) -> None:
    """Refuse all invalid parsed budgets before inspection, source import work or native execution."""
    cache = tmp_path / "cache"
    with pytest.raises(SystemExit) as error:
        main(["--cache", str(cache), option, value, "measure", "floor"])
    assert error.value.code == 2 and not cache.exists()


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_link"])
def test_existing_output_is_preserved_before_any_native_inspection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    """Refuse regular, directory and dangling-link destinations without altering owner data."""
    output = tmp_path / "existing"
    if kind == "file":
        output.write_text("Owner evidence\n")
    elif kind == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "missing")
    cache = tmp_path / "cache"
    assert main(["--cache", str(cache), "--output", str(output), "report"]) == 2
    assert not cache.exists()
    assert "refused" in capsys.readouterr().err
    if kind == "file":
        assert output.read_text() == "Owner evidence\n"
    elif kind == "directory":
        assert output.is_dir()
    else:
        assert output.is_symlink()


def test_invalid_format_is_refused_before_inspection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep an invalid numerical format from becoming a native request or output file."""
    cache = tmp_path / "cache"
    assert (
        main(["--cache", str(cache), "measure", "floor", "--data-width", "8", "--fraction", "8"])
        == 2
    )
    assert not cache.exists() and "refused" in capsys.readouterr().err


def test_missing_output_parent_does_not_discard_valid_evidence(
    command_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep admitted native evidence when the requested new report cannot be written."""
    output = tmp_path / "missing/report.json"
    assert (
        main(
            [
                "--cache",
                str(command_cache),
                "--output",
                str(output),
                "measure",
                "floor",
                "--data-width",
                "8",
                "--fraction",
                "0",
            ]
        )
        == 2
    )
    assert not output.parent.exists()
    assert "refused" in capsys.readouterr().err
    assert len(list((command_cache / "successful").iterdir())) == 1


def test_actual_missing_tool_is_a_setup_failure(tmp_path: Path) -> None:
    """Run the real entry point with no executable search path and return a setup error."""
    root = Path(__file__).resolve().parents[2]
    cache = tmp_path / "cache"
    result = subprocess.run(
        [sys.executable, "-m", "tools.function_synthesis", "--cache", str(cache), "report"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": f"{root / 'src'}:{root}:{os.environ.get('PYTHONPATH', '')}",
            "PATH": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2 and "refused" in result.stderr and not result.stdout
    assert not cache.exists()


@pytest.mark.parametrize(
    "changed", ["tools/function_synthesis.py", "tools/function_synthesis_report.py"]
)
def test_actual_command_or_report_source_change_refuses_publication(
    tmp_path: Path, changed: str
) -> None:
    """Change an actual copied orchestration source during native execution and refuse its report."""
    root = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "source-copy"
    sources = {
        path.relative_to(root).as_posix() for path in (root / "src/sc_neurocore").rglob("*.py")
    }
    sources.update(
        path.relative_to(root).as_posix()
        for path in (root / "tools").glob("function_synthesis*.py")
    )
    for name in sorted(sources):
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / name).read_bytes())
    before = hashlib.sha256((root / changed).read_bytes()).hexdigest()
    script = tmp_path / "changed-orchestration.py"
    script.write_text(
        """from pathlib import Path
import sys
import threading
import time

root = Path(sys.argv[1])
sys.path[:0] = [str(root / "src"), str(root)]
from tools.function_synthesis import main
changed = threading.Event()

def change_source() -> None:
    \"\"\"Alter only the isolated source copy after the actual native command starts.\"\"\"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if list((root / "cache/attempts").glob("*/measurement/yosys.command.json")):
            with (root / sys.argv[2]).open("a") as stream:
                stream.write("\\n# Actual source change after native execution began.\\n")
            changed.set()
            return
        time.sleep(0.002)

worker = threading.Thread(target=change_source, daemon=True)
worker.start()
output = root / "report.json"
status = main(["--cache", str(root / "cache"), "--output", str(output),
               "measure", "floor", "--data-width", "8", "--fraction", "0"])
worker.join(timeout=2)
assert changed.is_set() and status == 2
assert not output.exists()
entries = list((root / "cache/successful").iterdir())
assert len(entries) == 1
from tools.function_synthesis_retention import materialize_artifacts
with materialize_artifacts(entries[0]) as proof:
    assert "Networks are equivalent" in (proof / "equivalence.log").read_text()
print("Orchestration source change refused after successful actual native proof")
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script), str(checkout), changed],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "source change refused" in result.stdout
    assert hashlib.sha256((root / changed).read_bytes()).hexdigest() == before
    assert ast.dump(ast.parse((checkout / changed).read_text())) == ast.dump(
        ast.parse((root / changed).read_text())
    )


@pytest.mark.parametrize("failure", ["receipt", "native-command"])
def test_complete_bdd_fallback_survives_native_component_refusal_and_cli_cache_reuse(
    tmp_path: Path, failure: str
) -> None:
    """Prove a real DSP function independently after a failed component, then restore it through CLI and cache."""
    cache = tmp_path / "cache"
    output = tmp_path / "original.json"
    changed = False
    preserved: list[Path] = []

    def damage_component(event: str, args: tuple[object, ...]) -> None:
        """Retain and corrupt one actual component artifact at its native I/O boundary."""
        nonlocal changed
        if event == "open" and args[1] == "r" and failure == "receipt" and not changed:
            path = Path(str(args[0]))
            if path.name == "pre-dsp-preparation.log" and cache in path.parents:
                changed = True
                original = path.read_text()
                retained = path.with_suffix(".native.log")
                retained.write_text(original)
                preserved.append(retained)
                assert "/xilinx/cells_sim.v" in original
                path.write_text(
                    original.replace(
                        "/xilinx/cells_sim.v", "/uninspected-library/xilinx/cells_sim.v"
                    )
                )
        elif event == "subprocess.Popen" and failure == "native-command" and not changed:
            directory = Path(str(args[2]))
            if directory.name == "composition-proof" and cache in directory.parents:
                changed = True
                path = directory / "pre-dsp.ys"
                native_script = path.read_bytes()
                retained = path.with_suffix(".native.ys")
                retained.write_bytes(native_script)
                preserved.append(retained)
                path.write_bytes(native_script + b"\ninvalid_native_component_command\n")

    tools = SynthesisToolchain.inspect(tmp_path / "inspection")
    damage_component.__dict__["__cantrace__"] = True
    sys.addaudithook(damage_component)
    assert (
        main(
            [
                "--cache",
                str(cache),
                "--output",
                str(output),
                "measure",
                "sin",
                "--data-width",
                "8",
                "--fraction",
                "4",
            ]
        )
        == 0
    )
    original_report = json.loads(output.read_text())
    row = original_report["cases"][0]
    assert changed and len(preserved) == 1
    assert row["status"] == "measured" and not row["reused"]
    assert row["hardware"]["proof"]["method"] == "complete-bdd"
    assert row["hardware"]["resources"]["dsp_blocks"] == 18
    original = FunctionSynthesisCache(cache, tools).get(SynthesisCase("sin", 8, 4))
    assert original is not None and original.reused
    assert original.measurement.proof_method == "complete-bdd"
    manifest = json.loads((original.directory / "retained-artifacts.json").read_text())
    assert "complete-bdd-proof/proof-record.json" in manifest["artifacts"]
    assert "composition-proof/pre-dsp-preparation.log" in manifest["artifacts"]
    assert preserved[0].relative_to(preserved[0].parents[1]).as_posix() in manifest["artifacts"]
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in original.directory.rglob("*")
        if path.is_file()
    }
    readback = tmp_path / "readback.json"
    root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.function_synthesis",
            "--cache",
            str(cache),
            "--output",
            str(readback),
            "measure",
            "sin",
            "--data-width",
            "8",
            "--fraction",
            "4",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": f"{root / 'src'}:{root}:{os.environ.get('PYTHONPATH', '')}",
        },
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    reused = json.loads(readback.read_text())["cases"][0]
    assert reused["reused"] and reused["hardware"] == row["hardware"]
    assert reused["recorded_at"] == row["recorded_at"]
    assert before == {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in original.directory.rglob("*")
        if path.is_file()
    }
