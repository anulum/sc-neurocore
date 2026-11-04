# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real function measurement report contracts

"""Exercise reports through actual cache, mapping, proof and numerical measurements."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from filelock import FileLock

from sc_neurocore.compiler.lut_accuracy import measure
from tools.function_synthesis_cache import CachedFunctionSynthesis, FunctionSynthesisCache
from tools.function_synthesis_records import SynthesisCase, synthesis_cases
from tools.function_synthesis_report import build_function_synthesis_report
from tools.function_synthesis_toolchain import SynthesisToolchain

ReportFixture = tuple[FunctionSynthesisCache, CachedFunctionSynthesis]


@pytest.fixture(scope="module")
def report_cache(tmp_path_factory: pytest.TempPathFactory) -> ReportFixture:
    """Retain a real official-format floor mapping with all named-port proof artifacts."""
    directory = tmp_path_factory.mktemp("function-report")
    tools = SynthesisToolchain.inspect(directory / "inspection")
    cache = FunctionSynthesisCache(directory / "cache", tools)
    return cache, cache.measure(SynthesisCase("floor", 16, 8))


def _rows(report: dict[str, object]) -> list[dict[str, object]]:
    payload: object = json.loads(json.dumps(report, allow_nan=False))
    assert isinstance(payload, dict)
    rows: object = payload["cases"]
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) for row in rows)
    return [dict(row) for row in rows]


def test_real_diagnostic_keeps_original_counts_accuracy_and_time(
    report_cache: ReportFixture,
) -> None:
    """Combine historical proof with actual numerical assessment without claiming a new mapping."""
    cache, original = report_cache
    case = original.measurement.case
    report = build_function_synthesis_report(cache, cases=[case])
    row = _rows(report)[0]
    assert report["scope"] == "diagnostic" and report["status"] == "complete"
    assert report["matrix_complete"] is False
    assert row["hardware"] == original.measurement.to_dict()
    assert row["accuracy"] == measure("floor", 16, 8, lowering="v2").to_public_dict()
    assert row["recorded_at"] == original.recorded_at.isoformat()
    assert row["reused"] is True and row["assessment_basis"] == "selected_lowering"
    assert str(cache.directory) not in json.dumps(report)
    assert report["toolchain"] == cache.tools.manifest()
    assert isinstance(report["source_sha256"], dict)
    assert "src/sc_neurocore/neurons/equation_namespace.py" in report["source_sha256"]


def test_full_report_preserves_every_missing_or_unsupported_case(
    report_cache: ReportFixture,
) -> None:
    """Emit all 372 dispositions and refuse completion when 269 supported cases are unmeasured."""
    cache, _ = report_cache
    before = sorted(path.name for path in (cache.directory / "attempts").iterdir())
    report = build_function_synthesis_report(cache)
    rows = _rows(report)
    assert report["scope"] == "complete_matrix" and report["status"] == "incomplete"
    assert report["matrix_complete"] is False
    assert [row["case"] for row in rows] == [case.key for case in synthesis_cases()]
    assert report["counts"] == {
        "measured": 1,
        "unsupported": 102,
        "missing": 269,
        "timeout": 0,
        "tool_failure": 0,
        "writer_timeout": 0,
        "evidence_rejected": 0,
    }
    assert all(row["hardware"] is None for row in rows if row["status"] != "measured")
    assert before == sorted(path.name for path in (cache.directory / "attempts").iterdir())


@pytest.mark.parametrize("cases", [[], [SynthesisCase("floor", 8, 0)] * 2])
def test_invalid_selection_does_not_start_a_measurement(
    report_cache: ReportFixture, cases: list[SynthesisCase]
) -> None:
    """Reject empty or duplicate identities instead of producing misleading totals."""
    with pytest.raises(ValueError, match="nonempty unique"):
        build_function_synthesis_report(report_cache[0], cases=cases, measure_missing=True)


@pytest.mark.parametrize("function", ["trunc", "ceil", "floor", "sign"])
def test_legacy_exact_measurement_states_its_shared_accuracy_basis(
    report_cache: ReportFixture, tmp_path: Path, function: str
) -> None:
    """Map actual shared-exact legacy units and preserve the v2 numerical assessor identity."""
    cache = FunctionSynthesisCache(tmp_path / "cache", report_cache[0].tools)
    case = SynthesisCase(function, 8, 4, "lut-v1")
    report = build_function_synthesis_report(cache, cases=[case], measure_missing=True)
    row = _rows(report)[0]
    assert row["status"] == "measured" and row["reused"] is False
    assert row["assessment_basis"] == "shared_exact_arithmetic"
    assert row["accuracy"] == measure(function, 8, 4, lowering="v2").to_public_dict()
    assert isinstance(row["hardware"], dict)
    assert row["hardware"]["lowering"] == "lut-v1"
    assert row["hardware"]["declared_table_bits"] == 0
    again = build_function_synthesis_report(cache, cases=[case])
    assert _rows(again)[0]["reused"] is True
    assert _rows(again)[0]["recorded_at"] == row["recorded_at"]


def test_real_legacy_table_keeps_its_own_error_domain(
    report_cache: ReportFixture, tmp_path: Path
) -> None:
    """Report the actual legacy table's numerical errors separately from mapped resource counts."""
    cache = FunctionSynthesisCache(tmp_path / "cache", report_cache[0].tools)
    case = SynthesisCase("exp", 8, 4, "lut-v1")
    report = build_function_synthesis_report(cache, cases=[case], measure_missing=True)
    row = _rows(report)[0]
    assert row["status"] == "measured" and row["assessment_basis"] == "selected_lowering"
    assert row["accuracy"] == measure("exp", 8, 4, lowering="lut-v1").to_public_dict()
    assert isinstance(row["hardware"], dict)
    assert row["hardware"]["declared_table_bits"] == 256 * 8


def test_real_binary_report_keeps_complete_operand_domain(
    report_cache: ReportFixture, tmp_path: Path
) -> None:
    """Retain the independent second port and full binary error-domain metadata."""
    cache = FunctionSynthesisCache(tmp_path / "cache", report_cache[0].tools)
    case = SynthesisCase("heaviside", 8, 4)
    report = build_function_synthesis_report(cache, cases=[case], measure_missing=True)
    row = _rows(report)[0]
    assert row["status"] == "measured"
    assert isinstance(row["hardware"], dict) and isinstance(row["accuracy"], dict)
    assert row["hardware"]["proof"] == {
        "status": "equivalent",
        "method": "complete-aiger",
        "input_bits": 16,
        "output_bits": 8,
        "latches": 0,
    }
    assert row["accuracy"]["input_arity"] == 2
    assert row["accuracy"]["domain_cardinality"] == 1 << 16
    ranges = row["accuracy"]["operand_ranges"]
    assert isinstance(ranges, list) and len(ranges) == 2


def test_damaged_cache_is_unavailable_and_never_zero_cost(
    report_cache: ReportFixture, tmp_path: Path
) -> None:
    """Retain a real corrupted mapping and explicitly refuse its hardware/accuracy evidence."""
    original = report_cache[1]
    cache = FunctionSynthesisCache(tmp_path / "cache", report_cache[0].tools)
    entry = cache.directory / "successful" / original.identity
    shutil.copytree(original.directory, entry)
    evidence = entry / ".retained-artifacts/equivalence.log.gz"
    damaged = b"Networks are NOT EQUIVALENT.\n"
    assert evidence.is_file()
    evidence.write_bytes(damaged)
    report = build_function_synthesis_report(cache, cases=[original.measurement.case])
    row = _rows(report)[0]
    assert report["status"] == "incomplete" and row["status"] == "evidence_rejected"
    assert row["hardware"] is None and row["accuracy"] is None
    assert evidence.read_bytes() == damaged


def test_real_native_deadline_retains_causal_attempt_hashes(
    report_cache: ReportFixture, tmp_path: Path
) -> None:
    """Report explicit native deadline evidence without converting a failed attempt to success."""
    cache = FunctionSynthesisCache(tmp_path / "cache", report_cache[0].tools)
    report = build_function_synthesis_report(
        cache, cases=[SynthesisCase("floor", 8, 0)], measure_missing=True, synthesis_timeout=1e-9
    )
    row = _rows(report)[0]
    assert report["status"] == "incomplete" and row["status"] == "timeout"
    assert row["returncode"] != 0 and row["hardware"] is None
    attempt = next((cache.directory / "attempts").iterdir())
    assert row["attempt_artifact_sha256"] == {
        path.relative_to(attempt).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in attempt.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("damage", ["linked-directory", "fifo-artifact"])
def test_failed_native_attempt_with_unsafe_custody_is_explicitly_rejected(
    tmp_path: Path, damage: str
) -> None:
    """Refuse real linked or irregular failed-attempt evidence through the public report."""
    script = """from pathlib import Path
import json
import os
import sys

source, root = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(source / "src"), str(source)]
from tools.function_synthesis_cache import FunctionSynthesisCache
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_report import build_function_synthesis_report
from tools.function_synthesis_toolchain import SynthesisToolchain

tools = SynthesisToolchain.inspect(root / "inspection")
cache = FunctionSynthesisCache(root / "cache", tools)
changed = False
held = root / "preserved-attempt"
mode = sys.argv[3]

def change_owned_attempt(event, args):
    global changed
    if changed or event != "subprocess.Popen" or args[2] is None:
        return
    directory = Path(args[2])
    if directory.name != "measurement" or directory.parent.parent != cache.directory / "attempts":
        return
    changed = True
    attempt = directory.parent
    if mode == "linked-directory":
        attempt.rename(held)
        attempt.symlink_to(held, target_is_directory=True)
    else:
        os.mkfifo(attempt / "irregular-evidence")

sys.addaudithook(change_owned_attempt)
report = build_function_synthesis_report(
    cache, cases=[SynthesisCase("floor", 8, 0)],
    measure_missing=True, synthesis_timeout=1e-9,
)
assert changed
assert report["status"] == "incomplete"
assert report["counts"]["evidence_rejected"] == 1
row = report["cases"][0]
assert row["status"] == "evidence_rejected"
assert row["hardware"] is None and row["accuracy"] is None
attempt = next((cache.directory / "attempts").iterdir())
if mode == "linked-directory":
    assert attempt.is_symlink() and (held / "request.json").is_file()
else:
    assert (attempt / "irregular-evidence").is_fifo()
print(json.dumps({"changed": changed, "status": row["status"], "mode": mode}))
"""
    source = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(source), str(tmp_path), damage],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload == {"changed": True, "status": "evidence_rejected", "mode": damage}


def test_deeply_nested_retained_manifest_is_an_explicit_rejected_row(
    report_cache: ReportFixture, tmp_path: Path
) -> None:
    """Return unavailable evidence when a real copied cache exceeds JSON recursion depth."""
    cache = FunctionSynthesisCache(tmp_path / "cache", report_cache[0].tools)
    original = report_cache[1]
    entry = cache.directory / "successful" / original.identity
    shutil.copytree(original.directory, entry)
    manifest = entry / "retained-artifacts.json"
    damaged = b"[" * (sys.getrecursionlimit() * 2) + b"0" + b"]" * (sys.getrecursionlimit() * 2)
    manifest.write_bytes(damaged)
    before = hashlib.sha256(manifest.read_bytes()).hexdigest(), manifest.stat().st_mtime_ns
    report = build_function_synthesis_report(cache, cases=[original.measurement.case])
    row = _rows(report)[0]
    assert report["status"] == "incomplete" and row["status"] == "evidence_rejected"
    assert row["hardware"] is None and row["accuracy"] is None
    assert (
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        manifest.stat().st_mtime_ns,
    ) == before


def test_real_cached_read_audit_error_produces_a_rejected_report_row(
    report_cache: ReportFixture, tmp_path: Path
) -> None:
    """Convert an unexpected real artifact-read failure into explicit unavailable report evidence."""
    cache, original = report_cache
    manifest = original.directory / "retained-artifacts.json"
    before = manifest.read_bytes(), manifest.stat().st_mtime_ns
    script = """from pathlib import Path
import json
import sys
from tools.function_synthesis_cache import FunctionSynthesisCache
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_report import build_function_synthesis_report
from tools.function_synthesis_toolchain import SynthesisToolchain
root, directory, watched = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tools = SynthesisToolchain.inspect(root / "inspection", yosys=sys.argv[4])
cache = FunctionSynthesisCache(directory, tools)
def reject_actual_read(event, args):
    if event == "open" and args[0] in (watched, str(watched)):
        raise ArithmeticError("private artifact audit details")
sys.addaudithook(reject_actual_read)
report = build_function_synthesis_report(cache, cases=[SynthesisCase("floor", 16, 8)])
row = report["cases"][0]
assert report["status"] == "incomplete" and row["status"] == "evidence_rejected"
assert row["hardware"] is None and row["accuracy"] is None
print(json.dumps(report, allow_nan=False))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            str(cache.directory),
            str(manifest),
            str(cache.tools.yosys),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _rows(json.loads(result.stdout))[0]["status"] == "evidence_rejected"
    assert "private artifact audit details" not in result.stdout + result.stderr
    assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before


def test_actual_writer_lease_is_reported_without_reusing_around_it(
    report_cache: ReportFixture,
) -> None:
    """Report a finite real writer wait while preserving the already qualified measurement."""
    cache, original = report_cache
    with FileLock(str(cache.directory / "locks" / f"{original.identity}.lock")):
        report = build_function_synthesis_report(
            cache, cases=[original.measurement.case], measure_missing=True, lock_timeout=0.001
        )
    assert _rows(report)[0]["status"] == "writer_timeout"
    assert report["status"] == "incomplete"
    assert cache.get(original.measurement.case) is not None


def test_actual_killed_native_process_is_not_reported_as_a_deadline(
    report_cache: ReportFixture, tmp_path: Path
) -> None:
    """Terminate only the test's own native mapper and distinguish its signal from GNU timeout."""
    cache = FunctionSynthesisCache(tmp_path / "cache", report_cache[0].tools)
    halted = threading.Event()
    killed: list[int] = []

    def stop_owned_mapper() -> None:
        """Find a Yosys child in this request's exact working directory and send SIGTERM."""
        while not halted.wait(0.002):
            for attempt in (cache.directory / "attempts").glob("*/measurement"):
                for process in Path("/proc").iterdir():
                    if not process.name.isdigit():
                        continue
                    try:
                        if (process / "cwd").resolve() != attempt:
                            continue
                        if (process / "exe").resolve() != cache.tools.yosys:
                            continue
                        os.kill(int(process.name), signal.SIGTERM)
                        killed.append(int(process.name))
                        halted.set()
                        break
                    except (FileNotFoundError, PermissionError, ProcessLookupError):
                        continue

    watchdog = threading.Timer(15, halted.set)
    watchdog.daemon = True
    worker = threading.Thread(target=stop_owned_mapper, daemon=True)
    watchdog.start()
    worker.start()
    try:
        report = build_function_synthesis_report(
            cache, cases=[SynthesisCase("sin", 8, 4)], measure_missing=True
        )
    finally:
        watchdog.cancel()
        halted.set()
        worker.join(timeout=2)
        watchdog.join(timeout=2)
    assert len(killed) == 1 and not worker.is_alive() and not watchdog.is_alive()
    row = _rows(report)[0]
    assert row["status"] == "tool_failure" and row["returncode"] != 0
    assert row["hardware"] is None and report["status"] == "incomplete"


def test_source_change_during_real_mapping_refuses_the_complete_report(tmp_path: Path) -> None:
    """Refuse changed report provenance after an actual successful native mapping."""
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
        shutil.copyfile(root / name, destination)
    script = tmp_path / "report-source-change.py"
    script.write_text(
        """from pathlib import Path
import sys
import threading
import time

root = Path(sys.argv[1])
sys.path[:0] = [str(root / "src"), str(root)]
from tools.function_synthesis_cache import FunctionSynthesisCache
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_report import build_function_synthesis_report
from tools.function_synthesis_toolchain import SynthesisToolchain

tools = SynthesisToolchain.inspect(root / "inspection")
cache = FunctionSynthesisCache(root / "cache", tools)
changed = threading.Event()

def change_report_source() -> None:
    '''Change only the copied report dependency after its native mapper starts.'''
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if list((root / "cache/attempts").glob("*/measurement/yosys.command.json")):
            source = root / "src/sc_neurocore/neurons/equation_namespace.py"
            with source.open("a") as stream:
                stream.write("\\n")
            changed.set()
            return
        time.sleep(0.002)

worker = threading.Thread(target=change_report_source, daemon=True)
worker.start()
try:
    build_function_synthesis_report(
        cache, cases=[SynthesisCase("floor", 8, 0)], measure_missing=True
    )
except ValueError as error:
    assert str(error) == "Report sources changed during numerical assessment"
else:
    raise AssertionError("A report with changed source provenance was returned")
worker.join(timeout=2)
assert changed.is_set() and not worker.is_alive()
entries = list((cache.directory / "successful").iterdir())
assert len(entries) == 1
from tools.function_synthesis_retention import materialize_artifacts
with materialize_artifacts(entries[0]) as proof:
    assert "Networks are equivalent" in (proof / "equivalence.log").read_text()
print("Report source change refused after actual successful native mapping")
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script), str(checkout)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "source change refused after actual successful native mapping" in result.stdout
    dependency = Path("src/sc_neurocore/neurons/equation_namespace.py")
    assert (checkout / dependency).read_bytes() == (root / dependency).read_bytes() + b"\n"
    report_source = Path("tools/function_synthesis_report.py")
    assert (checkout / report_source).read_bytes() == (root / report_source).read_bytes()
