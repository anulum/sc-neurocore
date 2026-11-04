# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native checkpoint export contracts

"""Exercise actual public native exports and refuse changed retained acquisitions."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.function_synthesis_checkpoints import export_checkpoint_aigers, read_checkpoint_aigers
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import run_function_synthesis

NativeExports = tuple[SynthesisCase, Path, SynthesisToolchain]


@pytest.fixture(scope="module")
def native_exports(tmp_path_factory: pytest.TempPathFactory) -> NativeExports:
    """Acquire a real source-bound function through the public mapping and export APIs."""
    root = tmp_path_factory.mktemp("complete-native-checkpoints")
    tools = SynthesisToolchain.inspect(root / "inspection")
    case = SynthesisCase("floor", 8, 0)
    mapping = root / "measurement"
    run_function_synthesis(case, mapping, tools)
    export_checkpoint_aigers(case, mapping, tools, mapping / "checkpoints")
    return case, mapping, tools


def test_complete_native_checkpoint_readback_preserves_originals(
    native_exports: NativeExports,
) -> None:
    """Validate actual native checkpoints without changing any acquisition byte or time."""
    case, root, tools = native_exports
    files = [p for p in root.rglob("*") if p.is_file()]
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in files}
    read_checkpoint_aigers(case, root, tools, root / "checkpoints")
    assert before == {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in files
    }
    record = json.loads((root / "checkpoints/checkpoint-exports.json").read_text())
    assert record["original_input_bits"] == record["original_output_bits"] == 8
    assert set(record["checkpoints"]) == {"reference", "mapped"}
    assert record["schema_version"] == "sc-neurocore.complete-function-checkpoints.v2"
    assert set(record["prepared_checkpoints"]) == {"reference", "mapped"}
    for role in ("reference", "mapped"):
        assert (root / "checkpoints" / f"{role}.prepared.il.gz").is_file()
        assert (root / "checkpoints" / f"{role}-map.command.json").is_file()


def test_binary_native_checkpoint_exports_keep_both_independent_operands(
    native_exports: NativeExports,
    tmp_path: Path,
) -> None:
    """Export an actual two-argument function without tying its original inputs."""
    _, _, tools = native_exports
    case = SynthesisCase("heaviside", 8, 4)
    root = tmp_path / "binary"
    run_function_synthesis(case, root, tools)
    export_checkpoint_aigers(case, root, tools, root / "checkpoints")
    read_checkpoint_aigers(case, root, tools, root / "checkpoints")
    record = json.loads((root / "checkpoints/checkpoint-exports.json").read_text())
    assert record["original_input_bits"] == 16
    assert record["original_output_bits"] == 8


@pytest.mark.parametrize("budget", [False, 0.0, -1.0, 91.0, float("nan"), float("inf")])
def test_checkpoint_native_deadline_is_not_relaxed(
    native_exports: NativeExports,
    tmp_path: Path,
    budget: float,
) -> None:
    """Refuse invalid command budgets before creating another native acquisition."""
    case, root, tools = native_exports
    target = tmp_path / "refused"
    with pytest.raises(ValueError):
        export_checkpoint_aigers(case, root, tools, target, timeout=budget)
    assert not target.exists()


@pytest.mark.parametrize("budget", [False, 0.0, -1.0, 3301.0, float("nan"), float("inf")])
def test_checkpoint_cohort_allowance_is_finite_before_any_export(
    native_exports: NativeExports, tmp_path: Path, budget: float
) -> None:
    """Refuse invalid whole-export budgets while preserving the complete original acquisition."""
    case, root, tools = native_exports
    target = tmp_path / "refused"
    with pytest.raises(ValueError, match="finite cohort allowance"):
        export_checkpoint_aigers(case, root, tools, target, cohort_seconds=budget)
    assert not target.exists()


@pytest.mark.parametrize("stage", ["next-command", "validation", "readback"])
def test_real_export_io_delay_cannot_publish_a_within_budget_success(
    native_exports: NativeExports, tmp_path: Path, stage: str
) -> None:
    """Count actual command/validation IO delay against the original export cohort allowance."""
    case, original, tools = native_exports
    assert case == SynthesisCase("floor", 8, 0)
    script = """from pathlib import Path
import sys
import time
from tools.function_synthesis_checkpoints import export_checkpoint_aigers
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
original, root, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[4]
tools = SynthesisToolchain.inspect(root / "inspection", yosys=sys.argv[3])
case = SynthesisCase("floor", 8, 0)
began = time.monotonic()
export_checkpoint_aigers(case, original, tools, root / "calibration", timeout=5.0)
cohort = 5.0 + 2.0 * (time.monotonic() - began)
assert cohort <= 3300.0
target = root / "exports"
filename = "reference-finish.command.json" if stage == "next-command" else "mapped-finish.command.json"
seen = 0
waited = 0.0
delay = 0.0

def delay_actual_io(event, args):
    global seen, waited, delay
    if event == "open" and args[0] in (str(target / filename), target / filename):
        seen += 1
        selected = (
            stage == "next-command" and seen == 1
            or stage != "next-command" and args[1] == "r"
            and (target / "checkpoint-exports.json").is_file() == (stage == "readback")
        )
        if not waited and selected:
            limit = cohort - (5.0 if stage == "next-command" else 0.0)
            delay = max(0.0, limit - (time.monotonic() - started)) + 0.1
            began = time.monotonic()
            time.sleep(delay)
            waited = time.monotonic() - began

sys.addaudithook(delay_actual_io)
started = time.monotonic()
try:
    export_checkpoint_aigers(case, original, tools, target, timeout=5.0, cohort_seconds=cohort)
except ValueError as error:
    assert "checkpoint cohort is incomplete" in str(error), str(error)
else:
    raise AssertionError("An export with exceeded or unreserved cohort time succeeded")
assert seen >= 1 and waited >= delay > 0.0
assert time.monotonic() - started > cohort - (5.0 if stage == "next-command" else 0.0)
assert (target / "reference-prepare.log").is_file()
if stage == "next-command":
    assert (target / "reference-finish.log").is_file()
    assert not (target / "mapped-prepare.command.json").exists()
assert (target / "checkpoint-exports.json").exists() == (stage == "readback")
print("Actual export cohort refusal:", stage)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(original), str(tmp_path), str(tools.yosys), stage],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Actual export cohort refusal: {stage}" in result.stdout


@pytest.mark.parametrize(
    "filename",
    [
        "unit.v",
        "physical-module.json",
        "reference-prepare.ys",
        "reference-map.ys",
        "reference.prepared.il.gz",
        "mapped-finish.ys",
        "reference.aig",
        "reference-prepare.command.json",
        "mapped-map.command.json",
        "checkpoint-exports.json",
    ],
)
def test_changed_native_checkpoint_evidence_is_refused(
    native_exports: NativeExports,
    tmp_path: Path,
    filename: str,
) -> None:
    """Reject corruption of actual source, graph, script, interface and command artifacts."""
    case, root, tools = native_exports
    copied = tmp_path / "measurement"
    shutil.copytree(root, copied)
    path = copied / "checkpoints" / filename
    path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises((OSError, ValueError)):
        read_checkpoint_aigers(case, copied, tools, copied / "checkpoints")


@pytest.mark.parametrize("kind", ["empty", "truncated", "checksum", "expansion"])
def test_native_checkpoint_stream_integrity_and_size_are_checked(
    native_exports: NativeExports,
    tmp_path: Path,
    kind: str,
) -> None:
    """Reject incomplete native streams and real oversized decoded checkpoint data."""
    case, root, tools = native_exports
    copied = tmp_path / "measurement"
    shutil.copytree(root, copied)
    path = copied / "checkpoints/reference.il.gz"
    if kind == "empty":
        path.write_bytes(gzip.compress(b""))
    elif kind == "truncated":
        path.write_bytes(path.read_bytes()[:-5])
    elif kind == "checksum":
        raw = bytearray(path.read_bytes())
        raw[-8] ^= 1
        path.write_bytes(raw)
    else:
        with gzip.open(path, "wb", compresslevel=1) as stream:
            chunk = b"0" * (1024 * 1024)
            for _ in range(1024):
                stream.write(chunk)
            stream.write(b"0")
    with pytest.raises(ValueError):
        read_checkpoint_aigers(case, copied, tools, copied / "checkpoints")


@pytest.mark.parametrize("entry", ["unit.v", "checkpoints", "checkpoints/checkpoint-exports.json"])
def test_native_checkpoint_readback_never_follows_links(
    native_exports: NativeExports,
    tmp_path: Path,
    entry: str,
) -> None:
    """Reject actual retained inputs or directories replaced by symbolic links."""
    case, root, tools = native_exports
    copied = tmp_path / "measurement"
    shutil.copytree(root, copied)
    path = copied / entry
    retained = tmp_path / "original"
    path.rename(retained)
    path.symlink_to(retained, target_is_directory=retained.is_dir())
    with pytest.raises(ValueError):
        read_checkpoint_aigers(case, copied, tools, copied / "checkpoints")


@pytest.mark.parametrize("entry", ["unit.v", "flow.ys", "mapped.json"])
def test_original_mapping_changes_invalidate_checkpoint_exports(
    native_exports: NativeExports,
    tmp_path: Path,
    entry: str,
) -> None:
    """Bind checkpoint readback to the whole original physical acquisition."""
    case, root, tools = native_exports
    copied = tmp_path / "measurement"
    shutil.copytree(root, copied)
    path = copied / entry
    path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError):
        read_checkpoint_aigers(case, copied, tools, copied / "checkpoints")


def test_native_export_does_not_overwrite_an_existing_acquisition(
    native_exports: NativeExports,
) -> None:
    """Keep the actual completed source-bound exports when acquisition is repeated."""
    case, root, tools = native_exports
    record = root / "checkpoints/checkpoint-exports.json"
    before = record.read_bytes()
    with pytest.raises(FileExistsError):
        export_checkpoint_aigers(case, root, tools, root / "checkpoints")
    assert record.read_bytes() == before


@pytest.mark.parametrize(
    "mode", ["object", "nonstring", "length", "executable", "deadline", "nonfinite"]
)
def test_retained_native_command_identity_and_deadline_are_enforced(
    native_exports: NativeExports,
    tmp_path: Path,
    mode: str,
) -> None:
    """Refuse altered native executable receipts and expanded or malformed time limits."""
    case, root, tools = native_exports
    copied = tmp_path / "measurement"
    shutil.copytree(root, copied)
    path = copied / "checkpoints/reference-prepare.command.json"
    argv = json.loads(path.read_text())
    if mode == "object":
        payload: object = {}
    elif mode == "nonstring":
        argv[-1] = None
        payload = argv
    elif mode == "length":
        payload = argv[:-1]
    elif mode == "executable":
        argv[7] = "/different/yosys"
        payload = argv
    else:
        argv[6] = "91" if mode == "deadline" else "nan"
        payload = argv
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError):
        read_checkpoint_aigers(case, copied, tools, copied / "checkpoints")


@pytest.mark.parametrize("stamp", [None, "2026-11-01T01:00:00+01:00", "2026-11-01T00:00:00Z"])
def test_checkpoint_record_keeps_its_original_canonical_time(
    native_exports: NativeExports,
    tmp_path: Path,
    stamp: str | None,
) -> None:
    """Reject absent acquisition times and noncanonical timezone representations."""
    case, root, tools = native_exports
    copied = tmp_path / "measurement"
    shutil.copytree(root, copied)
    path = copied / "checkpoints/checkpoint-exports.json"
    payload = json.loads(path.read_text())
    payload["recorded_at"] = stamp
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError):
        read_checkpoint_aigers(case, copied, tools, copied / "checkpoints")


def test_checkpoint_record_cannot_rebind_to_a_linked_original_acquisition(
    native_exports: NativeExports,
    tmp_path: Path,
) -> None:
    """Require the actual retained original mapping directory rather than a linked alias."""
    case, root, tools = native_exports
    link = tmp_path / "linked-original"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        read_checkpoint_aigers(case, link, tools, root / "checkpoints")


@pytest.mark.parametrize("existing", [False, True])
def test_original_mapping_is_frozen_during_actual_native_export(
    native_exports: NativeExports,
    tmp_path: Path,
    existing: bool,
) -> None:
    """Reject an actual source-acquisition change while the public native exporter runs."""
    import threading
    import time

    case, root, tools = native_exports
    copied = tmp_path / "measurement"
    shutil.copytree(root, copied)
    target = copied / "new-checkpoints"
    if existing:
        target.mkdir()
    stop = threading.Event()
    changed = threading.Event()

    def change_original() -> None:
        """Alter the copied mapping only after the actual native command begins."""
        deadline = time.monotonic() + 30
        while not stop.is_set() and time.monotonic() < deadline:
            if (target / "reference-prepare.command.json").exists():
                path = copied / "mapped.v"
                path.write_bytes(path.read_bytes() + b"\n// changed during export\n")
                changed.set()
                return
            stop.wait(0.01)

    thread = threading.Thread(target=change_original)
    thread.start()
    try:
        error = FileExistsError if existing else ValueError
        message = None if existing else "changed during export"
        with pytest.raises(error, match=message):
            export_checkpoint_aigers(case, copied, tools, target)
    finally:
        stop.set()
        thread.join(5)
    assert changed.is_set() != existing and not thread.is_alive()
    assert not (target / "checkpoint-exports.json").exists()
