# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native partition acquisition limits and concurrent corruption

"""Exercise smaller admitted budgets and changes at actual native artifact boundaries."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread

import pytest

from tests.test_tools.test_function_synthesis_bdd import NativeProof
from tests.test_tools.test_function_synthesis_bdd import native_proof as native_proof
from tests.test_tools.test_function_synthesis_partition import NativePartition
from tests.test_tools.test_function_synthesis_partition import native_partition as native_partition
from tools.function_synthesis_bdd import prove_complete_bdd_mapping, read_complete_bdd_mapping
from tools.function_synthesis_partition import prove_partitioned_bdd, read_partitioned_bdd
from tools.function_synthesis_toolchain import SynthesisToolchain, SynthesisToolError
from tools.function_synthesis_yosys import read_function_synthesis_measurement

NativeFault = tuple[Path, SynthesisToolchain, int, str, list[Path], list[Thread]]


@pytest.mark.parametrize("budget", [False, 0.0, -1.0, 3301.0, float("nan"), float("inf")])
def test_partition_refuses_invalid_whole_cohort_allowances(
    native_partition: NativePartition, budget: float
) -> None:
    """Refuse relaxed or nonfinite cohort budgets before touching an original acquisition."""
    directory, tools, width = native_partition
    path = directory / "input-partition/partition-proof.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="cohort allowance"):
        prove_partitioned_bdd(directory, tools, width, cohort_seconds=budget)
    assert path.read_bytes() == before


def test_partition_reserves_a_complete_native_command_before_starting(
    native_partition: NativePartition, tmp_path: Path
) -> None:
    """Leave an incomplete acquisition when no native command fits the real remaining budget."""
    original, tools, width = native_partition
    directory = tmp_path / "short-budget"
    directory.mkdir()
    source = directory / "singleton.aig"
    shutil.copyfile(original / "singleton.aig", source)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="partition is incomplete"):
        prove_partitioned_bdd(directory, tools, width, cohort_seconds=0.000001)
    assert source.read_bytes() == before
    assert list((directory / "input-partition").iterdir()) == []


@pytest.mark.parametrize(
    "kind",
    ["source", "input-order", "cohort-reservation", "cohort-completion", "cohort-validation"],
)
def test_partition_acquisition_checks_actual_io_changes_and_elapsed_time(
    native_partition: NativePartition, tmp_path: Path, kind: str
) -> None:
    """Alter real native artifacts or spend real time without replacing a tool or clock."""
    original, tools, width = native_partition
    directory = tmp_path / "acquisition"
    directory.mkdir()
    cohort, timeout = (
        (5.0, 1.0) if kind in {"cohort-completion", "cohort-validation"} else (95.0, 90.0)
    )
    if kind == "cohort-reservation":
        baseline = read_partitioned_bdd(original, tools, width)
        cohort = timeout + 2.0 * float(str(baseline["elapsed_seconds"]))
        assert cohort <= 3300.0
    if kind in {"source", "cohort-completion", "cohort-validation"}:
        shutil.copyfile(original / "reference.aig", directory / "reference.aig")
        tools.run(
            [
                str(tools.abc),
                "-c",
                "&read reference.aig; &miter reference.aig; &put; orpos; &get; &write singleton.aig",
            ],
            directory,
            directory / "identity.log",
            timeout=90.0,
        )
    else:
        shutil.copyfile(original / "singleton.aig", directory / "singleton.aig")
    entry = {
        "source": "node-000/cofactor.aig",
        "input-order": "node-001/cofactor.log",
        "cohort-reservation": "node-004/cofactor.aig",
        "cohort-completion": "node-000/cofactor.aig",
        "cohort-validation": "node-000/cofactor.log",
    }[kind]
    target = directory / "input-partition" / entry
    changed = False
    reads = 0

    def change_at_native_read(event: str, args: tuple[object, ...]) -> None:
        """Change a completed native artifact or consume the real remaining wall time."""
        nonlocal changed, reads
        if event == "open" and args[0] == str(target) and args[1] == "r" and not changed:
            if kind == "cohort-validation":
                reads += 1
                if reads == 1:
                    return
            changed = True
            if kind == "source":
                path = directory / "singleton.aig"
                path.write_bytes(
                    path.read_bytes() + b"\noriginal miter changed during acquisition\n"
                )
            elif kind == "input-order":
                target.write_text(target.read_text().replace("pi00", "changed_original_input"))
            else:
                began = time.monotonic()
                boundary = cohort - (timeout if kind == "cohort-reservation" else 0.0)
                delay = max(0.0, boundary - (began - started)) + 0.1
                time.sleep(delay)
                assert time.monotonic() - started > boundary
                if kind == "cohort-validation":
                    (directory / "validation-wait.json").write_text(
                        json.dumps({"elapsed_seconds": time.monotonic() - began})
                    )

    change_at_native_read.__dict__["__cantrace__"] = True
    sys.addaudithook(change_at_native_read)
    started = time.monotonic()
    with pytest.raises(ValueError, match="changed|partition is incomplete|invalid time, budget"):
        prove_partitioned_bdd(directory, tools, width, timeout=timeout, cohort_seconds=cohort)
    assert changed
    assert not (directory / "input-partition/partition-proof.json").exists()


def test_partition_readback_refuses_a_changed_leaf_bdd_command(
    native_partition: NativePartition, tmp_path: Path
) -> None:
    """Reject a different native leaf command before accepting its otherwise intact zero graph."""
    original, tools, width = native_partition
    directory = tmp_path / "changed-command"
    shutil.copytree(original, directory)
    path = next((directory / "input-partition").glob("node-*/bdd.command.json"))
    command = json.loads(path.read_text())
    command[-1] += "; print_io"
    path.write_text(json.dumps(command, indent=2) + "\n")
    with pytest.raises(ValueError, match="original complete cofactor"):
        read_partitioned_bdd(directory, tools, width)


@pytest.mark.parametrize("budget", [False, 0.0, -1.0, 3301.0, float("nan"), float("inf")])
def test_whole_function_rejects_invalid_cohort_allowances(
    native_proof: NativeProof, budget: float
) -> None:
    """Refuse relaxed whole-function deadlines without altering the original proof record."""
    case, root, tools = native_proof
    record = root / "complete-bdd-proof/proof-record.json"
    before = record.read_bytes()
    with pytest.raises(ValueError, match="cohort allowance"):
        prove_complete_bdd_mapping(case, root, tools, cohort_seconds=budget)
    assert record.read_bytes() == before


def test_whole_function_reserves_both_original_exports(
    native_proof: NativeProof, tmp_path: Path
) -> None:
    """Leave the measurement untouched when its remaining budget cannot fit the original exports."""
    case, root, tools = native_proof
    changed = tmp_path / "insufficient-export-budget"
    shutil.copytree(root, changed, ignore=shutil.ignore_patterns("complete-bdd-proof"))
    before = {p: p.read_bytes() for p in changed.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="BDD cohort is incomplete"):
        prove_complete_bdd_mapping(case, changed, tools, cohort_seconds=0.000001)
    assert not (changed / "complete-bdd-proof").exists()
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("completion", [False, True])
def test_whole_function_charges_real_native_io_time_to_its_cohort(
    native_proof: NativeProof, tmp_path: Path, completion: bool
) -> None:
    """Spend actual wall time between native steps or at completion and require no proof credit."""
    case, root, tools = native_proof
    warmup = tmp_path / "positive-whole-cohort"
    shutil.copytree(root, warmup, ignore=shutil.ignore_patterns("complete-bdd-proof"))
    started = time.monotonic()
    prove_complete_bdd_mapping(case, warmup, tools, timeout=90.0)
    elapsed = time.monotonic() - started
    native_seconds = min(90.0, max(5.0, elapsed * 2))
    cohort_seconds = min(3300.0, max(20.0, elapsed * 4 + native_seconds))
    changed = tmp_path / "exhausted-whole-cohort"
    shutil.copytree(root, changed, ignore=shutil.ignore_patterns("complete-bdd-proof"))
    target = (
        changed
        / "complete-bdd-proof"
        / ("complete-bdd.log" if completion else "candidate-io.command.json")
    )
    observed = False

    def spend_actual_native_io_time(event: str, args: tuple[object, ...]) -> None:
        """Delay one actual production I/O boundary without changing its bytes or clock."""
        nonlocal observed
        if (
            event == "open"
            and args[0] == str(target)
            and args[1] == ("r" if completion else "x")
            and not observed
        ):
            observed = True
            time.sleep(cohort_seconds + 0.1)

    spend_actual_native_io_time.__dict__["__cantrace__"] = True
    sys.addaudithook(spend_actual_native_io_time)
    with pytest.raises(ValueError, match="cohort is incomplete|implementation changed"):
        prove_complete_bdd_mapping(
            case, changed, tools, timeout=native_seconds, cohort_seconds=cohort_seconds
        )
    assert observed
    assert not (changed / "complete-bdd-proof/proof-record.json").exists()


@pytest.fixture(params=["timeout", "signal", "all-timeout"])
def native_leaf_fault(
    native_partition: NativePartition, tmp_path: Path, request: pytest.FixtureRequest
) -> NativeFault:
    """Block an actual ABC input with a FIFO and retain its real deadline or signal outcome."""
    original, tools, width = native_partition
    directory = tmp_path / "native-leaf-fault"
    directory.mkdir()
    mode = str(request.param)
    if mode == "all-timeout":
        (directory / "fault.v").write_text(
            f"module original_miter(input [{width - 1}:0] argument0, output difference);\n"
            "assign difference = 1'b1;\nendmodule\n"
        )
        tools.run(
            [
                str(tools.yosys),
                "-Q",
                "-T",
                "-p",
                "read_verilog fault.v; hierarchy -check -top original_miter; aigmap; write_aiger -symbols singleton.aig",
            ],
            directory,
            directory / "fault-export.log",
            timeout=90.0,
        )
    else:
        shutil.copyfile(original / "singleton.aig", directory / "singleton.aig")
    blocked: list[Path] = []
    writers: list[Thread] = []

    def terminate_actual_native_process(graph: Path) -> None:
        """Wait for the real ABC reader, then kill only its inspected native process."""
        with graph.open("wb", buffering=0):
            children = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split()
            assert len(children) == 1
            wrapper = int(children[0])
            assert Path(f"/proc/{wrapper}/exe").resolve() == tools.deadline
            native = Path(f"/proc/{wrapper}/task/{wrapper}/children").read_text().split()
            assert len(native) == 1
            pid = int(native[0])
            assert Path(f"/proc/{pid}/exe").resolve() == tools.abc
            os.kill(pid, signal.SIGKILL)

    def block_actual_native_leaf(event: str, args: tuple[object, ...]) -> None:
        """Restore the original graph after a real failed command, before production readback."""
        if event != "open" or not isinstance(args[0], str):
            return
        path = Path(args[0])
        if path.parent.parent != directory / "input-partition":
            return
        graph = path.with_name("cofactor.aig")
        saved = path.with_name("cofactor-before-fault.aig")
        if path.name == "bdd.command.json" and args[1] == "x":
            if mode == "all-timeout" or path.parent.name == "node-004":
                graph.rename(saved)
                os.mkfifo(graph)
                blocked.append(path.parent)
                if mode == "signal":
                    writer = Thread(
                        target=terminate_actual_native_process, args=(graph,), daemon=True
                    )
                    writers.append(writer)
                    writer.start()
        elif path.name == "bdd.log" and args[1] == "r" and graph.is_fifo():
            graph.unlink()
            saved.rename(graph)

    block_actual_native_leaf.__dict__["__cantrace__"] = True
    sys.addaudithook(block_actual_native_leaf)
    return directory, tools, width, mode, blocked, writers


def test_partition_handles_real_native_deadlines_and_process_signals(
    native_leaf_fault: NativeFault,
) -> None:
    """Complete a genuine retry or refuse an unproved assignment without replacing native tools."""
    directory, tools, width, mode, blocked, writers = native_leaf_fault
    began = time.monotonic()
    if mode == "signal":
        with pytest.raises(SynthesisToolError) as error:
            prove_partitioned_bdd(directory, tools, width, timeout=1.0, cohort_seconds=180.0)
        assert not error.value.timed_out
        assert not (directory / "input-partition/partition-proof.json").exists()
    elif mode == "timeout":
        prove_partitioned_bdd(directory, tools, width, timeout=1.0, cohort_seconds=180.0)
        record = read_partitioned_bdd(directory, tools, width)
        assert record["covered_assignments"] == 1 << width
        nodes = record["nodes"]
        assert isinstance(nodes, list)
        assert any(row["bdd_returncode"] in {-9, 124, 137} for row in nodes)
    else:
        with pytest.raises(ValueError, match="assignment has no complete zero proof"):
            prove_partitioned_bdd(directory, tools, width, timeout=1.0, cohort_seconds=180.0)
        path = directory / "input-partition/partition-proof.json"
        assert not path.exists()
        # This intentionally incomplete payload is a negative reader fixture, never proof credit.
        rows = [
            {
                "prefix": "0" * index,
                "bdd_status": "not-attempted" if index < 4 else "incomplete",
                "bdd_returncode": 0,
            }
            for index in range(width + 1)
        ]
        path.write_text(
            json.dumps(
                {
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "native_seconds": 1.0,
                    "cohort_seconds": 180.0,
                    "elapsed_seconds": time.monotonic() - began,
                    "nodes": rows,
                }
            )
        )
        with pytest.raises(ValueError, match="assignment has no complete zero proof"):
            read_partitioned_bdd(directory, tools, width)
    assert blocked
    assert all(not (node / "cofactor.aig").is_fifo() for node in blocked)
    for writer in writers:
        writer.join(timeout=5.0)
        assert not writer.is_alive()


@pytest.mark.parametrize("damage", ["none", "global-attempt", "strategy", "signal"])
def test_whole_bdd_completes_a_real_global_deadline_with_partitioned_readback(
    native_proof: NativeProof, tmp_path: Path, damage: str
) -> None:
    """Prove a genuine global-timeout retry and refuse altered native proof provenance."""
    case, original, tools = native_proof
    root = tmp_path / "global-timeout"
    shutil.copytree(original, root, ignore=shutil.ignore_patterns("complete-bdd-proof"))
    directory = root / "complete-bdd-proof"
    graph = directory / "singleton.aig"
    saved = directory / "singleton-before-timeout.aig"
    command = directory / "complete-bdd.command.json"
    log = directory / "complete-bdd.log"
    blocked = False
    writers: list[Thread] = []
    killed: list[int] = []

    def terminate_actual_global_reader() -> None:
        """Kill only the inspected ABC child after its actual FIFO input handshake."""
        with graph.open("wb", buffering=0):
            children = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split()
            assert len(children) == 1
            wrapper = int(children[0])
            assert Path(f"/proc/{wrapper}/exe").resolve() == tools.deadline
            native = Path(f"/proc/{wrapper}/task/{wrapper}/children").read_text().split()
            assert len(native) == 1
            pid = int(native[0])
            assert Path(f"/proc/{pid}/exe").resolve() == tools.abc
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)

    def block_original_global_input(event: str, args: tuple[object, ...]) -> None:
        """Restore the real input after GNU timeout terminates its blocked ABC reader."""
        nonlocal blocked
        if event != "open":
            return
        if args[0] == str(command) and args[1] == "x":
            graph.rename(saved)
            os.mkfifo(graph)
            blocked = True
            if damage == "signal":
                writer = Thread(target=terminate_actual_global_reader, daemon=True)
                writers.append(writer)
                writer.start()
        elif args[0] == str(log) and args[1] == "r" and graph.is_fifo():
            graph.unlink()
            saved.rename(graph)

    block_original_global_input.__dict__["__cantrace__"] = True
    sys.addaudithook(block_original_global_input)
    if damage == "signal":
        with pytest.raises(SynthesisToolError) as error:
            prove_complete_bdd_mapping(case, root, tools, timeout=5.0, cohort_seconds=180.0)
        for writer in writers:
            writer.join(timeout=5.0)
            assert not writer.is_alive()
        assert killed and blocked and graph.is_file() and not saved.exists()
        assert not error.value.timed_out
        assert not (directory / "proof-record.json").exists()
        return
    prove_complete_bdd_mapping(case, root, tools, timeout=5.0, cohort_seconds=180.0)
    assert blocked and graph.is_file() and not saved.exists()
    assert "timeout: sending signal KILL to command" in log.read_text()
    partition = read_partitioned_bdd(directory, tools, case.input_bits)
    assert partition["covered_assignments"] == 1 << case.input_bits
    files = [path for path in root.rglob("*") if path.is_file()]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
    assert read_complete_bdd_mapping(case, root, tools).method == "complete-bdd-shannon"
    assert (
        read_function_synthesis_measurement(case, root, tools).proof_method
        == "complete-bdd-shannon"
    )
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
    if damage == "global-attempt":
        log.write_text(
            log.read_text().replace("timeout: sending signal KILL to command", "removed")
        )
        with pytest.raises(ValueError, match="requires its original bounded global attempt"):
            read_complete_bdd_mapping(case, root, tools)
    elif damage == "strategy":
        path = directory / "proof-record.json"
        record = json.loads(path.read_text())
        record["strategy"] = "unqualified-method"
        path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        with pytest.raises(ValueError, match="explicit native proof method"):
            read_complete_bdd_mapping(case, root, tools)
