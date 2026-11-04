# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real synthesis toolchain custody and deadline tests

"""Exercise installed tools, their actual library data and bounded subprocesses."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_toolchain import SynthesisToolchain, SynthesisToolError


@pytest.fixture(scope="module")
def inspected(tmp_path_factory: pytest.TempPathFactory) -> tuple[SynthesisToolchain, Path]:
    """Inspect the installed toolchain once through its actual public entry point."""
    directory = tmp_path_factory.mktemp("synthesis-tools") / "receipt"
    return SynthesisToolchain.inspect(directory), directory


def test_manifest_binds_every_actual_executable_and_library(
    inspected: tuple[SynthesisToolchain, Path],
) -> None:
    """Independently recompute all executable and technology-library file bindings."""
    tools, _ = inspected
    tools.verify()
    expected: dict[str, str] = {}
    for name, path in tools.executables:
        with path.open("rb") as stream:
            expected[f"bin/{name}"] = hashlib.file_digest(stream, "sha256").hexdigest()
    for path in tools.data_directory.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                expected[f"data/{path.relative_to(tools.data_directory).as_posix()}"] = (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                )
    assert tools.manifest() == {"versions": dict(tools.versions), "sha256": expected}
    encoded = json.dumps(tools.manifest(), sort_keys=True, separators=(",", ":"))
    assert tools.identity == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    assert "/home/" not in encoded and "/media/" not in encoded
    sibling = tools.yosys.parent / "yosys-abc"
    with sibling.open("rb") as stream:
        assert expected["bin/abc"] == hashlib.file_digest(stream, "sha256").hexdigest()
    assert "data/xilinx/cells_sim.v" in expected


def test_actual_abc_runs_without_startup_files(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path
) -> None:
    """Use the bound ABC and retain the exact no-initialisation argument vector."""
    tools, _ = inspected
    log = tmp_path / "abc.log"
    tools.run([str(tools.abc), "-c", "version"], tmp_path, log, timeout=10)
    assert dict(tools.versions)["abc"] in log.read_text()
    argv: list[str] = json.loads(log.with_suffix(".command.json").read_text())
    position = argv.index(str(tools.abc))
    assert argv[position + 1 : position + 3] == ["-s", "-c"]
    assert argv[:3] == [str(tools.limits), "--core=0", "--"]
    assert "--signal=KILL" in argv


@pytest.mark.parametrize("variable", ["LD_PRELOAD", "LD_AUDIT", "LD_BIND_NOW"])
def test_loader_overrides_are_refused_before_native_execution(
    inspected: tuple[SynthesisToolchain, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    """Reject ambient native-code overrides before creating a command receipt or child."""
    tools, _ = inspected
    monkeypatch.setenv(variable, str(tmp_path / "unbound-library"))
    log = tmp_path / "refused-loader.log"
    with pytest.raises(ValueError, match="dynamic-loader environment overrides"):
        tools.run([str(tools.yosys), "-V"], tmp_path, log, timeout=10)
    assert not log.exists() and not log.with_suffix(".command.json").exists()


def test_actual_native_environment_excludes_unbound_ambient_settings(
    inspected: tuple[SynthesisToolchain, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect the environment of a real Yosys child without inheriting arbitrary settings."""
    tools, _ = inspected
    temporary = tmp_path / "confined-temporary"
    temporary.mkdir()
    monkeypatch.setenv("SCN_UNBOUND_NATIVE_SETTING", "must-not-reach-native")
    monkeypatch.setenv("TMPDIR", str(temporary))
    log = tmp_path / "native-environment.log"
    tools.run(
        [str(tools.yosys), "-Q", "-T", "-p", "exec -- /usr/bin/env"],
        tmp_path,
        log,
        timeout=10,
    )
    text = log.read_text()
    assert "SCN_UNBOUND_NATIVE_SETTING=" not in text
    assert f"TMPDIR={temporary}\n" in text
    assert f"HOME={tmp_path}\n" in text
    assert "LC_ALL=C\n" in text and "COLUMNS=32768\n" in text


def test_interpreter_library_path_is_accepted_but_never_reaches_native(
    inspected: tuple[SynthesisToolchain, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept a standard interpreter search path while a real child runs without it."""
    tools, _ = inspected
    library = tmp_path / "interpreter-library"
    library.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(library))
    log = tmp_path / "library-path-environment.log"
    tools.run(
        [str(tools.yosys), "-Q", "-T", "-p", "exec -- /usr/bin/env"],
        tmp_path,
        log,
        timeout=10,
    )
    text = log.read_text()
    assert "LD_LIBRARY_PATH=" not in text
    assert str(library) not in text
    assert f"HOME={tmp_path}\n" in text


@pytest.mark.parametrize("mode", ["interrupt", "stopped-deadline"])
def test_interrupt_or_stopped_deadline_reaps_actual_native_processes(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path, mode: str
) -> None:
    """Exercise real process-group cleanup in an isolated caller with an actual native child."""
    tools, _ = inspected
    script = """from contextlib import suppress
from pathlib import Path
import json
import os
import signal
import sys
import threading
import time
from tools.function_synthesis_toolchain import SynthesisToolchain, SynthesisToolError
root, mode = Path(sys.argv[1]), sys.argv[3]
tools = SynthesisToolchain.inspect(root / "inspection", yosys=sys.argv[2])
command = root / "command"
command.mkdir()
(command / "child.sh").write_text('printf "%s\\n" "$$" > native.pid\\nexec /usr/bin/sleep 60\\n')
ready = command / "native.pid"
owned = {}
done = threading.Event()

def operate_on_owned_processes():
    deadline = time.monotonic() + 10
    while not done.wait(0.005) and time.monotonic() < deadline:
        if ready.exists():
            text = ready.read_text().strip()
            if not text:
                continue
            pid = int(text)
            group = os.getpgid(pid)
            assert group != os.getpgrp() and os.getsid(pid) == group
            owned.update(pid=pid, group=group)
            if mode == "stopped-deadline":
                os.kill(group, signal.SIGSTOP)
            else:
                os.kill(os.getpid(), signal.SIGINT)
            return

worker = threading.Thread(target=operate_on_owned_processes, daemon=True)
worker.start()
began = time.monotonic()
try:
    try:
        tools.run([str(tools.yosys), "-Q", "-T", "-p", "exec -- /bin/sh child.sh"],
                  command, command / "native.log", timeout=1 if mode == "stopped-deadline" else 30)
    except KeyboardInterrupt:
        assert mode == "interrupt"
    except SynthesisToolError as error:
        assert mode == "stopped-deadline" and not error.timed_out and error.returncode != 0
    else:
        raise AssertionError("An interrupted native call completed successfully")
    assert owned and time.monotonic() - began < 10
    deadline = time.monotonic() + 5
    while Path('/proc/' + str(owned['pid'])).exists() and time.monotonic() < deadline:
        state = Path('/proc/' + str(owned['pid']) + '/stat')
        try:
            if state.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                break
        except FileNotFoundError:
            break
        time.sleep(0.005)
    remaining = Path('/proc/' + str(owned['pid']) + '/stat')
    assert not remaining.exists() or remaining.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    print(json.dumps({'mode': mode, 'observed_native_pid': owned['pid'], 'group': owned['group'],
                      'native_child_no_longer_running': True}))
finally:
    done.set()
    worker.join(timeout=2)
    if owned:
        with suppress(ProcessLookupError):
            os.killpg(owned['group'], signal.SIGKILL)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(tools.yosys), mode],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout)
    assert record["mode"] == mode and record["native_child_no_longer_running"] is True
    assert record["observed_native_pid"] > 0 and record["group"] != os.getpgrp()


def test_actual_yosys_abc_protocol_survives_narrow_columns_and_long_native_paths(
    inspected: tuple[SynthesisToolchain, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete real ABC mapping of public RTL when readline would truncate its command echo."""
    tools, _ = inspected
    directory = tmp_path / ("native-abc-protocol-" + "long-path-" * 12)
    temporary = directory / "native-temporary-data"
    temporary.mkdir(parents=True)
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("TMPDIR", str(temporary))
    (directory / "unit.v").write_text(
        emit_function_unit_verilog("heaviside", data_width=8, fraction=4, function_lowering="v2")
    )
    (directory / "abc-map.script").write_text("strash; map\n")
    (directory / "flow.ys").write_text(
        "read_verilog -sv unit.v\nhierarchy -check -top function_unit\nproc\nopt\n"
        "techmap\nopt_clean\nabc -g AND -script abc-map.script\n"
        "opt_clean\naigmap\nopt_clean\ncheck -assert\nwrite_aiger -symbols actual.aig\n"
    )
    log = directory / "native.log"
    tools.run([str(tools.yosys), "-Q", "-T", "flow.ys"], directory, log, timeout=15)
    assert "ABC RESULTS:" in log.read_text()
    with (directory / "actual.aig").open("rb") as stream:
        header = stream.readline().split()
    assert header[0] == b"aig" and tuple(int(header[i]) for i in (2, 3, 4)) == (16, 0, 8)


def test_actual_yosys_child_inherits_disabled_core_limit(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path
) -> None:
    """Observe the child core limit through a real Yosys exec command."""
    tools, _ = inspected
    log = tmp_path / "core-limit.log"
    tools.run(
        [str(tools.yosys), "-Q", "-T", "-p", "exec -- prlimit --core"],
        tmp_path,
        log,
        timeout=10,
    )
    assert any(
        line.startswith("CORE") and line.split()[-3:] == ["0", "0", "bytes"]
        for line in log.read_text().splitlines()
    )


def test_actual_native_failure_is_not_a_deadline(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path
) -> None:
    """Retain an actual rejected Yosys command without accepting its partial output."""
    tools, _ = inspected
    log = tmp_path / "failed.log"
    with pytest.raises(SynthesisToolError) as error:
        tools.run(
            [str(tools.yosys), "-Q", "-T", "-p", "unknown_function_synthesis_command"],
            tmp_path,
            log,
            timeout=10,
        )
    assert error.value.returncode != 0 and not error.value.timed_out
    assert "ERROR" in log.read_text()


def test_actual_elapsed_deadline_is_identified_by_its_diagnostic(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path
) -> None:
    """Kill a real native command at its deadline and retain the causal diagnostic."""
    tools, _ = inspected
    log = tmp_path / "deadline.log"
    with pytest.raises(SynthesisToolError) as error:
        tools.run(
            [str(tools.yosys), "-Q", "-T", "-p", "help"],
            tmp_path,
            log,
            timeout=0.000001,
        )
    assert error.value.returncode != 0 and error.value.timed_out
    assert "timeout: sending signal KILL to command" in log.read_text()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_deadlines_start_no_native_command(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path, timeout: float
) -> None:
    """Refuse disabled, negative, non-finite and boolean deadlines before execution."""
    tools, _ = inspected
    log = tmp_path / "invalid-timeout.log"
    with pytest.raises(ValueError, match="finite positive timeout"):
        tools.run([str(tools.yosys), "-V"], tmp_path, log, timeout=timeout)
    assert not log.exists() and not log.with_suffix(".command.json").exists()


@pytest.mark.parametrize("command", [[], ["/usr/bin/true"]])
def test_uninspected_executables_cannot_be_launched(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path, command: list[str]
) -> None:
    """Reject a missing or different executable without relying on ambient PATH."""
    tools, _ = inspected
    log = tmp_path / "uninspected.log"
    with pytest.raises(ValueError, match="Only inspected"):
        tools.run(command, tmp_path, log, timeout=10)
    assert not log.exists()


def test_changed_binding_refuses_a_real_command_before_execution(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path
) -> None:
    """Compare deliberately corrupted metadata with the real unchanged tool bytes."""
    tools, _ = inspected
    wrong = replace(tools, fingerprints=(("bin/yosys", "0" * 64), *tools.fingerprints[1:]))
    assert wrong.identity != tools.identity
    with pytest.raises(ValueError, match="changed after inspection"):
        wrong.run([str(tools.yosys), "-V"], tmp_path, tmp_path / "changed.log", timeout=10)
    assert not (tmp_path / "changed.log").exists()


@pytest.mark.parametrize("corruption", ["missing", "different", "ambiguous"])
def test_library_log_must_identify_the_inspected_data_root(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path, corruption: str
) -> None:
    """Refuse omitted, substituted or ambiguous paths in an actual producer log."""
    tools, receipts = inspected
    source = receipts / "yosys-library.log"
    tools.check_library_log(source)
    text = source.read_text()
    changed = text.replace("xilinx/cells_sim.v", "../other/xilinx/cells_sim.v")
    if corruption == "missing":
        changed = text.replace("xilinx/cells_sim.v", "xilinx/missing.v")
    elif corruption == "ambiguous":
        changed = text + changed
    log = tmp_path / "changed-library.log"
    log.write_text(changed)
    with pytest.raises(ValueError, match="library"):
        tools.check_library_log(log)


@pytest.mark.parametrize(
    ("tool", "error", "message"),
    [
        ("missing_function_synthesis_executable", FileNotFoundError, "unavailable"),
        (sys.executable, SynthesisToolError, "status"),
        ("test", SynthesisToolError, "status"),
        ("echo", ValueError, "native executable"),
        ("false", SynthesisToolError, "status"),
    ],
)
def test_inspection_refuses_missing_or_wrong_real_tools(
    tmp_path: Path, tool: str, error: type[Exception], message: str
) -> None:
    """Reject missing files and real non-Yosys executables before any qualification."""
    if tool in ("test", "echo", "false"):
        assert shutil.which(tool) is not None
    with pytest.raises(error, match=message):
        SynthesisToolchain.inspect(tmp_path / "invalid", yosys=tool)


def test_receipt_directory_is_never_overwritten(
    inspected: tuple[SynthesisToolchain, Path],
) -> None:
    """Preserve the actual inspection outputs when a caller reuses their directory."""
    _, directory = inspected
    before = (directory / "yosys-library.log").read_bytes()
    with pytest.raises(FileExistsError):
        SynthesisToolchain.inspect(directory)
    assert (directory / "yosys-library.log").read_bytes() == before


def test_native_command_receipts_are_not_overwritten(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path
) -> None:
    """Preserve previous native output and argv when a caller reuses their paths."""
    tools, _ = inspected
    log = tmp_path / "version.log"
    command = [str(tools.yosys), "-V"]
    tools.run(command, tmp_path, log, timeout=10)
    before = log.read_bytes(), log.with_suffix(".command.json").read_bytes()
    with pytest.raises(FileExistsError):
        tools.run(command, tmp_path, log, timeout=10)
    assert (log.read_bytes(), log.with_suffix(".command.json").read_bytes()) == before


def test_actual_systemverilog_library_read_matches_bound_data_root(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path
) -> None:
    """Identify the same actual library when the native parser reports SystemVerilog."""
    tools, _ = inspected
    log = tmp_path / "systemverilog-library.log"
    tools.run(
        [str(tools.yosys), "-Q", "-T", "-p", "read_verilog -sv -D EQUIV +/xilinx/cells_sim.v"],
        tmp_path,
        log,
        timeout=30,
    )
    assert "Parsing SystemVerilog input from" in log.read_text()
    tools.check_library_log(log)


def test_launcher_binds_the_native_binary_and_detects_its_replacement(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path
) -> None:
    """Follow a real launcher and reject changed native bytes behind unchanged scripts."""
    tools, _ = inspected
    native = tmp_path / "native" / "bin"
    native.mkdir(parents=True)
    copied = native / "yosys"
    shutil.copy2(tools.yosys, copied)
    (native / "yosys-abc").symlink_to(tools.abc)
    data = native.parent / "share"
    data.mkdir()
    (data / "yosys").symlink_to(tools.data_directory, target_is_directory=True)
    launchers = tmp_path / "launchers"
    launchers.mkdir()
    for name, binary in (("yosys", copied), ("yosys-abc", tools.abc)):
        launcher = launchers / name
        launcher.write_text(f'#!/bin/sh\nexec {shlex.quote(str(binary))} "$@"\n')
        launcher.chmod(0o755)
    wrapper = launchers / "yosys"
    before = wrapper.read_bytes()
    bound = SynthesisToolchain.inspect(tmp_path / "launched", yosys=str(wrapper))
    assert bound.yosys == copied.resolve() and bound.abc == tools.abc
    with copied.open("rb") as stream:
        assert (
            dict(bound.fingerprints)["bin/yosys"]
            == hashlib.file_digest(stream, "sha256").hexdigest()
        )
    assert bound.data_directory == tools.data_directory
    log = tmp_path / "launched-version.log"
    bound.run([str(bound.yosys), "-V"], tmp_path, log, timeout=10)
    assert dict(bound.versions)["yosys"] in log.read_text()
    with copied.open("ab") as stream:
        stream.write(b"\n")
    assert wrapper.read_bytes() == before
    with pytest.raises(ValueError, match="changed after inspection"):
        bound.run([str(bound.yosys), "-V"], tmp_path, tmp_path / "changed-native.log", timeout=10)
    assert not (tmp_path / "changed-native.log").exists()


@pytest.mark.parametrize("corruption", ["missing", "ambiguous", "empty-version"])
def test_inspection_refuses_invalid_launcher_discovery(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path, corruption: str
) -> None:
    """Reject missing native files, multiple paths and empty observed version output."""
    tools, _ = inspected
    launcher = tmp_path / "yosys"
    (tmp_path / "yosys-abc").symlink_to(tools.abc)
    reported = str(launcher)
    error: type[Exception] = ValueError
    message = "Empty version"
    if corruption == "missing":
        reported = str(tmp_path / "absent-yosys")
        error, message = FileNotFoundError, "unavailable"
    elif corruption == "ambiguous":
        reported += "\n" + str(tools.yosys)
        message = "native executable"
    launcher.write_text(
        '#!/bin/sh\nif [ "$1" = "--version" ]; then exit 0; fi\n'
        f"printf '%s\\n' {shlex.quote(reported)}\n"
    )
    launcher.chmod(0o755)
    with pytest.raises(error, match=message):
        SynthesisToolchain.inspect(tmp_path / "invalid-launcher", yosys=str(launcher))


@pytest.mark.parametrize("deadline", [0.0, float("nan"), float("inf")])
def test_shared_deadline_refuses_native_start_and_preserves_tool_identity(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path, deadline: float
) -> None:
    """Keep runtime deadline admission separate from actual executable and library provenance."""
    original, _ = inspected
    tools = replace(original, cohort_deadline=deadline)
    assert tools.identity == original.identity and tools.manifest() == original.manifest()
    log = tmp_path / "never-started.log"
    with pytest.raises(ValueError, match="shared proof cohort budget"):
        tools.run([str(tools.yosys), "-V"], tmp_path, log, timeout=1)
    assert not log.exists() and not log.with_suffix(".command.json").exists()


def test_real_final_tool_validation_cannot_overrun_the_shared_deadline(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path
) -> None:
    """Refuse successful native output when its final real identity read exhausts time."""
    original, _ = inspected
    started = time.monotonic()
    original.run([str(original.yosys), "-V"], tmp_path, tmp_path / "positive.log", timeout=10)
    allowance = max(2.0, (time.monotonic() - started) * 8)
    deadline = time.monotonic() + allowance
    tools = replace(original, cohort_deadline=deadline)
    native_started = False
    delayed = False

    def delay_final_identity_read(event: str, args: tuple[object, ...]) -> None:
        """Spend real elapsed time only in the post-command native identity validation."""
        nonlocal native_started, delayed
        if event == "subprocess.Popen" and args[2] in (tmp_path, str(tmp_path)):
            native_started = True
        if (
            event == "open"
            and args[0] in (tools.yosys, str(tools.yosys))
            and native_started
            and not delayed
        ):
            delayed = True
            time.sleep(max(0.0, deadline - time.monotonic()) + 0.01)

    delay_final_identity_read.__dict__["__cantrace__"] = True
    sys.addaudithook(delay_final_identity_read)
    log = tmp_path / "late-validation.log"
    with pytest.raises(ValueError, match="validation exhausted the shared proof cohort budget"):
        tools.run([str(tools.yosys), "-V"], tmp_path, log, timeout=1)
    assert native_started and delayed
    assert dict(original.versions)["yosys"] in log.read_text()
    assert log.with_suffix(".command.json").is_file()
    original.verify()


def test_actual_native_error_text_cannot_impersonate_a_timeout(
    inspected: tuple[SynthesisToolchain, Path], tmp_path: Path
) -> None:
    """Require deadline status as well as a matching line in real native process output."""
    tools, _ = inspected
    marker = "timeout: sending signal KILL to command native"
    log = tmp_path / "actual-native-error.log"
    with pytest.raises(SynthesisToolError) as error:
        tools.run(
            [
                str(tools.yosys),
                "-Q",
                "-T",
                "-p",
                f"log -stderr -nolog {marker}; unknown_native_command",
            ],
            tmp_path,
            log,
            timeout=10,
        )
    assert marker in log.read_text().splitlines()
    assert error.value.returncode == 1 and not error.value.timed_out
