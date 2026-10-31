# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded cargo-fuzz runner

"""The cargo-fuzz runner bounds every build and run and leaves no compiler behind."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest


def _load_tool() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "security_scan" / "run_cargo_fuzz_scanners.py"
    spec = importlib.util.spec_from_file_location("run_cargo_fuzz_scanners", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest_tool() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "security_scanner_manifest.py"
    spec = importlib.util.spec_from_file_location("security_scanner_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_cargo_fuzz_command_uses_packet_runner() -> None:
    """The scanner manifest runs cargo-fuzz through this packet runner with a pinned version."""
    manifest_tool = _load_manifest_tool()
    manifest = manifest_tool.build_scanner_manifest()
    scanners = {scanner["name"]: scanner for scanner in manifest["scanners"]}

    assert scanners["cargo-fuzz-nightly"]["command"] == (
        "python tools/security_scan/run_cargo_fuzz_scanners.py "
        "--output-dir security/ci-security-packet --target all --max-total-time 300"
    )
    assert scanners["cargo-fuzz-nightly"]["pinned_version"] == "cargo-fuzz==0.13.2"


def test_discovers_fuzz_targets_from_cargo_manifest() -> None:
    """Both fuzz targets declared in ``fuzz/Cargo.toml`` are discovered."""
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]

    targets = tool.discover_fuzz_targets(repo_root)

    assert targets == ["bitstream_ops", "ir_parser"]


def test_runner_executes_each_target_with_bounded_time_and_writes_summary(tmp_path: Path) -> None:
    """Each target is built under the build budget, then run under its share of the run budget."""
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == repo_root
        assert capture_output is True
        assert text is True
        assert check is False
        assert timeout in {320, 3600}
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    summary = tool.run_cargo_fuzz_scanners(
        repo_root=repo_root,
        output_dir=tmp_path / "packet",
        selected_targets=("bitstream_ops", "ir_parser"),
        max_total_time=40,
        run_command=fake_run,
    )

    assert [call[:5] for call in calls] == [
        ["cargo", "+nightly", "fuzz", "build", "bitstream_ops"],
        ["cargo", "+nightly", "fuzz", "run", "bitstream_ops"],
        ["cargo", "+nightly", "fuzz", "build", "ir_parser"],
        ["cargo", "+nightly", "fuzz", "run", "ir_parser"],
    ]
    assert all("--fuzz-dir" in call for call in calls)
    run_calls = [call for call in calls if call[3] == "run"]
    assert all("-max_total_time=20" in call for call in run_calls)
    assert all("-runs=0" not in call for call in calls)
    assert summary["passed"] is True
    assert summary["target_count"] == 2
    assert (
        json.loads(
            (tmp_path / "packet" / "security" / "cargo_fuzz_summary.json").read_text(
                encoding="utf-8"
            )
        )["passed"]
        is True
    )


def test_runner_rejects_unknown_target(tmp_path: Path) -> None:
    """An unknown target name is refused before anything is built."""
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]

    try:
        tool.run_cargo_fuzz_scanners(
            repo_root=repo_root,
            output_dir=tmp_path / "packet",
            selected_targets=("missing",),
            max_total_time=10,
        )
    except ValueError as exc:
        assert "unknown fuzz targets" in str(exc)
    else:
        raise AssertionError("unknown target was accepted")


def test_runner_reports_target_failures_without_dropping_artifacts(tmp_path: Path) -> None:
    """A crashing target fails the summary and keeps its own report."""
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, timeout, check
        if command[3] == "build":
            return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="crash")

    summary = tool.run_cargo_fuzz_scanners(
        repo_root=repo_root,
        output_dir=tmp_path / "packet",
        selected_targets=("ir_parser",),
        max_total_time=5,
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["failed_targets"] == ["ir_parser"]
    report = json.loads(
        (tmp_path / "packet" / "security" / "cargo_fuzz_ir_parser.json").read_text(encoding="utf-8")
    )
    assert report["phase"] == "run"
    assert report["build_returncode"] == 0
    assert report["stderr_tail"] == ["crash"]


def test_runner_records_timeout_as_target_failure(tmp_path: Path) -> None:
    """A run that exceeds its budget is recorded as a failure with exit status 124."""
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check
        if command[3] == "build":
            return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")
        raise subprocess.TimeoutExpired(command, timeout, output="partial", stderr="still running")

    summary = tool.run_cargo_fuzz_scanners(
        repo_root=repo_root,
        output_dir=tmp_path / "packet",
        selected_targets=("ir_parser",),
        max_total_time=5,
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["failed_targets"] == ["ir_parser"]
    report = json.loads(
        (tmp_path / "packet" / "security" / "cargo_fuzz_ir_parser.json").read_text(encoding="utf-8")
    )
    assert report["phase"] == "run"
    assert report["returncode"] == 124
    assert report["stdout_tail"] == ["partial"]
    assert report["stderr_tail"] == ["still running", "command timed out after 305 seconds"]


def test_runner_records_build_timeout_before_fuzz_execution(tmp_path: Path) -> None:
    """A build that exceeds its budget fails the target without running it."""
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check
        calls.append(command)
        raise subprocess.TimeoutExpired(command, timeout, output="partial", stderr="compiling")

    summary = tool.run_cargo_fuzz_scanners(
        repo_root=repo_root,
        output_dir=tmp_path / "packet",
        selected_targets=("ir_parser",),
        max_total_time=5,
        build_timeout=7,
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["failed_targets"] == ["ir_parser"]
    assert [call[3] for call in calls] == ["build"]
    report = json.loads(
        (tmp_path / "packet" / "security" / "cargo_fuzz_ir_parser.json").read_text(encoding="utf-8")
    )
    assert report["phase"] == "build"
    assert report["build_returncode"] == 124
    assert report["build_stdout_tail"] == ["partial"]
    assert report["build_stderr_tail"] == ["compiling", "command timed out after 7 seconds"]


def _process_is_gone(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    try:
        state = stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return True
    return state == "Z"


def test_default_build_budget_covers_a_slow_instrumented_engine_build() -> None:
    """The default build budget exceeds the slowest measured hosted build of 30 minutes."""
    tool = _load_tool()

    assert tool.CARGO_FUZZ_BUILD_TIMEOUT_SECONDS == 3600
    assert tool.build_parser().parse_args(["--output-dir", "packet"]).build_timeout == 3600
    assert tool.run_cargo_fuzz_scanners.__kwdefaults__["run_command"] is tool.run_in_process_group


def test_process_group_runner_returns_the_real_output_and_status(tmp_path: Path) -> None:
    """A finished command returns its own exit status and captured text."""
    tool = _load_tool()

    result = tool.run_in_process_group(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 3
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


@pytest.mark.parametrize(
    "options",
    [
        {"capture_output": False, "text": True, "check": False},
        {"capture_output": True, "text": False, "check": False},
        {"capture_output": True, "text": True, "check": True},
    ],
)
def test_process_group_runner_refuses_modes_it_does_not_implement(
    tmp_path: Path, options: dict[str, bool]
) -> None:
    """Only captured text without an exit-status check is supported."""
    tool = _load_tool()

    with pytest.raises(ValueError, match="captures text output and never checks"):
        tool.run_in_process_group(["true"], cwd=tmp_path, timeout=60, **options)


def test_timeout_kills_every_process_the_command_started(tmp_path: Path) -> None:
    """A timed-out command leaves no child behind to hold the cargo build lock."""
    tool = _load_tool()
    pid_file = tmp_path / "child.pid"
    command = [
        "sh",
        "-c",
        f"sleep 120 >/dev/null 2>&1 & echo $! > {pid_file}; echo started; echo compiling >&2; wait",
    ]

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        tool.run_in_process_group(
            command, cwd=tmp_path, capture_output=True, text=True, timeout=2, check=False
        )

    assert raised.value.timeout == 2
    captured: tuple[object, object] = (raised.value.output, raised.value.stderr)
    assert captured == ("started\n", "compiling\n")
    child = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10
    while not _process_is_gone(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _process_is_gone(child), f"child {child} outlived the timed-out command"


def test_timed_out_target_reports_through_the_real_runner(tmp_path: Path) -> None:
    """The real runner turns a timed-out command into a 124 report with its partial output."""
    tool = _load_tool()

    result = tool._run(
        ["sh", "-c", "echo partial; sleep 120"],
        repo_root=tmp_path,
        run_command=tool.run_in_process_group,
        timeout=1,
    )

    assert result.returncode == 124
    assert result.stdout == "partial\n"
    assert result.stderr == "command timed out after 1 seconds"


def _fuzz_repo(tmp_path: Path, manifest: str, *sources: str) -> Path:
    fuzz = tmp_path / "repo" / "fuzz"
    fuzz.mkdir(parents=True)
    (fuzz / "Cargo.toml").write_text(manifest, encoding="utf-8")
    for source in sources:
        (fuzz / source).parent.mkdir(parents=True, exist_ok=True)
        (fuzz / source).write_text("", encoding="utf-8")
    return fuzz.parent


def test_discovery_skips_malformed_and_missing_targets(tmp_path: Path) -> None:
    """Only well-formed targets whose source file exists are discovered."""
    tool = _load_tool()
    repo = _fuzz_repo(
        tmp_path,
        "bin = [\n"
        '  "not a table",\n'
        '  { name = "no_path" },\n'
        '  { name = 7, path = "fuzz_targets/numeric.rs" },\n'
        '  { name = "missing", path = "fuzz_targets/missing.rs" },\n'
        '  { name = "zeta", path = "fuzz_targets/zeta.rs" },\n'
        '  { name = "alpha", path = "fuzz_targets/alpha.rs" },\n'
        "]\n",
        "fuzz_targets/numeric.rs",
        "fuzz_targets/zeta.rs",
        "fuzz_targets/alpha.rs",
    )

    assert tool.discover_fuzz_targets(repo) == ["alpha", "zeta"]


def test_discovery_of_a_manifest_without_a_target_list_finds_nothing(tmp_path: Path) -> None:
    """A ``bin`` entry that is not an array of tables yields no targets."""
    tool = _load_tool()
    repo = _fuzz_repo(tmp_path, 'bin = "bitstream_ops"\n')

    assert tool.discover_fuzz_targets(repo) == []


@pytest.mark.parametrize(
    ("manifest", "max_total_time", "build_timeout", "message"),
    [
        ("[package]\nname = 'fuzz'\n", 300, 3600, "no fuzz targets discovered"),
        (
            '[[bin]]\nname = "a"\npath = "a.rs"\n[[bin]]\nname = "b"\npath = "b.rs"\n',
            1,
            3600,
            "max_total_time must be at least the selected target count",
        ),
        ('[[bin]]\nname = "a"\npath = "a.rs"\n', 10, 0, "build_timeout must be positive"),
    ],
)
def test_runner_refuses_an_unusable_budget_or_empty_target_set(
    tmp_path: Path, manifest: str, max_total_time: int, build_timeout: int, message: str
) -> None:
    """Budgets and target sets that cannot produce evidence are refused before any build."""
    tool = _load_tool()
    repo = _fuzz_repo(tmp_path, manifest, "a.rs", "b.rs")

    with pytest.raises(ValueError, match=message):
        tool.run_cargo_fuzz_scanners(
            repo_root=repo,
            output_dir=tmp_path / "packet",
            max_total_time=max_total_time,
            build_timeout=build_timeout,
        )
    assert not (tmp_path / "packet").exists()


def test_command_line_refuses_an_unknown_target_with_status_2(tmp_path: Path) -> None:
    """Run as a script, an unknown target exits 2 and reports why, without building."""
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "security_scan" / "run_cargo_fuzz_scanners.py"),
            "--output-dir",
            str(tmp_path / "packet"),
            "--target",
            "missing",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout) == {
        "error": "unknown fuzz targets: missing",
        "passed": False,
    }
    assert not (tmp_path / "packet").exists()


@pytest.mark.parametrize(("passed", "status"), [(True, 0), (False, 1)])
def test_command_line_status_follows_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], passed: bool, status: int
) -> None:
    """The exit status is 0 only for a passing summary, which is printed in full."""
    tool = _load_tool()
    received: dict[str, Any] = {}

    def runner(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"passed": passed, "target_count": 2}

    argv = ["--output-dir", str(tmp_path), "--max-total-time", "60", "--build-timeout", "90"]

    assert tool.main(argv, runner=runner) == status
    assert json.loads(capsys.readouterr().out) == {"passed": passed, "target_count": 2}
    assert received["selected_targets"] == ("all",)
    assert received["max_total_time"] == 60
    assert received["build_timeout"] == 90
    assert received["output_dir"] == tmp_path


def test_main_reports_a_refused_selection_as_status_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real runner's refusal becomes a printed error and exit status 2."""
    tool = _load_tool()

    status = tool.main(["--output-dir", str(tmp_path / "packet"), "--target", "missing"])

    assert status == 2
    assert json.loads(capsys.readouterr().out) == {
        "error": "unknown fuzz targets: missing",
        "passed": False,
    }
