# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (commands_and_build) from former test_build_accel_backends.py

from __future__ import annotations

from build_accel_backends_support import *  # noqa: F403

def test_go_command(tmp_path: Path) -> None:
    target = _target(tmp_path, "go")
    assert MOD._go_command(target) == [
        "go",
        "build",
        "-buildmode=c-shared",
        "-o",
        "libx.so",
        ".",
    ]


def test_mojo_command(tmp_path: Path) -> None:
    target = _target(tmp_path, "mojo")
    cmd = MOD._mojo_command(target, ["pixi", "run", "mojo"])
    assert cmd == [
        "pixi",
        "run",
        "mojo",
        "build",
        "--emit",
        "shared-lib",
        # Portable ISA baseline so the library runs on any x86-64 CI runner (AVX2, no AVX-512).
        "--target-cpu",
        "x86-64-v3",
        "-o",
        "libx.so",
        "src.go",
    ]


def test_build_target_success_writes_library(tmp_path: Path) -> None:
    target = _target(tmp_path, "go")

    def runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        target.output.write_bytes(b"\x7fELF")
        return _fake_completed(0)

    result = MOD.build_target(target, runner=runner)
    assert result.ok and result.detail == "CGO_ENABLED=1"


def test_build_target_mojo_success(tmp_path: Path) -> None:
    target = _target(tmp_path, "mojo")

    def runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        target.output.write_bytes(b"\x7fELF")
        return _fake_completed(0)

    result = MOD.build_target(target, mojo_command=["mojo"], runner=runner)
    assert result.ok and result.detail == "ok"


def test_build_target_nonzero_exit(tmp_path: Path) -> None:
    target = _target(tmp_path, "go")
    result = MOD.build_target(target, runner=lambda c, w: _fake_completed(2, "boom\nkaboom"))
    assert not result.ok and "exit 2" in result.detail and "kaboom" in result.detail


def test_build_target_success_but_no_library(tmp_path: Path) -> None:
    target = _target(tmp_path, "go")
    result = MOD.build_target(target, runner=lambda c, w: _fake_completed(0))
    assert not result.ok and "no library" in result.detail


def test_build_target_toolchain_missing(tmp_path: Path) -> None:
    target = _target(tmp_path, "mojo")

    def runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("mojo")

    result = MOD.build_target(target, runner=runner)
    assert not result.ok and "toolchain missing" in result.detail


def test_default_runner_sets_cgo_for_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    # Control the ambient toolchain env so the assertions are deterministic.
    monkeypatch.delenv("GOTOOLCHAIN", raising=False)

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _fake_completed(0)

    monkeypatch.setattr(MOD.subprocess, "run", fake_run)
    MOD._default_runner(["go", "build"], tmp_path)
    assert captured["env"]["CGO_ENABLED"] == "1"
    # Toolchain auto-management so the newer go.mod requirement is fetched
    # instead of failing under a GOTOOLCHAIN=local CI runner.
    assert captured["env"]["GOTOOLCHAIN"] == "auto"
    # non-go command leaves CGO/toolchain untouched (delegates to inherited env)
    captured.clear()
    monkeypatch.setattr(MOD.subprocess, "run", fake_run)
    MOD._default_runner(["mojo", "build"], tmp_path)
    assert "cmd" in captured
    assert "GOTOOLCHAIN" not in captured["env"]
