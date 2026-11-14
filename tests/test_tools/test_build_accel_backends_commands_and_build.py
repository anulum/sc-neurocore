# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (commands_and_build) from former test_build_accel_backends.py

"""Commands and single-target builds of the Go and Mojo accelerator backend builder."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from build_accel_backends_support import MOD, _fake_completed, _target


def test_go_command(tmp_path: Path) -> None:
    """The package is built to the path it is given, not beside its source."""
    scratch = tmp_path / "scratch" / "libx.so"
    assert MOD._go_command(scratch) == [
        "go",
        "build",
        "-buildmode=c-shared",
        "-o",
        str(scratch),
        ".",
    ]


def _go_runner(library: bytes, header: bytes | None, built: list[Path]) -> Any:
    """Return a runner that writes what cgo writes, at the path the command names.

    Parameters
    ----------
    library : bytes
        The shared library's bytes.
    header : bytes or None
        The header's bytes, or ``None`` for a build that writes none.
    built : list[Path]
        Receives the output path of each command.

    Returns
    -------
    Any
        The runner.
    """

    def runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        output = Path(cmd[cmd.index("-o") + 1])
        built.append(output)
        output.write_bytes(library)
        if header is not None:
            output.with_suffix(".h").write_bytes(header)
        return _fake_completed(0)

    return runner


def test_a_go_build_writes_only_the_library_beside_its_source(tmp_path: Path) -> None:
    """The header cgo writes stays in the scratch folder, which is gone after the build."""
    target = _target(tmp_path, "go")
    built: list[Path] = []
    result = MOD.build_target(target, runner=_go_runner(b"\x7fELF", b"/* cgo */\n", built))
    assert result.ok and result.detail == "CGO_ENABLED=1"
    assert target.output.read_bytes() == b"\x7fELF"
    assert not (tmp_path / "libx.h").exists()
    assert [output.parent == tmp_path for output in built] == [False]
    assert not built[0].parent.exists()
    assert result.header_differs is False


def test_a_go_build_leaves_the_tracked_header_and_reports_a_difference(tmp_path: Path) -> None:
    """A header cgo writes differently from the tracked one is reported, never written."""
    target = _target(tmp_path, "go")
    pinned = tmp_path / "libx.h"
    pinned.write_bytes(b"/* the committed ABI */\n")
    result = MOD.build_target(target, runner=_go_runner(b"\x7fELF", b"/* other */\n", []))
    assert result.ok and result.header_differs is True
    assert (
        result.detail == "CGO_ENABLED=1; cgo wrote a header that differs from libx.h (not written)"
    )
    assert pinned.read_bytes() == b"/* the committed ABI */\n"


def test_a_go_build_whose_header_matches_the_tracked_one_reports_nothing(tmp_path: Path) -> None:
    """The committed header reproduced byte for byte is the expected case."""
    target = _target(tmp_path, "go")
    (tmp_path / "libx.h").write_bytes(b"/* cgo */\n")
    result = MOD.build_target(target, runner=_go_runner(b"\x7fELF", b"/* cgo */\n", []))
    assert result.ok and result.header_differs is False
    assert result.detail == "CGO_ENABLED=1"


def test_a_go_build_replaces_a_link_to_a_library_and_leaves_the_linked_file(
    tmp_path: Path,
) -> None:
    """A library path that links elsewhere becomes the new library; the linked file stays."""
    target = _target(tmp_path, "go")
    elsewhere = tmp_path / "another_checkout" / "libx.so"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"another checkout's library")
    target.output.symlink_to(elsewhere)
    result = MOD.build_target(target, runner=_go_runner(b"\x7fELF", None, []))
    assert result.ok
    assert not target.output.is_symlink()
    assert target.output.read_bytes() == b"\x7fELF"
    assert elsewhere.read_bytes() == b"another checkout's library"


def test_a_go_build_gives_the_library_a_new_file_with_the_build_mode(tmp_path: Path) -> None:
    """A process holding the former library keeps reading it; the new one has the build's mode."""
    target = _target(tmp_path, "go")
    target.output.write_bytes(b"former library")
    built: list[Path] = []
    write = _go_runner(b"\x7fELF", None, built)

    def runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        completed: subprocess.CompletedProcess[str] = write(cmd, cwd)
        built[-1].chmod(0o751)
        return completed

    with target.output.open("rb") as holder:
        former = os.fstat(holder.fileno()).st_ino
        result = MOD.build_target(target, runner=runner)
        assert holder.read() == b"former library"
    assert result.ok
    assert target.output.read_bytes() == b"\x7fELF"
    assert target.output.stat().st_ino != former
    assert stat.S_IMODE(target.output.stat().st_mode) == 0o751


def test_a_failed_library_copy_leaves_the_former_library_and_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy that fails removes its partial file and leaves the former library in place."""
    target = _target(tmp_path, "go")
    target.output.write_bytes(b"former library")
    before = sorted(path.name for path in tmp_path.iterdir())

    def full_disk(source: Path, destination: str | Path) -> None:
        Path(destination).write_bytes(b"\x7f")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(MOD.shutil, "copyfile", full_disk)
    with pytest.raises(OSError, match="No space left on device"):
        MOD.build_target(target, runner=_go_runner(b"\x7fELF", None, []))
    assert target.output.read_bytes() == b"former library"
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_mojo_command(tmp_path: Path) -> None:
    """The Mojo build pins the portable ISA baseline and names the source and the library."""
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
    """A Go build that writes its library succeeds with cgo, the library beside its source."""
    target = _target(tmp_path, "go")
    result = MOD.build_target(target, runner=_go_runner(b"\x7fELF", None, []))
    assert result.ok and result.detail == "CGO_ENABLED=1"
    assert target.output.read_bytes() == b"\x7fELF"


def test_build_target_mojo_success(tmp_path: Path) -> None:
    """A Mojo build that writes its library beside its source succeeds."""
    target = _target(tmp_path, "mojo")

    def runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        target.output.write_bytes(b"\x7fELF")
        return _fake_completed(0)

    result = MOD.build_target(target, mojo_command=["mojo"], runner=runner)
    assert result.ok and result.detail == "ok"


def test_build_target_nonzero_exit(tmp_path: Path) -> None:
    """Keep the first compiler error when warnings extend beyond the old tail."""
    target = _target(tmp_path, "go")
    diagnostics = "first error\n" + "warning\n" * 8 + "last note\n"
    result = MOD.build_target(
        target,
        runner=lambda c, w: subprocess.CompletedProcess(
            args=c, returncode=2, stdout="compiler summary\n", stderr=diagnostics
        ),
    )
    assert not result.ok
    assert result.detail == (
        f"exit 2; source={target.source}\n"
        f"stdout:\ncompiler summary\n"
        f"stderr:\n{diagnostics.rstrip()}"
    )


def test_build_target_success_but_no_library(tmp_path: Path) -> None:
    """A build that exits 0 without writing a library is a failure."""
    target = _target(tmp_path, "go")
    result = MOD.build_target(target, runner=lambda c, w: _fake_completed(0))
    assert not result.ok and "no library" in result.detail


def test_a_mojo_build_without_a_library_fails(tmp_path: Path) -> None:
    """A Mojo build that exits 0 without writing its library is a failure."""
    target = _target(tmp_path, "mojo")
    result = MOD.build_target(target, mojo_command=["mojo"], runner=lambda c, w: _fake_completed(0))
    assert not result.ok and result.detail == "build reported success but no library produced"


def test_a_failed_build_without_output_names_its_exit_and_source(tmp_path: Path) -> None:
    """A build that fails silently is reported by its exit status and source alone."""
    target = _target(tmp_path, "go")
    result = MOD.build_target(target, runner=lambda c, w: _fake_completed(2))
    assert not result.ok and result.detail == f"exit 2; source={target.source}"


def test_build_target_toolchain_missing(tmp_path: Path) -> None:
    """A missing toolchain is reported, not raised."""
    target = _target(tmp_path, "mojo")

    def runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("mojo")

    result = MOD.build_target(target, runner=runner)
    assert not result.ok and "toolchain missing" in result.detail


def test_default_runner_sets_cgo_for_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Go builds run with cgo and an automatic toolchain; others inherit the environment."""
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
