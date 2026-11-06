# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coverage for the maintained Mojo runner surface

"""Run the Mojo runner against real stand-in ``pixi`` executables and read its log.

Each stand-in is a POSIX shell script in the test directory. It records how it
was called, prints benchmark lines, fails, or outlives the timeout, so the
runner's subprocess handling and its logged failures are exercised for real.
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

import pytest

from sc_neurocore.accel import mojo as mojo_namespace
from sc_neurocore.accel.mojo import runner as mojo_runner_module
from sc_neurocore.accel.mojo.runner import MojoKernelRunner

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX shell scripts stand in for the pixi executable"
)


def _make_runner(tmp_path: Path, pixi_script: str | None = None) -> MojoKernelRunner:
    """Return a runner over a test kernel directory, with an optional stand-in ``pixi``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test directory that receives ``kernels.mojo`` and the script.
    pixi_script : str, optional
        Body of the stand-in ``pixi`` shell script; a missing path when omitted.

    Returns
    -------
    MojoKernelRunner
        Runner whose ``_pixi_bin`` is the stand-in script.
    """
    (tmp_path / "kernels.mojo").write_text("// test kernel file\n", encoding="utf-8")
    pixi = tmp_path / "pixi"
    if pixi_script is not None:
        pixi.write_text("#!/bin/sh\n" + pixi_script, encoding="utf-8")
        pixi.chmod(0o755)
    return MojoKernelRunner(_mojo_dir=tmp_path, _pixi_bin=str(pixi))


def _runner_helper_methods() -> dict[str, ast.FunctionDef]:
    """Return the scalar helper methods of the runner class, read from its source."""
    module_file = mojo_runner_module.__file__
    assert module_file is not None
    source_path = Path(module_file).resolve()
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "MojoKernelRunner":
            return {
                child.name: child
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name in {"popcount", "lfsr_encode"}
            }
    raise AssertionError("MojoKernelRunner class not found")


def _errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the runner's ERROR messages captured so far."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == mojo_runner_module.__name__ and record.levelno == logging.ERROR
    ]


def test_helper_contract_declares_python_fallback_not_mojo_ipc() -> None:
    """The scalar helpers are declared as Python fallbacks without Mojo IPC."""
    assert mojo_runner_module.MOJO_HELPER_BACKEND == "python-fallback"
    assert mojo_runner_module.MOJO_HELPER_IPC_AVAILABLE is False
    assert mojo_namespace.MOJO_HELPER_BACKEND == mojo_runner_module.MOJO_HELPER_BACKEND
    assert mojo_namespace.MOJO_HELPER_IPC_AVAILABLE is False


def test_helper_methods_do_not_hide_mojo_ipc_stubs() -> None:
    """The scalar helpers neither raise nor swallow a broad exception."""
    helper_methods = _runner_helper_methods()
    assert set(helper_methods) == {"popcount", "lfsr_encode"}

    for method in helper_methods.values():
        assert not any(isinstance(node, ast.Raise) for node in ast.walk(method))
        assert not any(
            isinstance(node, ast.ExceptHandler)
            and (
                node.type is None
                or (isinstance(node.type, ast.Name) and node.type.id == "Exception")
            )
            for node in ast.walk(method)
        )


def test_post_init_falls_back_to_installed_package_dir(tmp_path: Path) -> None:
    """A directory without ``kernels.mojo`` falls back to the installed package copy."""
    runner = MojoKernelRunner(_mojo_dir=tmp_path)
    assert runner._mojo_dir != tmp_path
    assert (runner._mojo_dir / "kernels.mojo").exists()


def test_post_init_raises_when_source_and_package_kernels_are_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without ``kernels.mojo`` in either place the runner refuses to construct."""
    source_dir = tmp_path / "source"
    package_dir = tmp_path / "package"
    source_dir.mkdir()
    package_dir.mkdir()
    monkeypatch.setattr(mojo_runner_module, "__file__", str(package_dir / "runner.py"))

    with pytest.raises(FileNotFoundError, match="kernels\\.mojo not found"):
        MojoKernelRunner(_mojo_dir=source_dir)


@posix_only
def test_build_success_invokes_pixi(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A clean build runs ``pixi run mojo build`` in the kernel directory and logs nothing."""
    runner = _make_runner(tmp_path, 'printf "%s\\n" "$PWD" "$@" > "$(dirname "$0")/call.txt"\n')
    with caplog.at_level(logging.ERROR, logger=mojo_runner_module.__name__):
        assert runner.build() is True
    call = (tmp_path / "call.txt").read_text(encoding="utf-8").splitlines()
    assert call == [
        str(tmp_path),
        "run",
        "mojo",
        "build",
        "--target-cpu",
        "x86-64-v3",
        "kernels.mojo",
    ]
    assert _errors(caplog) == []


@posix_only
def test_build_failure_returns_false_and_logs_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing build returns ``False`` and logs the error instead of printing it."""
    runner = _make_runner(tmp_path, "exit 3\n")
    with caplog.at_level(logging.ERROR, logger=mojo_runner_module.__name__):
        assert runner.build() is False
    (message,) = _errors(caplog)
    assert message.startswith("Mojo kernel build failed: ") and "exit status 3" in message


@posix_only
def test_run_benchmark_parses_millisecond_lines(tmp_path: Path) -> None:
    """Lines with a label and a millisecond value become timings; other lines are ignored."""
    runner = _make_runner(tmp_path, 'printf "popcount: 1.25 ms\\nlfsr: 2.5ms\\nignored line\\n"\n')
    assert runner.run_benchmark(timeout_sec=12) == {"popcount": 1.25, "lfsr": 2.5}


@posix_only
@pytest.mark.parametrize(
    ("script", "timeout_sec", "logged"),
    [
        ("echo boom >&2\nexit 1\n", 60, "Mojo kernel benchmark failed: boom"),
        ("exec sleep 5\n", 1, "Mojo kernel benchmark exceeded its 1 s timeout"),
        (None, 60, "pixi was not found at "),
    ],
)
def test_run_benchmark_error_paths_return_empty_and_log(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    script: str | None,
    timeout_sec: int,
    logged: str,
) -> None:
    """A failing, overrunning or missing toolchain returns no timings and logs why.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test directory.
    caplog : pytest.LogCaptureFixture
        Captured log records.
    script : str or None
        Body of the stand-in ``pixi``, or ``None`` for a missing executable.
    timeout_sec : int
        Timeout handed to the runner.
    logged : str
        Start of the expected ERROR message.
    """
    runner = _make_runner(tmp_path, script)
    with caplog.at_level(logging.ERROR, logger=mojo_runner_module.__name__):
        assert runner.run_benchmark(timeout_sec=timeout_sec) == {}
    (message,) = _errors(caplog)
    assert message.startswith(logged)


def test_popcount_falls_back_to_python(tmp_path: Path) -> None:
    """``popcount`` counts the set bits of packed words in Python."""
    runner = _make_runner(tmp_path)
    assert runner.popcount([0b1010, 0b1111]) == 6


def test_lfsr_encode_falls_back_to_python(tmp_path: Path) -> None:
    """``lfsr_encode`` returns packed integer words from the Python LFSR."""
    runner = _make_runner(tmp_path)
    encoded = runner.lfsr_encode(seed=0xACE1, threshold=32768, bits=32)
    assert encoded
    assert all(isinstance(word, int) for word in encoded)
