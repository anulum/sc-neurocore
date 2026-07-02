# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coverage for the maintained Mojo runner surface

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sc_neurocore.accel.mojo import runner as mojo_runner_module
from sc_neurocore.accel.mojo.runner import MojoKernelRunner


def _make_runner(tmp_path: Path) -> MojoKernelRunner:
    (tmp_path / "kernels.mojo").write_text("// test kernel file\n", encoding="utf-8")
    return MojoKernelRunner(_mojo_dir=tmp_path, _pixi_bin="/fake/pixi")


def test_post_init_falls_back_to_installed_package_dir(tmp_path: Path) -> None:
    runner = MojoKernelRunner(_mojo_dir=tmp_path)
    assert runner._mojo_dir != tmp_path
    assert (runner._mojo_dir / "kernels.mojo").exists()


def test_post_init_raises_when_source_and_package_kernels_are_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_dir = tmp_path / "source"
    package_dir = tmp_path / "package"
    source_dir.mkdir()
    package_dir.mkdir()
    monkeypatch.setattr(mojo_runner_module, "__file__", str(package_dir / "runner.py"))

    with pytest.raises(FileNotFoundError, match="kernels\\.mojo not found"):
        MojoKernelRunner(_mojo_dir=source_dir)


def test_build_success_invokes_pixi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], cwd: str, check: bool) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert runner.build() is True
    assert captured["cmd"] == ["/fake/pixi", "run", "mojo", "build", "kernels.mojo"]
    assert captured["cwd"] == str(tmp_path)
    assert captured["check"] is True


def test_build_failure_returns_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)

    def fake_run(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("broken build")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert runner.build() is False


def test_run_benchmark_parses_millisecond_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _make_runner(tmp_path)

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["/fake/pixi"],
            returncode=0,
            stdout="popcount: 1.25 ms\nlfsr: 2.5ms\nignored line\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    timings = runner.run_benchmark(timeout_sec=12)
    assert timings == {"popcount": 1.25, "lfsr": 2.5}


@pytest.mark.parametrize(
    ("exc", "match"),
    [
        (subprocess.CalledProcessError(1, ["/fake/pixi"], stderr="boom"), {}),
        (subprocess.TimeoutExpired(cmd=["/fake/pixi"], timeout=1), {}),
        (FileNotFoundError("missing pixi"), {}),
    ],
)
def test_run_benchmark_error_paths_return_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exc: Exception,
    match: dict[str, float],
) -> None:
    runner = _make_runner(tmp_path)

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise exc

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert runner.run_benchmark() == match


def test_popcount_falls_back_to_python(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    assert runner.popcount([0b1010, 0b1111]) == 6


def test_lfsr_encode_falls_back_to_python(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    encoded = runner.lfsr_encode(seed=0xACE1, threshold=32768, bits=32)
    assert encoded
    assert all(isinstance(word, int) for word in encoded)
