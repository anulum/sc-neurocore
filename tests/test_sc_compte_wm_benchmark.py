# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC Compte benchmark runtime provenance

"""Exercise actual benchmark reports with real PATH aliases and runtime refusal."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
BENCHMARK = REPOSITORY / "benchmarks/bench_sc_compte_wm_network_all_runtimes.py"
RECORDING = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_network_all_runtimes.json"
TOOLS = ("rustc", "cargo", "julia", "go", "mojo")


def benchmark_environment(
    directory: Path, omitted: str | None = None, *, mojo_via_pixi: bool = False
) -> dict[str, str]:
    """Expose real installed runtimes through a distinct PATH, optionally omitting one.

    Parameters
    ----------
    directory : Path
        Owned test allocation for executable aliases and process temporaries.
    omitted : str, optional
        Actual executable deliberately absent from the child PATH.
    mojo_via_pixi : bool, optional
        Invoke the installed compiler through the hosted Pixi wrapper.

    Returns
    -------
    dict of str to str
        Environment pointing to real tool symlinks, with no synthetic output.
    """
    binaries = directory / "bin"
    binaries.mkdir()
    for name in TOOLS:
        if name == omitted:
            continue
        installed = shutil.which(name)
        assert installed is not None, f"this real-runtime test requires installed {name}"
        (binaries / name).symlink_to(installed)
    pixi = shutil.which("pixi")
    if pixi is not None:
        (binaries / "pixi").symlink_to(pixi)
    if mojo_via_pixi:
        assert pixi is not None, "the hosted Mojo wrapper requires installed pixi"
        manifest = REPOSITORY / "src/sc_neurocore/accel/mojo/pixi.toml"
        wrapper = binaries / "mojo"
        wrapper.unlink()
        wrapper.write_text(
            f'#!/bin/sh\nexec pixi run --manifest-path {shlex.quote(str(manifest))} mojo "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
    return {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.defpath,
        "PYTHONPATH": str(REPOSITORY / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "TMPDIR": str(directory),
        "GOTOOLCHAIN": "local",
    }


@pytest.mark.parametrize(
    ("omitted", "mojo_via_pixi"),
    [(None, False), ("mojo", False), ("rustc", False), (None, True)],
    ids=["native", "no-mojo-command", "no-rustc-command", "hosted-pixi-wrapper"],
)
def test_environment_report_uses_dispatch_path_and_module(
    tmp_path: Path, omitted: str | None, mojo_via_pixi: bool
) -> None:
    """Report actual aliases, module-selected Go and the hosted Mojo launcher.

    Parameters
    ----------
    tmp_path : Path
        Owned directory for executable aliases and process temporaries.
    omitted : str or None
        Standalone compiler command deliberately absent from PATH.
    mojo_via_pixi : bool
        Use the real Pixi compiler wrapper installed by hosted CI.
    """
    environment = benchmark_environment(tmp_path, omitted, mojo_via_pixi=mojo_via_pixi)
    output = tmp_path / "unused-recording.json"
    result = subprocess.run(
        [sys.executable, str(BENCHMARK), "--show-environment", "--output", str(output)],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not output.exists()
    report = json.loads(result.stdout)
    for name in TOOLS:
        expected = None if name == omitted else str(tmp_path / "bin" / name)
        assert report["runtime_commands"][name] == expected
    if omitted == "mojo":
        assert report["mojo"] is None
    else:
        assert isinstance(report["mojo"], str) and report["mojo"]
    if omitted == "rustc":
        assert report["rust"] is None
    else:
        assert isinstance(report["rust"], str) and report["rust"]
    go_module = REPOSITORY / "src/sc_neurocore/accel/go"
    go_home = REPOSITORY / ".venv/go"
    selected_go = subprocess.run(
        [str(tmp_path / "bin/go"), "version"],
        cwd=go_module,
        env={
            **environment,
            "GOTOOLCHAIN": "auto",
            "GOPATH": str(go_home),
            "GOMODCACHE": str(go_home / "pkg/mod"),
            "GOCACHE": str(go_home / "cache"),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert report["go"] == selected_go.stdout.strip()
    library = REPOSITORY / report["mojo_library"]["path"]
    assert report["mojo_library"]["sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
    assert report["mojo_library"]["build_compiler_verified"] is False


@pytest.mark.parametrize("missing", ["cargo", "go"])
def test_missing_dispatch_runtime_refuses_before_recording(tmp_path: Path, missing: str) -> None:
    """A missing real command yields an authored refusal and leaves the destination absent."""
    environment = benchmark_environment(tmp_path, missing)
    output = tmp_path / "refused-recording.json"
    result = subprocess.run(
        [sys.executable, str(BENCHMARK), "--show-environment", "--output", str(output)],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert f"required {missing} runtime is not on PATH" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(("steps", "repeats"), [(0, 3), (1000, 0)])
def test_public_recording_api_refuses_empty_measurements(steps: int, repeats: int) -> None:
    """Refuse a zero-length task or empty repetition set before any runtime execution."""
    from benchmarks.bench_sc_compte_wm_network_all_runtimes import build_payload

    with pytest.raises(ValueError, match="steps and repeats must be positive"):
        build_payload(steps, repeats)


def test_public_recording_api_refuses_a_missing_backend(tmp_path: Path) -> None:
    """An actually absent Go command prevents the whole five-runtime measurement."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from benchmarks.bench_sc_compte_wm_network_all_runtimes import build_payload; "
            "build_payload(1000, 3)",
        ],
        cwd=REPOSITORY,
        env=benchmark_environment(tmp_path, "go"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1
    assert "unavailable SC Compte backends: go" in result.stderr


def test_full_recording_retains_all_five_runtime_custody() -> None:
    """Bind the actual default rerun to live source, all lanes and loaded-library identity."""
    payload = json.loads(RECORDING.read_text())
    assert payload["configuration"]["steps"] == 1000
    assert payload["configuration"]["repeats"] == 3
    assert tuple(payload["backends"]) == ("python", "rust", "julia", "go", "mojo")
    assert payload["all_runtime_input_spike_count_exact"] is True
    assert payload["all_repeat_receipts_exact"] is True
    assert payload["passed"] is True
    assert payload["source_sha256"][BENCHMARK.relative_to(REPOSITORY).as_posix()] == (
        hashlib.sha256(BENCHMARK.read_bytes()).hexdigest()
    )
    assert len(payload["environment"]["mojo_library"]["sha256"]) == 64
    assert payload["environment"]["mojo_library"]["build_compiler_verified"] is False
    assert payload["environment"]["go_toolchain_selection"] == "auto in src/sc_neurocore/accel/go"
    assert len(payload["command"]) >= 2
    assert payload["isolation"]["isolated"] is False
    for result in payload["backends"].values():
        assert len(result["execution_samples_ns"]) == len(result["end_to_end_samples_ns"]) == 3
