# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia public startup configuration acceptance

"""Exercise actual Python option precedence and already-initialized Julia admission."""

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from tests.julia_runtimes import require_julia_runtime

_PROBE = r"""
import os
from pathlib import Path
import sys
measurement = None
if sys.argv[1]:
    import coverage
    measurement = coverage.Coverage(data_file=sys.argv[1], branch=True, source=[sys.argv[2]], config_file=False)
    measurement.start()
mode = sys.argv[3]
if mode == "torch":
    import torch
elif mode in ("threads", "signals", "project"):
    import juliacall
    os.environ["PYTHON_JULIACALL_THREADS"] = "1"
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    if mode == "project":
        old = Path(os.environ["PYTHON_JULIACALL_PROJECT"])
        changed = old / "changed"
        changed.mkdir()
        for name in ("Project.toml", "Manifest.toml"):
            (changed / name).write_bytes((old / name).read_bytes())
        os.environ["PYTHON_JULIACALL_PROJECT"] = str(changed)
from sc_neurocore.conversion import ConvertedSNN
model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
try:
    model.run([1.0], backend="julia")
except RuntimeError as error:
    if mode == "options":
        assert "startup options conflict" in str(error), str(error)
        assert "juliacall" not in sys.modules
    elif mode == "torch":
        assert "before importing PyTorch" in str(error), str(error)
        assert "juliacall" not in sys.modules
    else:
        assert "managed runtime unavailable or incompatible" in str(error), str(error)
        expected = "configuration changed" if mode == "project" else "thread or signal settings are incompatible"
        assert expected in str(error.__cause__), str(error.__cause__)
else:
    raise AssertionError("conflicting runtime configuration accepted")
assert model.run([1.0], backend="numpy").tolist() == [1.0]
if measurement is not None:
    measurement.stop()
    measurement.save()
print("public startup refusal verified: " + mode, flush=True)
"""


@pytest.fixture(scope="module", params=["1.11", "1.13"])
def julia_environment(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> tuple[dict[str, str], str]:
    """Resolve a real version-matched project once per installed runtime, exclusively offline."""
    runtime = cast(str, request.param)
    binaries = [require_julia_runtime(runtime)]
    assert binaries, f"Julia {runtime} native runtime required"
    project = tmp_path_factory.mktemp(f"julia-startup-{runtime}")
    version = importlib.metadata.version("juliacall")
    (project / "Project.toml").write_text(
        '[deps]\nPythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"\n[compat]\nPythonCall = "='
        + version
        + '"\n'
    )
    environment = dict(os.environ, JULIA_PKG_OFFLINE="true", JULIA_PKG_PRECOMPILE_AUTO="0")
    subprocess.run(
        [
            str(binaries[-1]),
            "--startup-file=no",
            f"--project={project}",
            "-e",
            "using Pkg; Pkg.resolve()",
        ],
        env=environment,
        check=True,
        capture_output=True,
        timeout=60,
    )
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHON_JULIACALL_EXE=str(binaries[-1]),
        PYTHON_JULIACALL_PROJECT=str(project),
        PYTHON_JULIACALL_THREADS="1",
        PYTHON_JULIACALL_HANDLE_SIGNALS="yes",
        PYTHON_JULIACALL_STARTUP_FILE="no",
        JULIA_CONDAPKG_BACKEND="Null",
        SC_NEUROCORE_IF_JULIA_ENABLED="1",
    )
    for name in ("SC_NEUROCORE_IF_RUST_LIB", "SC_NEUROCORE_IF_GO_LIB"):
        environment.pop(name, None)
    return environment, runtime


@pytest.mark.parametrize(
    "mode,option",
    [
        ("options", "juliacall-exe=/absent/julia"),
        ("options", "juliacall-project=/absent/project"),
        ("options", "juliacall-threads=2"),
        ("options", "juliacall-handle-signals=no"),
        ("options", "juliacall-init=no"),
        ("options", "juliacall-exe"),
        ("threads", ""),
        ("signals", ""),
        ("torch", ""),
        ("project", ""),
    ],
)
def test_public_julia_refuses_conflicting_actual_startup(
    tmp_path: Path, julia_environment: tuple[dict[str, str], str], mode: str, option: str
) -> None:
    """Refuse genuine command-line overrides or initialized runtime drift through public run."""
    environment, runtime = julia_environment
    environment = environment.copy()
    if mode == "threads":
        environment["PYTHON_JULIACALL_THREADS"] = "2"
    elif mode == "signals":
        environment["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"
    parent_data = os.environ.get("COVERAGE_FILE", "")
    child_data = f"{parent_data}-startup-{runtime}-{tmp_path.name}" if parent_data else ""
    command = [sys.executable] + (["-X", option] if option else [])
    source = Path(__file__).resolve().parents[1] / "src/sc_neurocore/conversion"
    result = subprocess.run(
        command + ["-c", _PROBE, child_data, str(source), mode],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "native-stdout.txt").write_text(result.stdout)
    (tmp_path / "native-stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "public startup refusal verified: " + mode in result.stdout
