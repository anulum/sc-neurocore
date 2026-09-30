# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — managed Julia public ABI lifetime acceptance

"""Exercise stored public NativeAPI callbacks across actual Julia shutdown."""

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.julia_runtimes import require_julia_runtime

_PROBE = r"""
import atexit
import ctypes
import sys
measurement = None
if sys.argv[1]:
    import coverage
    measurement = coverage.Coverage(data_file=sys.argv[1], branch=True, source=[sys.argv[2]], config_file=False)
    measurement.start()
from sc_neurocore.conversion.if_julia import load_julia
from sc_neurocore.conversion.if_native_types import LayerSpec, ReplayRequest, BufferView
weights = (ctypes.c_double * 1)(1.0)
frames = (ctypes.c_double * 1)(1.0)
specs = (LayerSpec * 1)(LayerSpec(1, 1, ctypes.addressof(weights), 0, 0, 1.0, 0.0, 0, 0))
request = ReplayRequest(1, 3, ctypes.addressof(specs), 1, ctypes.addressof(frames), 1, 1, 1, 128)
handle = ctypes.c_void_p()
view = BufferView()
def at_exit():
    for call in (
        lambda: api.replay(ctypes.byref(request), ctypes.byref(handle)),
        lambda: api.buffer(handle, 0, 0, ctypes.byref(view)),
    ):
        try:
            call()
        except RuntimeError as error:
            assert "closing" in str(error), str(error)
        else:
            raise AssertionError("stored managed callback entered a closed Julia runtime")
    api.free(handle)
    if measurement is not None:
        measurement.stop()
        measurement.save()
    print("stored public ABI refused replay/view and deferred release after shutdown", flush=True)
atexit.register(at_exit)
api = load_julia()
assert api.replay(ctypes.byref(request), ctypes.byref(handle)) == 0
assert handle.value is not None
assert api.buffer(handle, 0, 0, ctypes.byref(view)) == 0
assert view.length == 1
assert ctypes.cast(view.data, ctypes.POINTER(ctypes.c_double))[0] == 1.0
print("actual public ABI owner created and read before shutdown", flush=True)
"""


@pytest.mark.parametrize("runtime", ["1.11", "1.13"])
def test_stored_public_julia_callbacks_refuse_after_actual_shutdown(
    tmp_path: Path, runtime: str
) -> None:
    """Use complete live C metadata and genuine managed runtime exit, without synthetic providers."""
    root = Path(__file__).resolve().parents[1]
    binaries = [require_julia_runtime(runtime)]
    assert binaries, f"Julia {runtime} native runtime required"
    project = tmp_path / "project"
    project.mkdir()
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
        PYTHONPATH=str(root / "src"),
        PYTHON_JULIACALL_EXE=str(binaries[-1]),
        PYTHON_JULIACALL_PROJECT=str(project),
        PYTHON_JULIACALL_THREADS="1",
        PYTHON_JULIACALL_HANDLE_SIGNALS="yes",
        PYTHON_JULIACALL_STARTUP_FILE="no",
        JULIA_CONDAPKG_BACKEND="Null",
        SC_NEUROCORE_IF_JULIA_ENABLED="1",
    )
    parent_data = os.environ.get("COVERAGE_FILE", "")
    child_data = f"{parent_data}-lifetime-{runtime}" if parent_data else ""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, child_data, str(root / "src/sc_neurocore/conversion")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "native-stdout.txt").write_text(result.stdout)
    (tmp_path / "native-stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "actual public ABI owner created and read before shutdown" in result.stdout
    assert (
        "stored public ABI refused replay/view and deferred release after shutdown" in result.stdout
    )
