# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual Julia allocation refusal and recovery

"""Exercise native allocation failure under a process-owned Linux address-space limit."""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

_PROBE = r"""
import gc
import json
import os
from pathlib import Path
import resource
import sys
measurement = None
if sys.argv[1]:
    import coverage
    measurement = coverage.Coverage(
        data_file=sys.argv[1], branch=True, source=[sys.argv[2]], config_file=False
    )
    measurement.start()
import numpy as np
from sc_neurocore.conversion import ConvertedSNN
model = ConvertedSNN([np.ones((16, 1))], [None], [1.0], T=1)
small = np.ones((2, 1, 1))
for _ in range(3):
    actual = model.replay(small, trace=True, backend="julia")
    del actual
gc.collect()
import juliacall
juliacall.Main.GC.gc()
frames = np.ones((1000000, 1, 1))
expected = model.replay(small, trace=True, backend="numpy")
old_limit = resource.getrlimit(resource.RLIMIT_AS)
virtual = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
headroom = 128 * 1024 * 1024
print(json.dumps({"virtual_bytes": virtual, "headroom_bytes": headroom}), flush=True)
try:
    resource.setrlimit(resource.RLIMIT_AS, (virtual + headroom, old_limit[1]))
    try:
        model.replay(frames, trace=True, backend="julia", max_working_bytes=512 * 1024 * 1024)
    except MemoryError as error:
        assert str(error) == "native dense IF numeric reservation refused", str(error)
        print("actual native allocation refused", flush=True)
    else:
        raise AssertionError("actual native allocation did not exceed process limit")
finally:
    resource.setrlimit(resource.RLIMIT_AS, old_limit)
assert resource.getrlimit(resource.RLIMIT_AS) == old_limit
assert (frames == 1).all()
gc.collect()
juliacall.Main.GC.gc()
actual = model.replay(small, trace=True, backend="julia")
groups = [(actual.output,), actual.final_state, actual.state_trace, actual.spike_trace]
references = [(expected.output,), expected.final_state, expected.state_trace, expected.spike_trace]
for got, wanted in zip(groups, references, strict=True):
    for a, b in zip(got, wanted, strict=True):
        assert a.shape == b.shape and a.tobytes() == b.tobytes()
assert "torch" not in sys.modules
print("public replay recovered with complete bit parity", flush=True)
if measurement is not None:
    measurement.stop()
    measurement.save()
"""


def test_actual_julia_allocation_failure_preserves_inputs_and_recovers(tmp_path: Path) -> None:
    """Limit only a fresh child after Julia warmup and exercise real native refusal and recovery."""
    root = Path(__file__).resolve().parents[1]
    executables = sorted((Path.home() / ".julia/juliaup").glob("julia-1.11.*/bin/julia"))
    assert executables, "installed Julia 1.11 runtime required"
    project = tmp_path / "project"
    project.mkdir()
    version = importlib.metadata.version("juliacall")
    (project / "Project.toml").write_text(
        '[deps]\nPythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"\n'
        f'[compat]\nPythonCall = "={version}"\n'
    )
    environment = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        PYTHON_JULIACALL_EXE=str(executables[-1]),
        PYTHON_JULIACALL_PROJECT=str(project),
        PYTHON_JULIACALL_THREADS="1",
        PYTHON_JULIACALL_HANDLE_SIGNALS="yes",
        PYTHON_JULIACALL_STARTUP_FILE="no",
        JULIA_CONDAPKG_BACKEND="Null",
        JULIA_PKG_OFFLINE="true",
        JULIA_PKG_PRECOMPILE_AUTO="0",
        SC_NEUROCORE_IF_JULIA_ENABLED="1",
    )
    subprocess.run(
        [
            str(executables[-1]),
            "--startup-file=no",
            f"--project={project}",
            "-e",
            "using Pkg; Pkg.resolve()",
        ],
        env=environment,
        capture_output=True,
        check=True,
        timeout=120,
    )
    parent_data = os.environ.get("COVERAGE_FILE", "")
    child_data = f"{parent_data}-julia-oom" if parent_data else ""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, child_data, str(root / "src/sc_neurocore/conversion")],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "native-stdout.txt").write_text(result.stdout)
    (tmp_path / "native-stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    record = next(json.loads(line) for line in result.stdout.splitlines() if line.startswith("{"))
    assert record["headroom_bytes"] == 128 * 1024 * 1024
    assert "actual native allocation refused" in result.stdout
    assert "public replay recovered with complete bit parity" in result.stdout
