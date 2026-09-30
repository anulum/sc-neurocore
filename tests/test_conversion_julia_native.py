# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Managed Julia public dense IF runtime acceptance

"""Exercise actual locked Julia runtimes, complete replay and returned-view lifetime in fresh processes."""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.julia_runtimes import juliacall_host_environment, require_julia_runtime


_PROBE = r"""
import atexit
import gc
import json
import os
from pathlib import Path
import sys
import weakref
measurement = None
if sys.argv[1]:
    import coverage
    measurement = coverage.Coverage(data_file=sys.argv[1], branch=True, source=[sys.argv[2]], config_file=False)
    measurement.start()
import numpy as np
from sc_neurocore.conversion import ConvertedSNN
assert "torch" not in sys.modules
last_model = None

def at_exit():
    try:
        last_model.run([1.0], backend="julia")
    except RuntimeError as error:
        assert "closing" in str(error)
        print("actual shutdown admission refused", flush=True)
    else:
        raise AssertionError("closed Julia runtime accepted public computation")
    expired = [weakref.ref(result.output) for result in results[:-1]]
    results.clear()
    gc.collect()
    assert all(reference() is None for reference in expired)
    print("retained result views released after shutdown", flush=True)
    if measurement is not None:
        measurement.stop()
        measurement.save()
atexit.register(at_exit)

def same(actual, expected):
    for left, right in zip([(actual.output,), actual.final_state, actual.state_trace, actual.spike_trace],
                           [(expected.output,), expected.final_state, expected.state_trace, expected.spike_trace], strict=True):
        for a, b in zip(left, right, strict=True):
            assert a.shape == b.shape and a.tobytes() == b.tobytes()

cases = 0
for mode in ("spikes", "linear"):
    for binary in (False, True):
        for steps, batch in ((7, 3), (0, 2), (5, 0)):
            for seed in range(4):
                rng = np.random.default_rng(seed)
                model = ConvertedSNN([rng.normal(0.3, 0.4, (4, 3)), rng.normal(0.1, 0.5, (2, 4))],
                    [None if seed % 2 else rng.normal(0, 0.1, 4), rng.normal(0, 0.1, 2)],
                    [0.75, 1.25], T=129, output_mode=mode, layer_membrane_fractions=[0.0, 0.5])
                frames = rng.random((steps, batch, 3))
                if binary:
                    frames = (frames < 0.5).astype(np.float64)
                initial = [rng.normal(0, 0.5, (batch, 4)), rng.normal(0, 0.5, (batch, 2))] if seed % 2 else None
                expected = model.replay(frames, initial_state=initial, trace=True, binary_inputs=binary, backend="numpy")
                actual = model.replay(frames, initial_state=initial, trace=True, binary_inputs=binary, backend="julia")
                same(actual, expected)
                same(model.replay(frames, initial_state=initial, trace=True, binary_inputs=binary), expected)
                split = steps // 2
                first = model.replay(frames[:split], initial_state=initial, binary_inputs=binary, backend="julia")
                second = model.replay(frames[split:], initial_state=first.final_state, binary_inputs=binary, backend="julia")
                for a, b in zip(second.final_state, expected.final_state, strict=True):
                    assert a.tobytes() == b.tobytes()
                held = actual.final_state[0][:]
                saved = held.copy()
                frames.fill(0)
                model.weights[0].fill(99)
                del actual, first, second
                gc.collect()
                import juliacall
                juliacall.Main.GC.gc()
                assert held.tobytes() == saved.tobytes()
                held.fill(-99)
                assert expected.final_state[0].tobytes() == saved.tobytes()
                cases += 1
for mode in ("spikes", "linear"):
    model = ConvertedSNN([[[0.75, -0.25], [0.25, 0.5]]], [[0.1, -0.2]], [0.75], T=129, output_mode=mode, output_scale=3.0)
    for encoding in ("poisson", "constant"):
        for x in (np.array([0.75, 0.25]), np.array([[0.75, 0.25], [0.1, 0.9]])):
            expected = model.run(x, input_mode=encoding, seed=71, backend="numpy")
            assert model.run(x, input_mode=encoding, seed=71, backend="julia").tobytes() == expected.tobytes()
            assert model.rates(x, input_mode=encoding, seed=71, backend="julia").tobytes() == (expected / 129 * 3).tobytes()
            np.testing.assert_array_equal(model.classify(x, backend="julia"), model.classify(x, backend="numpy"))
last_model = ConvertedSNN([[[1.0]]], [None], [1.0], T=129)
assert last_model.run(np.empty((0, 1)), backend="julia").shape == (0, 1)
assert last_model.run([1.0], backend="julia", max_working_bytes=1136).tolist() == [129]
try:
    last_model.run([1.0], backend="julia", max_working_bytes=1135)
except MemoryError:
    pass
else:
    raise AssertionError("native encoded reservation not enforced")
extreme = np.finfo(np.float64).max
try:
    ConvertedSNN([[[extreme]]], [[extreme]], [1.0], T=1).replay([[[1.0]]], backend="julia")
except FloatingPointError:
    pass
else:
    raise AssertionError("native arithmetic overflow not refused")
from concurrent.futures import ThreadPoolExecutor
frames = np.ones((7, 2, 1))
expected = last_model.replay(frames, trace=True, backend="numpy")
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(lambda _: last_model.replay(frames, trace=True, backend="julia"), range(16)))
for actual in results:
    same(actual, expected)
results[0].final_state[0].fill(-99)
for actual in results[1:]:
    same(actual, expected)
print(json.dumps({"complete_cases": cases, "concurrent_calls": 16, "torch_not_loaded": "torch" not in sys.modules}), flush=True)
"""


@pytest.mark.parametrize("runtime", ["1.11", "1.13"])
def test_julia_public_replay_and_owned_buffers_in_locked_runtime(
    tmp_path: Path, runtime: str
) -> None:
    """Run actual JuliaCall with a cached, version-matched offline graph and the complete public corpus."""
    root = Path(__file__).resolve().parents[1]
    binaries = [require_julia_runtime(runtime)]
    assert binaries, f"Julia {runtime} native runtime required"
    project = tmp_path / "project"
    project.mkdir()
    version = importlib.metadata.version("juliacall")
    (project / "Project.toml").write_text(
        '[deps]\nPythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"\n\n[compat]\nPythonCall = "='
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
    environment.update(juliacall_host_environment(binaries[-1]))
    environment.pop("SC_NEUROCORE_IF_RUST_LIB", None)
    environment.pop("SC_NEUROCORE_IF_GO_LIB", None)
    parent_data = os.environ.get("COVERAGE_FILE", "")
    child_data = f"{parent_data}-julia-{runtime}" if parent_data else ""
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
    records = [
        json.loads(row) for row in result.stdout.splitlines() if row.startswith('{"complete_cases"')
    ]
    assert records == [{"complete_cases": 48, "concurrent_calls": 16, "torch_not_loaded": True}]
    assert "actual shutdown admission refused" in result.stdout
    assert "retained result views released after shutdown" in result.stdout
