# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed Kuramoto worker startup and phase recovery

"""Exercise cold Kuramoto workers, input admission and phase recovery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("surface", ["native", "facade"])
@pytest.mark.parametrize("method", ["step", "run", "step_ssgf", "run_ssgf"])
def test_worker_refusal_preserves_phases_and_recovers(surface: str, method: str) -> None:
    """Retain phases after actual worker refusal and match the Euler equations.

    Parameters
    ----------
    surface
        Installed extension or the maintained Python wrapper.
    method
        Baseline or SSGF stepping entry point.
    """
    program = """
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import cast
import numpy as np
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native
import sc_neurocore_engine.scpn as scpn
origins = cast(dict[str, dict[str, str]], json.loads(sys.argv[1]))
for module in (engine, native, scpn):
    filename = module.__file__
    assert isinstance(filename, str)
    path = Path(filename).resolve()
    expected_origin = origins[module.__name__]
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == expected_origin['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_origin['sha256']
assert engine.KuramotoSolver is scpn.KuramotoSolver
surface, method = sys.argv[2:]
constructor = native.KuramotoSolver if surface == 'native' else scpn.KuramotoSolver
n = 16
omega = np.ones(n)
coupling = np.full((n, n), 0.3)
initial = np.arange(n, dtype=np.float64) * 0.3
w = np.ones((n, n))
h = np.roll(np.eye(n), 1, axis=1)
for matrix in (coupling, w, h):
    matrix.setflags(write=False)
input_bytes = [matrix.tobytes() for matrix in (coupling, w, h)]
solver = constructor(omega.tolist(), coupling, initial.tolist(), noise_amp=0.0)
solver.set_field_pressure(0.25)
before = np.asarray(solver.phases, dtype=np.float64).tobytes()
engine.set_num_threads(0)
assert len(solver.run(0, 0.01, seed=42)) == 0
assert len(solver.run_ssgf(0, 0.01, seed=42, W=w, sigma_g=0.5,
                           h_munu=h, pgbo_weight=0.2)) == 0
assert np.asarray(solver.phases, dtype=np.float64).tobytes() == before
operation = getattr(solver, method)
is_run = method.startswith('run')
is_ssgf = method.endswith('_ssgf')
arguments: dict[str, object] = {'seed': 42}
if is_ssgf:
    arguments.update(W=w, sigma_g=0.5, h_munu=h, pgbo_weight=0.2)
pressure = sys.platform == 'linux'
if pressure:
    import resource
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    virtual = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('VmSize:'))) * 1024
    previous = resource.getrlimit(resource.RLIMIT_AS)
    limit = virtual + 1024**2
    assert previous[1] == resource.RLIM_INFINITY or limit <= previous[1]
    error: MemoryError | None = None
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
    try:
        try:
            if is_run:
                operation(4, 0.0, **arguments)
            else:
                operation(0.0, **arguments)
        except ValueError as caught:
            assert str(caught) == 'dt must be finite and positive'
        else:
            raise AssertionError('invalid timestep was accepted')
        if is_run:
            operation(4, 0.01, **arguments)
        else:
            operation(0.01, **arguments)
    except MemoryError as caught:
        error = caught
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
    assert np.asarray(solver.phases, dtype=np.float64).tobytes() == before
    assert [matrix.tobytes() for matrix in (coupling, w, h)] == input_bytes
    assert all(not matrix.flags.writeable for matrix in (coupling, w, h))
if is_run:
    actual = operation(4, 0.01, **arguments)
    assert isinstance(actual, list if surface == 'native' else np.ndarray)
else:
    actual = operation(0.01, **arguments)
    assert isinstance(actual, float)
expected_phases = initial.copy()
expected_orders = []
for _ in range(4 if is_run else 1):
    sin_diff = np.sin(expected_phases[None, :] - expected_phases[:, None])
    derivative = omega + (coupling * sin_diff).sum(axis=1) / n
    if is_ssgf:
        derivative += 0.5 * (w * sin_diff).sum(axis=1)
        derivative += 0.2 * (h * sin_diff).sum(axis=1)
        derivative += 0.25 * np.cos(expected_phases)
    expected_phases = np.remainder(expected_phases + derivative * 0.01, 2 * np.pi)
    expected_orders.append(float(abs(np.exp(1j * expected_phases).mean())))
expected = expected_orders if is_run else expected_orders[-1]
np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
np.testing.assert_allclose(solver.phases, expected_phases, rtol=0.0, atol=1e-14)
assert [matrix.tobytes() for matrix in (coupling, w, h)] == input_bytes
assert all(not matrix.flags.writeable for matrix in (coupling, w, h))
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('successful default startup did not fix the global pool')
print(json.dumps({'surface': surface, 'method': method,
                  'pressure_exercised': pressure, 'phase_recovery': True}))
"""
    origins = installed_engine_origins()
    wrapper = Path(origins["sc_neurocore_engine"]["path"]).parent / "scpn.py"
    assert wrapper.is_relative_to(Path(sys.prefix).resolve())
    origins["sc_neurocore_engine.scpn"] = {
        "path": str(wrapper),
        "sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
    }
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    environment.pop("RAYON_NUM_THREADS", None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, json.dumps(origins), surface, method],
        capture_output=True,
        text=True,
        env=environment,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    observation = json.loads(result.stdout.splitlines()[-1])
    assert observation == {
        "surface": surface,
        "method": method,
        "pressure_exercised": sys.platform == "linux",
        "phase_recovery": True,
    }
