# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed Ising and annealing worker recovery

"""Exercise Ising batches, annealing samples and gauges after worker refusal."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("operation", ["batch", "anneal", "gauges"])
def test_quantum_worker_refusal_preserves_model_and_recovery(operation: str) -> None:
    """Recover actual numerical Ising operations with the original inputs and seed.

    Parameters
    ----------
    operation
        Parallel energy batch, simulated annealing or gauge generation.
    """
    program = """
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import cast
import numpy as np
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native
origins = cast(dict[str, dict[str, str]], json.loads(sys.argv[1]))
for module in (engine, native):
    filename = module.__file__
    assert isinstance(filename, str)
    path = Path(filename).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
operation = sys.argv[2]
for name in ('py_qa_batch_ising_energy', 'py_qa_simulated_annealing', 'py_qa_generate_gauges'):
    assert getattr(engine, name) is getattr(native, name)
engine.set_num_threads(0)
assert native.py_qa_batch_ising_energy([], [], [], [], [], []) == []
assert native.py_qa_generate_gauges(4, n_gauges=0, seed=42) == []
empty = native.py_qa_simulated_annealing([], [], [], [], [], 4, num_reads=0, seed=42)
assert empty['samples'] == [] and empty['energies'] == []
assert empty['best_spins'] == [1, 1, 1, 1] and math.isinf(empty['best_energy'])
h_indices, h_values = [0, 1], [0.5, -0.25]
j_i, j_j, j_values = [0, 1, 2], [1, 2, 3], [-1.0, 0.5, -0.75]
configs = [[1 if (bits >> qubit) & 1 else -1 for qubit in range(4)] for bits in range(16)]
before = [list(x) for x in (h_indices, h_values, j_i, j_j, j_values)]
if operation == 'batch':
    function = native.py_qa_batch_ising_energy
    arguments = [h_indices, h_values, j_i, j_j, j_values, configs]
    keywords: dict[str, object] = {'offset': 0.125}
elif operation == 'anneal':
    function = native.py_qa_simulated_annealing
    arguments = [h_indices, h_values, j_i, j_j, j_values, 4]
    keywords = {'offset': 0.125, 'n_sweeps': 8, 'num_reads': 4, 'seed': 42}
else:
    function = native.py_qa_generate_gauges
    arguments = [4]
    keywords = {'n_gauges': 4, 'seed': 42}
pressure = sys.platform == 'linux'
if pressure:
    import resource
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    virtual = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('VmSize:'))) * 1024
    previous = resource.getrlimit(resource.RLIMIT_AS)
    limit = virtual + 1024**2
    assert previous[1] == resource.RLIM_INFINITY or limit <= previous[1]
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
error: MemoryError | None = None
try:
    try:
        native.py_qa_simulated_annealing([], [], [0], [4], [1.0], 4,
                                       n_sweeps=8, num_reads=4, seed=42)
    except ValueError as caught:
        assert str(caught) == 'Ising coupling references an out-of-range qubit'
    else:
        raise AssertionError('active out-of-range annealing edge was accepted')
    if pressure:
        function(*arguments, **keywords)
except MemoryError as caught:
    error = caught
finally:
    if pressure:
        resource.setrlimit(resource.RLIMIT_AS, previous)
if pressure:
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
assert [list(x) for x in (h_indices, h_values, j_i, j_j, j_values)] == before
actual = function(*arguments, **keywords)
assert function(*arguments, **keywords) == actual
if operation == 'batch':
    spins = np.array(configs)
    expected = 0.125 + spins[:, h_indices] @ np.array(h_values)
    expected += (spins[:, j_i] * spins[:, j_j]) @ np.array(j_values)
    np.testing.assert_array_equal(actual, expected)
elif operation == 'anneal':
    spins = np.array(actual['samples'])
    assert spins.shape == (4, 4) and np.all((spins == 1) | (spins == -1))
    expected = 0.125 + spins[:, h_indices] @ np.array(h_values)
    expected += (spins[:, j_i] * spins[:, j_j]) @ np.array(j_values)
    np.testing.assert_array_equal(actual['energies'], expected)
    best = int(np.argmin(expected))
    assert actual['best_spins'] == actual['samples'][best]
    assert actual['best_energy'] == expected[best]
else:
    gauges = np.array(actual)
    assert gauges.shape == (4, 4) and np.all((gauges == 1) | (gauges == -1))
assert [list(x) for x in (h_indices, h_values, j_i, j_j, j_values)] == before
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('successful numerical startup did not fix the pool')
print(json.dumps({'operation': operation, 'pressure_exercised': pressure,
                  'model_recovery': True}))
"""
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    environment.pop("RAYON_NUM_THREADS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            program,
            json.dumps(installed_engine_origins()),
            operation,
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "operation": operation,
        "pressure_exercised": sys.platform == "linux",
        "model_recovery": True,
    }
