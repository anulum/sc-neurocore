# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed Granger and spectral causality contracts

"""Exercise installed causality layouts, actual VAR admission and complete outputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("operation", ["pairwise", "conditional", "spectral", "pdc", "dtf"])
@pytest.mark.parametrize("pool", ["cold", "warm"])
def test_causality_inputs_fit_branches_and_resource_recovery(operation: str, pool: str) -> None:
    """Compare public decoder results and preserve inputs across real worker pressure.

    Parameters
    ----------
    operation
        Pairwise, conditional, spectral Granger, PDC or directed transfer.
    pool
        Cold worker admission or an existing global worker pool.
    """
    program = """
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Callable, cast
import numpy as np
from numpy.typing import NDArray
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native
origins = cast(dict[str, dict[str, str]], json.loads(sys.argv[1]))
for module in (engine, native):
    path = Path(cast(str, module.__file__)).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
operation, pool = sys.argv[2:]
names = {'pairwise': 'py_pairwise_granger_causality',
         'conditional': 'py_conditional_granger_causality',
         'spectral': 'py_spectral_granger_causality',
         'pdc': 'py_partial_directed_coherence', 'dtf': 'py_directed_transfer_function'}
function = cast(Callable[..., object], getattr(native, names[operation]))
rng = np.random.default_rng(42)
source = rng.integers(0, 2, 505, dtype=np.int32)
target = np.roll(source, 5) + rng.integers(0, 2, 505, dtype=np.int32)
condition = rng.integers(0, 3, 505, dtype=np.int32)
arrays = (source, target, condition)
for array in arrays:
    array.setflags(write=False)
snapshots = [array.tobytes() for array in arrays]
parallel = operation in ('spectral', 'pdc', 'dtf')
arguments: list[object] = [[source, target, condition], 5, 2, 7] if parallel else (
    [source, target, condition, 5, 2] if operation == 'conditional' else [source, target, 5, 2])
def refuse(args: list[object], kind: type[Exception], text: str) -> None:
    try:
        function(*args)
    except kind as caught:
        assert str(caught) == text
    else:
        raise AssertionError('invalid causality input accepted')
engine.set_num_threads(0)
if parallel:
    empty = cast(tuple[NDArray[np.float64], int, int], function([], 0, 0, 7))
    assert empty[0].shape == (0,) and empty[1:] == (0, 7)
    for index in range(3):
        bad = list(arrays)
        bad[index] = arrays[index][::2]
        refuse([bad, 5, 2, 7], ValueError, 'trains must be contiguous')
    refuse([list(arrays), 0, 2, 7], ValueError, 'bin_size must be positive')
    refuse([list(arrays), 5, 0, 7], ValueError,
           'order must be positive for a fitted VAR model')
    refuse([[source, target[:-5], condition], 5, 2, 7], ValueError,
           "each train must contain at least the first train's binned length")
    refuse([list(arrays), 5, 2, sys.maxsize], MemoryError, 'requested matrix is too large')
    refuse([list(arrays), 5, sys.maxsize, 7], MemoryError, 'requested matrix is too large')
    for order in (0, 2):
        fallback = cast(tuple[NDArray[np.float64], int, int],
                        function([source[:3], target[:0]], 5, order, 7))
        expected_fallback = np.zeros((2, 2, 7))
        if operation != 'spectral':
            expected_fallback[0, 0] = expected_fallback[1, 1] = 1.
        np.testing.assert_array_equal(fallback[0].reshape(2, 2, 7), expected_fallback)
else:
    n_inputs = 3 if operation == 'conditional' else 2
    for index, name in enumerate(('source', 'target', 'condition')[:n_inputs]):
        bad = list(arguments)
        bad[index] = arrays[index][::2]
        refuse(bad, ValueError, f'{name} must be contiguous')
    refuse([*arrays[:n_inputs], 0, 2], ValueError, 'bin_size must be positive')
    assert function(*arrays[:n_inputs], 5, 0) == 0.0
    assert function(*arrays[:n_inputs], 5, sys.maxsize) == 0.0
    assert function(*[array[:3] for array in arrays[:n_inputs]], 5, 2) == 0.0
engine.set_num_threads(0)
if pool == 'warm':
    engine.set_num_threads(1)
pressure = sys.platform == 'linux'
error: MemoryError | None = None
if pressure:
    import resource
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    virtual = int(next(line.split()[1]
                       for line in Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('VmSize:'))) * 1024
    previous = resource.getrlimit(resource.RLIMIT_AS)
    limit = virtual + 1024**2
    assert previous[1] == resource.RLIM_INFINITY or limit <= previous[1]
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
    try:
        if parallel and pool == 'cold':
            refuse([list(arrays), 5, 2, 0], MemoryError,
                   'parallel execution resources cannot be allocated')
        function(*arguments)
    except MemoryError as caught:
        error = caught
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    assert (error is not None) == (parallel and pool == 'cold')
    if error is not None:
        assert str(error) == 'parallel execution resources cannot be allocated'
    if pool == 'cold':
        engine.set_num_threads(1)
actual = function(*arguments)
binned = np.array([array.reshape(101, 5).sum(axis=1) for array in arrays], dtype=np.float64)
if parallel:
    design = np.array([[binned[ch, time-lag-1] for lag in range(2) for ch in range(3)]
                       for time in range(2, 101)])
    response = binned[:, 2:].T
    beta = np.linalg.solve(design.T @ design + np.eye(6) * 1e-8, design.T @ response)
    residual = response - design @ beta
    sigma = residual.T @ residual / 99
    expected = np.zeros((3, 3, 7))
    for frequency in range(7):
        f = frequency / 14
        matrix = np.eye(3, dtype=np.complex128)
        for lag in range(2):
            matrix -= beta[lag*3:(lag+1)*3].T * np.exp(-2j * np.pi * f * (lag+1))
        transfer = np.linalg.inv(matrix)
        if operation == 'pdc':
            expected[:, :, frequency] = np.abs(matrix) / np.linalg.norm(matrix, axis=0)
        elif operation == 'dtf':
            expected[:, :, frequency] = np.abs(transfer) / np.linalg.norm(
                transfer, axis=1)[:, None]
        else:
            spectrum = transfer @ sigma @ transfer.conj().T
            for i in range(3):
                for j in range(3):
                    if i != j:
                        sii = abs(spectrum[i, i])
                        reduced = sii - sigma[j, j] * abs(transfer[i, j])**2
                        if sii > 1e-30 and 0 < reduced < sii:
                            expected[i, j, frequency] = max(0., math.log(sii/reduced))
    values, d, n_freqs = cast(tuple[NDArray[np.float64], int, int], actual)
    assert values.dtype == np.float64 and values.shape == (63,) and (d, n_freqs) == (3, 7)
    np.testing.assert_allclose(values.reshape(3, 3, 7), expected, rtol=2e-10, atol=2e-12)
    longer = [source, np.concatenate((target, [0, 1, 0, 1, 0])).astype(np.int32),
              np.concatenate((condition, [2, 0, 1, 0, 2])).astype(np.int32)]
    extended = cast(tuple[NDArray[np.float64], int, int], function(longer, 5, 2, 7))
    np.testing.assert_array_equal(extended[0], values)
    zero = cast(tuple[NDArray[np.float64], int, int], function(list(arrays), 5, 2, 0))
    assert zero[0].shape == (0,) and zero[1:] == (3, 0)
else:
    def sse(channels: list[int]) -> float:
        design = np.array([[binned[ch, time-lag-1] for ch in channels for lag in range(2)]
                           for time in range(2, 101)])
        response = binned[1, 2:]
        beta = np.linalg.solve(design.T @ design + np.eye(len(channels)*2) * 1e-8,
                               design.T @ response)
        residual = response - design @ beta
        return float(residual @ residual)
    restricted = sse([1, 2] if operation == 'conditional' else [1])
    full = sse([1, 2, 0] if operation == 'conditional' else [1, 0])
    expected_scalar = 0.0 if full <= 0 else math.log(max(restricted, 1e-30)/max(full, 1e-30))
    assert math.isclose(cast(float, actual), expected_scalar, rel_tol=2e-10, abs_tol=2e-12)
assert snapshots == [array.tobytes() for array in arrays]
assert all(not array.flags.writeable for array in arrays)
print(json.dumps({'operation': operation, 'pool': pool, 'pressure': pressure,
                  'worker_refused': error is not None, 'all_inputs_unchanged': True}))
"""
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            program,
            json.dumps(installed_engine_origins()),
            operation,
            pool,
        ],
        cwd=os.environ.get("TMPDIR", os.getcwd()),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "RAYON_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(process.stdout)
    assert report["operation"] == operation and report["pool"] == pool
    assert report["all_inputs_unchanged"]
    assert report["worker_refused"] == (
        report["pressure"] and pool == "cold" and operation in ("spectral", "pdc", "dtf")
    )
