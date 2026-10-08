# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed patterns array and allocation contracts

"""Exercise installed patterns shapes, checked output and worker recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("operation", ["direction", "order", "cumulant"])
@pytest.mark.parametrize("pool", ["cold", "warm"])
def test_patterns_inputs_allocation_and_worker_recovery(operation: str, pool: str) -> None:
    """Compare public decoder results and preserve inputs across real worker pressure.

    Parameters
    ----------
    operation
        Directionality, train ordering or third-order cumulants.
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
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native
origins = cast(dict[str, dict[str, str]], json.loads(sys.argv[1]))
for module in (engine, native):
    path = Path(cast(str, module.__file__)).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
operation, pool = sys.argv[2:]
def refuse(args: list[object], text: str) -> None:
    try:
        function(*args)
    except ValueError as caught:
        assert str(caught) == text
    else:
        raise AssertionError('invalid array accepted')
engine.set_num_threads(0)

names = {'direction': 'py_spike_directionality', 'order': 'py_spike_train_order',
         'cumulant': 'py_cubic_higher_order'}
function = cast(Callable[..., object], getattr(native, names[operation]))
a = np.array([0.7, 0.1, 0.5, 0.3, 0.1, 2., -0.4])
b = np.array([0.75, 0.15, 0.55, 0.35, 1.2, -0.1])
c = np.array([0.8, 0.2, 0.6, 0.4])
train = (np.arange(65, dtype=np.int32) * 7 % 11 == 0).astype(np.int32)
arrays = (a, b, c, train)
for array in arrays:
    array.setflags(write=False)
before = [array.tobytes() for array in arrays]
if operation == 'direction':
    assert function(a[:0], b) == 0.0
    refuse([a[::2], b], 'times_a must be contiguous')
    refuse([a, b[::2]], 'times_b must be contiguous')
    arguments: list[object] = [a, b, 0., 1.]
    large_arguments: list[object] = []
elif operation == 'order':
    assert np.asarray(function([])).shape == (0, 0)
    np.testing.assert_array_equal(np.asarray(function([a])), np.zeros((1, 1)))
    for index in range(3):
        bad = [a, b, c]
        bad[index] = bad[index][::2]
        refuse([bad], 'times_list must be contiguous')
    arguments = [[a, b, c], 0., 1.]
    large_arguments = [[a[:0]] * 1025, 0., 1.]
else:
    assert np.asarray(function(train, 0.001, 0)).shape == (0, 0)
    np.testing.assert_array_equal(np.asarray(function(train[:0], 0.001, 8)), np.zeros((8, 8)))
    refuse([train[::2]], 'binary_train must be contiguous')
    try:
        function(train, 0.001, sys.maxsize)
    except MemoryError as caught:
        assert str(caught) == 'requested matrix is too large'
    else:
        raise AssertionError('lag capacity overflow was accepted')
    arguments = [train, 0.001, 8]
    large_arguments = [train, 0.001, 1025]

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
        if operation != 'direction':
            try:
                function(*large_arguments)
            except MemoryError as caught:
                assert type(caught).__name__ == 'MemoryError'
                assert type(caught).__module__.startswith('numpy')
            else:
                raise AssertionError('large checked NumPy output unexpectedly allocated')
        function(*arguments)
    except MemoryError as caught:
        error = caught
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    assert (error is not None) == (pool == 'cold' and operation != 'direction')
    if error is not None:
        assert str(error) == 'parallel execution resources cannot be allocated'
    if pool == 'cold':
        engine.set_num_threads(1)
actual = function(*arguments)

def direction(left: list[float], right: list[float]) -> float:
    import bisect
    first = sorted(value for value in left if 0 <= value <= 1)
    second = sorted(value for value in right if 0 <= value <= 1)
    ahead = behind = 0
    for value in first:
        index = bisect.bisect_left(second, value)
        if 0 < index < len(second):
            if value - second[index-1] < second[index] - value:
                behind += 1
            else:
                ahead += 1
    return (ahead - behind) / (ahead + behind) if ahead + behind else 0.0
if operation == 'direction':
    assert actual == direction(a.tolist(), b.tolist())
    assert function(b, a) == direction(b.tolist(), a.tolist())
    assert function(a, b, 1., 0.) == 0.0
elif operation == 'order':
    expected = np.zeros((3, 3))
    times = [a.tolist(), b.tolist(), c.tolist()]
    for i in range(3):
        for j in range(i+1, 3):
            expected[i, j] = direction(times[i], times[j])
            expected[j, i] = -expected[i, j]
    np.testing.assert_array_equal(np.asarray(actual), expected)
else:
    def cumulant(samples: list[int], lag: int) -> list[list[float]]:
        mean = sum(samples) / len(samples)
        centered = [sample - mean for sample in samples]
        return [[sum(centered[index] * centered[index+i] * centered[index+j]
                     for index in range(len(samples)-max(i,j))) / (len(samples)-max(i,j))
                 if len(samples) > max(i,j) else 0.0 for j in range(lag)] for i in range(lag)]
    np.testing.assert_allclose(np.asarray(actual), cumulant(train.tolist(), 8),
                               rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(np.asarray(function(train[:3], 0.001, 8)),
                               cumulant(train[:3].tolist(), 8), rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(np.asarray(function(train, 9., 8)), np.asarray(actual))
if operation != 'direction':
    value = np.asarray(actual)
    assert value.dtype == np.float64 and value.flags.owndata
    assert value.flags.c_contiguous and value.flags.writeable

assert before == [array.tobytes() for array in arrays]
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
        report["pressure"] and pool == "cold" and operation != "direction"
    )
