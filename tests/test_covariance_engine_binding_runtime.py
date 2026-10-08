# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed covariance array and allocation contracts

"""Exercise installed covariance shapes, checked output and worker recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("operation", ["covariance", "count_covariance"])
@pytest.mark.parametrize("pool", ["cold", "warm"])
def test_covariance_inputs_allocation_and_worker_recovery(operation: str, pool: str) -> None:
    """Compare public decoder results and preserve inputs across real worker pressure.

    Parameters
    ----------
    operation
        Covariance and windowed spike-count covariance.
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

function = cast(Callable[..., object], getattr(native,
    'py_covariance_matrix' if operation == 'covariance' else 'py_spike_count_covariance'))
rng = np.random.default_rng(317)
arrays = tuple(rng.integers(0, 4, length, dtype=np.int32) for length in (2567, 2566, 2565))
for array in arrays:
    array.setflags(write=False)
before = [array.tobytes() for array in arrays]
np.testing.assert_array_equal(np.asarray(function([], 0)), np.empty((1, 0)))
np.testing.assert_array_equal(np.asarray(function([arrays[0][:0]], 5)), np.zeros((1, 1)))
for index in range(3):
    bad = list(arrays)
    bad[index] = arrays[index][::2]
    refuse([bad, 5], 'trains must be contiguous')
refuse([list(arrays), 0], 'bin_size must be positive')
arguments: list[object] = [list(arrays), 5]
empty = np.empty(0, dtype=np.int32)
large_arguments: list[object] = [[empty] * 1025, 5]

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

counts = np.array([array[:2565].reshape(513, 5).sum(axis=1) for array in arrays], dtype=np.float64)
centered = counts - counts.mean(axis=1)[:, None]
expected = centered @ centered.T / 512
value = np.asarray(actual)
assert value.shape == (3, 3) and value.dtype == np.float64
assert value.flags.owndata and value.flags.c_contiguous and value.flags.writeable
np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)
np.testing.assert_array_equal(value, value.T)
np.testing.assert_allclose(np.asarray(function([arrays[0]], 5)), expected[:1, :1],
                           rtol=2e-12, atol=2e-12)
np.testing.assert_array_equal(np.asarray(function([array[:3] for array in arrays], 5)),
                              np.zeros((3, 3)))
np.testing.assert_array_equal(np.asarray(function([empty, empty], 5)), np.zeros((2, 2)))

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
