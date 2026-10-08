# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed neural-decoder array and worker contracts

"""Exercise public neural-decoder refusals, complete outputs and worker recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("operation", ["tokens", "positions", "scaled", "gaussian", "infonce"])
@pytest.mark.parametrize("pool", ["cold", "warm"])
def test_neural_decoder_inputs_workers_and_numerical_recovery(operation: str, pool: str) -> None:
    """Verify real installed calls before, during and after address-space pressure.

    Parameters
    ----------
    operation
        Tokenisation, position encoding, attention or contrastive loss.
    pool
        Fresh global workers or workers admitted before memory pressure.
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
names = {'tokens': 'py_tokenise_spikes', 'positions': 'py_sinusoidal_position_encode',
         'scaled': 'py_scaled_dot_product_attention', 'gaussian': 'py_gaussian_attention',
         'infonce': 'py_infonce_loss'}
for name in names.values():
    assert callable(getattr(native, name))
assert engine.set_num_threads is native.set_num_threads
function = cast(Callable[..., object], getattr(native, names[operation]))
train = np.array([0, 2, 0, -1, 1, 0, 3], dtype=np.int32)
timestamps = np.array([-2., -0., 0.5, 4., 100., 1e4])
q = np.sin(np.arange(513 * 5, dtype=np.float64) * 0.17)
k = np.cos(np.arange(7 * 5, dtype=np.float64) * 0.23)
v = np.sin(np.arange(7 * 5, dtype=np.float64) * 0.31)
a = np.sin(np.arange(6 * 5, dtype=np.float64) * 0.19)
p = np.cos(np.arange(6 * 5, dtype=np.float64) * 0.11)
for array in (train, timestamps, q, k, v, a, p):
    array.setflags(write=False)
snapshots = [array.tobytes() for array in (train, timestamps, q, k, v, a, p)]
if operation == 'tokens':
    arguments: list[object] = [[train, train], -0.5]
elif operation == 'positions':
    arguments = [timestamps, 5]
elif operation in ('scaled', 'gaussian'):
    arguments = [q, k, v, 513, 7, 5]
    if operation == 'gaussian':
        arguments.append(0.75)
else:
    arguments = [a, p, 6, 5, 0.2]

def refuse(args: list[object], kind: type[Exception], text: str) -> None:
    try:
        function(*args)
    except kind as caught:
        assert str(caught) == text, (type(caught).__name__, str(caught), text)
    else:
        raise AssertionError('malformed input was accepted')

engine.set_num_threads(0)
if operation == 'tokens':
    empty = cast(tuple[NDArray[np.int64], NDArray[np.float64]], function([]))
    assert empty[0].shape == empty[1].shape == (0,)
    empty = cast(tuple[NDArray[np.int64], NDArray[np.float64]], function([train[:0]]))
    assert empty[0].size == empty[1].size == 0
    refuse([[train[::2]]], ValueError, 'trains must be contiguous')
elif operation == 'positions':
    assert np.asarray(function(timestamps, 0)).shape == (0,)
    assert np.asarray(function(timestamps[:0], 5)).shape == (0,)
    refuse([timestamps[::2], 5], ValueError, 'timestamps must be contiguous')
    refuse([timestamps, sys.maxsize], MemoryError, 'requested matrix is too large')
elif operation in ('scaled', 'gaussian'):
    tail: list[object] = [0.75] if operation == 'gaussian' else []
    assert np.asarray(function(q[:0], k[:0], v[:0], 0, 7, 5, *tail)).shape == (0,)
    assert np.asarray(function(q[:0], k[:0], v[:0], 513, 7, 0, *tail)).shape == (0,)
    for index, name in enumerate(('queries', 'keys', 'values')):
        wrong = list(arguments)
        wrong[index] = (q, k, v)[index][::2]
        refuse(wrong, ValueError, f'{name} must be contiguous')
        wrong[index] = (q, k, v)[index][:-1]
        size = 2565 if index == 0 else 35
        refuse(wrong, ValueError, f'{name} must contain at least {size} elements')
    refuse([q, k, v, sys.maxsize, 7, 5, *tail], MemoryError, 'requested matrix is too large')
    refuse([q, k, v, 513, sys.maxsize, 5, *tail], MemoryError, 'requested matrix is too large')
else:
    assert function(a[:0], p[:0], 0, 5, 0.2) == 0.0
    assert function(a[:0], p[:0], 6, 0, 0.2) == 0.0
    for index, name in enumerate(('anchors', 'positives')):
        wrong = list(arguments)
        wrong[index] = (a, p)[index][::2]
        refuse(wrong, ValueError, f'{name} must be contiguous')
        wrong[index] = (a, p)[index][:-1]
        refuse(wrong, ValueError, f'{name} must contain at least 30 elements')
    refuse([a, p, sys.maxsize, 5, 0.2], MemoryError, 'requested matrix is too large')
engine.set_num_threads(0)
if pool == 'warm':
    function(*arguments)
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
        function(*arguments)
    except MemoryError as caught:
        error = caught
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    if pool == 'cold':
        assert error is not None
        assert str(error) == 'parallel execution resources cannot be allocated'
        engine.set_num_threads(1)
    else:
        assert error is None

actual = function(*arguments)
if operation == 'tokens':
    ids, times = cast(tuple[NDArray[np.int64], NDArray[np.float64]], actual)
    events = [(uid, index * -0.5) for uid in range(2)
              for index, spike in enumerate(train.tolist()) if spike != 0]
    events.sort(key=lambda event: event[1])
    assert ids.dtype == np.int64 and times.dtype == np.float64
    np.testing.assert_array_equal(ids, [uid for uid, _ in events])
    np.testing.assert_array_equal(times, [time for _, time in events])
elif operation == 'positions':
    expected = np.array([math.sin(time / 10000.**(column / 5)) if column % 2 == 0
                         else math.cos(time / 10000.**((column - 1) / 5))
                         for time in timestamps.tolist() for column in range(5)])
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-13, atol=2e-13)
elif operation in ('scaled', 'gaussian'):
    output: list[float] = []
    for row in q.reshape(513, 5).tolist():
        scores = [(sum(x*y for x, y in zip(row, key)) / math.sqrt(5)
                   if operation == 'scaled' else
                   -sum((x-y)**2 for x, y in zip(row, key)) / (2 * 0.75**2))
                  for key in k.reshape(7, 5).tolist()]
        weights = [math.exp(score - max(scores)) for score in scores]
        denominator = sum(weights) + 1e-30
        output.extend(sum(weight * value[column] for weight, value in
                          zip(weights, v.reshape(7, 5).tolist())) / denominator
                      for column in range(5))
    np.testing.assert_allclose(np.asarray(actual), output, rtol=2e-13, atol=2e-13)
    zeros = function(q, k[:0], v[:0], 513, 0, 5,
                     *([0.75] if operation == 'gaussian' else []))
    np.testing.assert_array_equal(np.asarray(zeros), np.zeros(2565))
    extended = [np.concatenate((array, [99.])) for array in (q, k, v)]
    tail_result = function(*extended, *arguments[3:])
    np.testing.assert_array_equal(np.asarray(tail_result), np.asarray(actual))
else:
    normal_a = [[value / (math.sqrt(sum(x*x for x in row)) + 1e-30) for value in row]
                for row in a.reshape(6, 5).tolist()]
    normal_p = [[value / (math.sqrt(sum(x*x for x in row)) + 1e-30) for value in row]
                for row in p.reshape(6, 5).tolist()]
    loss = 0.0
    for index, row in enumerate(normal_a):
        scores = [sum(x*y for x, y in zip(row, positive)) / 0.2 for positive in normal_p]
        maximum = max(scores)
        loss += maximum + math.log(sum(math.exp(score-maximum) for score in scores)) - scores[index]
    assert math.isclose(cast(float, actual), loss / 6, rel_tol=2e-13, abs_tol=2e-13)
    assert function(np.concatenate((a, [99.])), np.concatenate((p, [99.])),
                    6, 5, 0.2) == actual
assert snapshots == [array.tobytes() for array in (train, timestamps, q, k, v, a, p)]
assert all(not array.flags.writeable for array in (train, timestamps, q, k, v, a, p))
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
    assert report["worker_refused"] == (report["pressure"] and pool == "cold")
