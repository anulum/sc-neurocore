# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed population and statistical decoding contracts

"""Exercise installed decoder shapes, legacy defaults and numerical worker recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("operation", ["population", "bayesian", "likelihood", "fisher", "naive"])
@pytest.mark.parametrize("pool", ["cold", "warm"])
def test_decoding_inputs_defaults_and_resource_recovery(operation: str, pool: str) -> None:
    """Compare public decoder results and preserve inputs across real worker pressure.

    Parameters
    ----------
    operation
        Population vector, MAP, likelihood, Fisher or Gaussian naive Bayes.
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
names = {'population': 'py_population_vector_decode', 'bayesian': 'py_bayesian_decode',
         'likelihood': 'py_maximum_likelihood_decode', 'fisher': 'py_linear_discriminant_decode',
         'naive': 'py_naive_bayes_decode'}
function = cast(Callable[..., object], getattr(native, names[operation]))
assert native.set_num_threads is engine.set_num_threads
train = (np.arange(2567, dtype=np.int32) * 13 % 17) - 3
train_b = (np.arange(2566, dtype=np.int32) * 5 % 13) - 2
train_c = (np.arange(2565, dtype=np.int32) * 7 % 19) - 4
directions = np.array([0., 0.7])
counts = np.array([2., 7., 0., 3., 1.])
rates = np.array([0., -1., 0.1, 0.2, 0.3, 2., 7., 0.2, 3., 1., 1., 1., 1., 1., 1.])
prior = np.array([0.2, 0.8])
data = np.array([[-4., -3.], [-3.8, -3.1], [-4.2, -2.9],
                 [0.5, 0.8], [0.6, 0.7], [0.4, 0.9],
                 [8., 7.], [8.1, 7.2], [7.9, 6.8]]).ravel()
labels = np.repeat(np.array([-7, 12, 1200], dtype=np.int64), 3)
point = np.array([7.8, 7.1])
arrays = (train, train_b, train_c, directions, counts, rates, prior, data, labels, point)
for array in arrays:
    array.setflags(write=False)
snapshots = [array.tobytes() for array in arrays]
if operation == 'population':
    arguments: list[object] = [[train, train_b, train_c], directions, 5]
elif operation in ('bayesian', 'likelihood'):
    arguments = [counts, rates, 3, 5]
    if operation == 'bayesian':
        arguments.append(prior)
else:
    arguments = [data, 9, 2, labels, point]

def refuse(args: list[object], kind: type[Exception], text: str) -> None:
    try:
        function(*args)
    except kind as caught:
        assert str(caught) == text
    else:
        raise AssertionError('malformed decoder input accepted')

engine.set_num_threads(0)
if operation == 'population':
    assert np.asarray(function([], directions)).shape == (0,)
    assert np.asarray(function([train], directions, 0)).shape == (0,)
    assert np.asarray(function([train[:3]], directions, 5)).shape == (0,)
    refuse([[train[::2]], directions, 5], ValueError, 'trains must be contiguous')
    refuse([[train], np.arange(4, dtype=np.float64)[::2], 5],
           ValueError, 'preferred_directions must be contiguous')
elif operation in ('bayesian', 'likelihood'):
    optional: list[object] = [prior] if operation == 'bayesian' else []
    assert function(counts[:0], rates[:0], 0, 5, *optional) == 0
    assert function(counts[:0], rates[:0], 3, 0, *optional) == 0
    for index, name, size in [(0, 'spike_counts', 5), (1, 'tuning_rates', 15)]:
        bad = list(arguments)
        bad[index] = (counts, rates)[index][::2]
        refuse(bad, ValueError, f'{name} must be contiguous')
        bad[index] = (counts, rates)[index][:-1]
        refuse(bad, ValueError, f'{name} must contain at least {size} elements')
    refuse([counts, rates, sys.maxsize, 5, *optional],
           MemoryError, 'requested matrix is too large')
    if operation == 'bayesian':
        refuse([counts, rates, 3, 5, np.arange(4, dtype=np.float64)[::2]],
               ValueError, 'prior must be contiguous')
else:
    assert function(data[:0], 0, 2, labels[:0], point[:0]) == 0
    assert function(data[:0], 9, 0, labels[:0], point[:0]) == 0
    for index, name, array, size in [(0, 'train_data', data, 18),
                                   (3, 'labels', labels, 9), (4, 'test_point', point, 2)]:
        bad = list(arguments)
        if index == 4:
            bad[index] = np.arange(4, dtype=np.float64)[::2]
        else:
            bad[index] = array[::2]
        refuse(bad, ValueError, f'{name} must be contiguous')
        bad[index] = array[:-1]
        refuse(bad, ValueError, f'{name} must contain at least {size} elements')
    refuse([data, 9, sys.maxsize, labels, point],
           MemoryError, 'requested matrix is too large')
    if operation == 'fisher':
        single = np.full(9, -7, dtype=np.int64)
        assert function(data[:0], 9, 2, single, point[:0]) == -7
        assert function(data[:0], 9, sys.maxsize, single, point[:0]) == -7
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
        function(*arguments)
    except MemoryError as caught:
        error = caught
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    if pool == 'cold' and operation != 'naive':
        assert error is not None
        assert str(error) == 'parallel execution resources cannot be allocated'
        engine.set_num_threads(1)
    else:
        assert error is None
        if pool == 'cold':
            engine.set_num_threads(1)
actual = function(*arguments)
if operation == 'population':
    expected: list[float] = []
    for start in range(0, 2565, 5):
        sums = [sum(int(x) for x in samples[start:start+5])
                for samples in (train, train_b, train_c)]
        angles = [0., 0.7, 0.]
        expected.append(math.atan2(sum(count * math.sin(angle)
                                       for count, angle in zip(sums, angles)),
                                   sum(count * math.cos(angle)
                                       for count, angle in zip(sums, angles))))
    result = np.asarray(actual)
    assert result.shape == (513,) and result.dtype == np.float64
    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(np.asarray(function(arguments[0], np.array([0., 0.7, 0.]), 5)),
                                  result)
elif operation in ('bayesian', 'likelihood'):
    def expected_stimulus(priors: list[float]) -> int:
        scores = []
        for stimulus, row in enumerate(rates.reshape(3, 5).tolist()):
            score = -math.log(3) if not priors else math.log(
                (priors[stimulus] if stimulus < len(priors) else 1e-30) + 1e-30)
            for count, rate in zip(counts.tolist(), row):
                lam = max(rate, 1e-10)
                score += count * math.log(lam) - lam
            scores.append(score)
        assert len(set(scores)) == 3
        return max(range(3), key=scores.__getitem__)
    assert actual == expected_stimulus(prior.tolist() if operation == 'bayesian' else [])
    if operation == 'bayesian':
        for selected in (None, np.array([]), np.array([0.2]), np.array([0.2, 0.8, 0.1, 99.])):
            assert function(counts, rates, 3, 5, selected) == expected_stimulus(
                [] if selected is None else selected.tolist())
    assert function(np.concatenate((counts, [99.])), np.concatenate((rates, [99.])),
                    *arguments[2:]) == actual
else:
    samples = data.reshape(9, 2)
    classes = [-7, 12, 1200]
    if operation == 'fisher':
        means = np.array([samples[index:index+3].mean(axis=0) for index in (0, 3, 6)])
        scatter = np.zeros((2, 2))
        for index, mean in zip((0, 3, 6), means):
            for row in samples[index:index+3]:
                delta = row - mean
                scatter += np.outer(delta, delta)
        scatter += np.eye(2) * 1e-8
        weights = np.linalg.solve(scatter, (means - samples.mean(axis=0)).T)
        for probe in (point, np.array([-8., -6.]), np.array([1., 0.])):
            expected_class = classes[int(np.argmax(weights.T @ probe))]
            assert function(data, 9, 2, labels, probe) == expected_class
    else:
        for probe in (point, np.array([-4., -3.]), np.array([0.5, 0.8])):
            scores = []
            for index in (0, 3, 6):
                population = samples[index:index+3]
                mean = population.mean(axis=0)
                variance = ((population - mean)**2).mean(axis=0) + 1e-10
                score = math.log(1/3) + sum(-0.5 * (math.log(2*math.pi*var)
                           + (value - mu)**2/var) for value, mu, var in
                           zip(probe.tolist(), mean.tolist(), variance.tolist()))
                scores.append(score)
            assert function(data, 9, 2, labels, probe) == classes[int(np.argmax(scores))]
    assert function(np.concatenate((data, [99.])), 9, 2,
                    np.concatenate((labels, [99])), np.concatenate((point, [99.]))) == actual
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
        report["pressure"] and pool == "cold" and operation != "naive"
    )
