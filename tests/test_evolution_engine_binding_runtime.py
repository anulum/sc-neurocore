# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed evolutionary worker refusal and serial selection

"""Exercise evolutionary public resource admission and retained population formulas."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize(
    "operation", ["mutate", "fitness", "crossover", "diversity", "novelty", "tournament"]
)
def test_evolution_worker_refusal_or_serial_selection_recovery(operation: str) -> None:
    """Preserve whole populations across real worker refusal and unchanged serial selection.

    Parameters
    ----------
    operation
        One of the six registered population mutation, selection or metric functions.
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
origins = cast(dict[str, dict[str, str]], json.loads(sys.argv[1]))
for module in (engine, native):
    filename = module.__file__
    assert isinstance(filename, str)
    path = Path(filename).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
engine.set_num_threads(0)
operation = sys.argv[2]
pressure = sys.platform == 'linux'
for name in ('py_evo_batch_mutate', 'py_evo_batch_fitness', 'py_evo_batch_crossover',
             'py_evo_diversity', 'py_evo_novelty', 'py_evo_tournament'):
    assert getattr(engine, name) is getattr(native, name)
assert native.py_evo_batch_mutate([]) == []
assert native.py_evo_batch_fitness([], [], 1.) == []
assert native.py_evo_batch_crossover([[1.]], []) == []
assert native.py_evo_batch_crossover([], [[1.]]) == []
assert native.py_evo_diversity([]) == 0.
assert native.py_evo_diversity([[1.]]) == 0.
assert native.py_evo_novelty([], [[1.]], k_nearest=2) == []
assert native.py_evo_tournament([], 0) == []
assert native.py_evo_tournament([0.5], 8, tournament_size=3, seed=42) == [0] * 8
population = [[0., 0., 0.], [3., 4.], [3., 4., 12.], []]
archive = [[0., 0.], [6., 8.]]
inputs = [1., -2., 0.5, 100.]
parents_a, parents_b = [[0., 0., 0.], [1., 1.], [2.]], [[0., 0., 0.], [1., 1., 1.]]
before = json.dumps([population, archive, inputs, parents_a, parents_b])
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
        native.py_evo_tournament([], 1, tournament_size=0, seed=42)
    except ValueError as caught:
        assert str(caught) == 'fitness must be nonempty when selecting winners'
    else:
        raise AssertionError('nonzero selection from an empty population was accepted')
    if pressure:
        if operation == 'mutate':
            native.py_evo_batch_mutate(population, mutation_rate=0.5, mutation_scale=0.1, seed=42)
        elif operation == 'fitness':
            native.py_evo_batch_fitness(population, inputs, 1.)
        elif operation == 'crossover':
            native.py_evo_batch_crossover(parents_a, parents_b, seed=42)
        elif operation == 'diversity':
            native.py_evo_diversity(population)
        elif operation == 'novelty':
            native.py_evo_novelty(population, archive, k_nearest=2)
        else:
            assert native.py_evo_tournament([0.5], 8, tournament_size=0, seed=42) == [0] * 8
except MemoryError as caught:
    error = caught
finally:
    if pressure:
        resource.setrlimit(resource.RLIMIT_AS, previous)
if pressure and operation != 'tournament':
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
else:
    assert error is None
if operation == 'mutate':
    result = native.py_evo_batch_mutate(population, mutation_rate=0.5, mutation_scale=0.1, seed=42)
    np.testing.assert_array_equal([len(row) for row in result], [len(row) for row in population])
    assert result == native.py_evo_batch_mutate(population, mutation_rate=0.5, mutation_scale=0.1, seed=42)
    assert native.py_evo_batch_mutate(population, mutation_rate=0.) == population
    clipped = native.py_evo_batch_mutate([[-100., -2., 0., 100.]], mutation_rate=1., mutation_scale=0.)
    assert clipped == [[-10., -2., 0., 10.]]
elif operation == 'fitness':
    result = native.py_evo_batch_fitness(population, inputs, 1.)
    expected = [-float(np.dot(np.array(row), np.array(inputs[:len(row)])) - 1.) ** 2 for row in population]
    np.testing.assert_array_equal(result, expected)
    assert native.py_evo_batch_fitness([[]], inputs, 2.) == [-4.]
elif operation == 'crossover':
    result = native.py_evo_batch_crossover(parents_a, parents_b, seed=42)
    assert result == [[0., 0., 0.], [1., 1.]]
    other = [[10., 11., 12.], [13., 14., 15.]]
    mixed = native.py_evo_batch_crossover(parents_a, other, seed=42)
    assert mixed == native.py_evo_batch_crossover(parents_a, other, seed=42)
    for i, row in enumerate(mixed):
        for j, value in enumerate(row):
            assert value in (parents_a[i][j], other[i][j])
elif operation == 'diversity':
    result = native.py_evo_diversity(population)
    distances = [float(np.linalg.norm(np.array(population[i][:min(len(population[i]), len(population[j]))])
                                    - np.array(population[j][:min(len(population[i]), len(population[j]))])))
                 for i in range(len(population)) for j in range(i + 1, len(population))]
    assert result == sum(distances) / len(distances) == 3.
    assert native.py_evo_diversity([[], []]) == 0.
elif operation == 'novelty':
    result = native.py_evo_novelty(population, archive, k_nearest=2)
    expected_scores = []
    for i, row in enumerate(population):
        others = archive + [other for j, other in enumerate(population) if i != j]
        distances = sorted(float(np.linalg.norm(np.array(row[:min(len(row), len(other))])
                                               - np.array(other[:min(len(row), len(other))])))
                           for other in others)
        expected_scores.append(sum(distances[:2]) / min(2, len(distances)))
    np.testing.assert_array_equal(result, expected_scores)
    assert native.py_evo_novelty(population, archive, k_nearest=0) == [0.] * len(population)
    assert native.py_evo_novelty([[1.]], [], k_nearest=5) == [0.]
else:
    fitness = [0.1, 0.9, 0.5, 0.3]
    winners = native.py_evo_tournament(fitness, 10, tournament_size=3, seed=42)
    assert len(winners) == 10 and all(0 <= i < len(fitness) for i in winners)
    assert winners == native.py_evo_tournament(fitness, 10, tournament_size=3, seed=42)
    assert native.py_evo_tournament(fitness, 10, tournament_size=0, seed=42) == native.py_evo_tournament(fitness, 10, tournament_size=1, seed=42)
    assert fitness == [0.1, 0.9, 0.5, 0.3]
assert json.dumps([population, archive, inputs, parents_a, parents_b]) == before
if operation == 'tournament':
    engine.set_num_threads(1)
else:
    try:
        engine.set_num_threads(2)
    except ValueError as caught:
        assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
    else:
        raise AssertionError('successful parallel operation did not fix the pool')
print(json.dumps({'operation': operation, 'pressure_exercised': pressure,
                  'worker_refusal': pressure and operation != 'tournament', 'recovery': True}))
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
        "worker_refusal": sys.platform == "linux" and operation != "tournament",
        "recovery": True,
    }
