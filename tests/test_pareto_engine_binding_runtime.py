# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed Pareto worker admission and recovery

"""Exercise Pareto input admission, empty work and cold worker recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


def test_pareto_worker_refusal_retains_columns_and_frontier() -> None:
    """Restore the real operation after worker refusal without changing its inputs."""
    program = """
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import cast
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
pressure = sys.platform == 'linux'

assert engine.py_opt_extract_pareto is native.py_opt_extract_pareto
assert native.py_opt_extract_pareto([], [], [])['indices'] == []
luts = [100, 200, 150, 250]
power = [1.0, 0.5, 0.8, 1.2, float('nan')]
score = [0.9, 0.95, 0.85, 0.7, float('nan')]
before = (list(luts), list(power[:4]), list(score[:4]))
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
    for bad_power, bad_score in ((power[:3], score), (power, score[:3])):
        try:
            native.py_opt_extract_pareto(luts, bad_power, bad_score)
        except ValueError as caught:
            assert str(caught) == 'power and score must cover every LUT entry'
        else:
            raise AssertionError('short Pareto column was accepted')
    if pressure:
        native.py_opt_extract_pareto(luts, power, score)
except MemoryError as caught:
    error = caught
finally:
    if pressure:
        resource.setrlimit(resource.RLIMIT_AS, previous)
if pressure:
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
actual = native.py_opt_extract_pareto(luts, power, score)
indices = [i for i in range(len(luts)) if not any(
    j != i and luts[j] <= luts[i] and power[j] <= power[i] and score[j] >= score[i]
    and (luts[j] < luts[i] or power[j] < power[i] or score[j] > score[i])
    for j in range(len(luts)))]
assert indices == [0, 1, 2]
assert actual == {'backend': 'rust', 'indices': indices,
                  'luts': [luts[i] for i in indices],
                  'power': [power[i] for i in indices],
                  'score': [score[i] for i in indices]}
assert native.py_opt_extract_pareto(luts, power, score) == actual
assert (luts, power[:4], score[:4]) == before
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('successful Pareto startup did not fix the pool')
print(json.dumps({'pressure_exercised': pressure, 'input_recovery': True}))
"""
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    environment.pop("RAYON_NUM_THREADS", None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, json.dumps(installed_engine_origins())],
        capture_output=True,
        text=True,
        env=environment,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "pressure_exercised": sys.platform == "linux",
        "input_recovery": True,
    }
