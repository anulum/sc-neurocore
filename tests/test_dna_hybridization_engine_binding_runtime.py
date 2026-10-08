# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed DNA hybridization worker admission and recovery

"""Exercise real strand-pair scoring after cold worker refusal."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


def test_hybridization_worker_refusal_retains_strands_and_scores() -> None:
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

assert engine.py_dna_check_cross_hybridization is native.py_dna_check_cross_hybridization
assert native.py_dna_check_cross_hybridization([], 2) == []
assert native.py_dna_check_cross_hybridization(['ATCGATCG'], 2) == []
sequences = ['ATCGATCG', 'CGATCGAT', 'AAAATTTT']
before = list(sequences)
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
        native.py_dna_check_cross_hybridization(sequences, 2)
    except MemoryError as caught:
        error = caught
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
    assert sequences == before
actual = native.py_dna_check_cross_hybridization(sequences, 2)
expected = []
translation = str.maketrans('ATCG', 'TAGC')
for i in range(len(sequences)):
    for j in range(i + 1, len(sequences)):
        a, b = sequences[i], sequences[j].translate(translation)[::-1]
        best = 0
        for start in range(len(a)):
            for end in range(start + 1, len(a) + 1):
                if a[start:end] in b:
                    best = max(best, end - start)
        if best >= 2:
            expected.append({'strand_a': i, 'strand_b': j, 'score': best})
assert actual == expected
assert native.py_dna_check_cross_hybridization(sequences, 2) == actual
assert sequences == before
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('successful hybridization startup did not fix the pool')
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
