# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed network worker admission and state recovery

"""Exercise cold network workers, empty work and neuron-state recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("method", ["step_population", "run"])
def test_worker_refusal_preserves_network_and_serial_recovery(method: str) -> None:
    """Refuse cold workers before advancing a real heterogeneous population.

    Parameters
    ----------
    method
        Population stepping or whole-network execution.
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
assert engine.NetworkRunner is native.NetworkRunner
method = sys.argv[2]
runner = native.NetworkRunner()
assert runner.add_population('Izhikevich', 3) == 0
assert runner.add_population('AdEx', 2) == 1
initial = runner.run(0)
before = [row.tobytes() for row in initial['voltages']]
currents = np.full(3, 15.0)
currents.setflags(write=False)
input_bytes = currents.tobytes()
engine.set_num_threads(0)
empty = native.NetworkRunner()
assert empty.run(4)['voltages'] == []
empty.add_population('Izhikevich', 0)
assert empty.step_population(0, np.empty(0))['voltages'].size == 0
assert empty.run(4)['voltages'][0].size == 0
assert [row.tobytes() for row in runner.run(0)['voltages']] == before
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
            runner.step_population(2, currents)
        except ValueError as caught:
            assert str(caught) == 'population index 2 out of range'
        else:
            raise AssertionError('missing population was accepted')
        try:
            runner.step_population(1, currents)
        except ValueError as caught:
            assert str(caught) == 'current vector length mismatch: got 3, expected 2'
        else:
            raise AssertionError('wrong current vector shape was accepted')
        if method == 'step_population':
            runner.step_population(0, currents)
        else:
            runner.run(4)
    except MemoryError as caught:
        error = caught
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
    assert [row.tobytes() for row in runner.run(0)['voltages']] == before
    assert currents.tobytes() == input_bytes and not currents.flags.writeable
if method == 'step_population':
    serial = native.py_batch_simulate('Izhikevich', 4, np.full(4, 15.0))
    voltage_rows = []
    spike_rows = []
    for _ in range(4):
        result = runner.step_population(0, currents)
        assert result['voltages'].dtype == np.float64
        assert result['spikes'].dtype == np.uint8
        voltage_rows.append(result['voltages'].copy())
        spike_rows.append(result['spikes'].copy())
    expected_voltages = np.repeat(serial['voltages'][:, None], 3, axis=1)
    expected_spikes = np.zeros((4, 3), dtype=np.uint8)
    expected_spikes[serial['spikes'], :] = 1
    np.testing.assert_array_equal(voltage_rows, expected_voltages)
    np.testing.assert_array_equal(spike_rows, expected_spikes)
    assert runner.run(0)['voltages'][1].tobytes() == before[1]
else:
    serial_results = [native.py_batch_simulate(model, 4, np.zeros(4))
                      for model in ('Izhikevich', 'AdEx')]
    result = runner.run(4)
    for index, (size, serial) in enumerate(zip((3, 2), serial_results, strict=True)):
        assert result['voltages'][index].dtype == np.float64
        np.testing.assert_array_equal(result['voltages'][index],
                                      np.full(size, serial['voltages'][-1]))
        assert result['spike_counts'][index] == size * len(serial['spikes'])
        expected_events = [(neuron << 32) | int(step)
                           for step in serial['spikes'] for neuron in range(size)]
        assert result['spike_data'][index].tolist() == expected_events
assert currents.tobytes() == input_bytes and not currents.flags.writeable
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('successful network startup did not fix the global pool')
print(json.dumps({'method': method, 'pressure_exercised': pressure,
                  'state_recovery': True}))
"""
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    environment.pop("RAYON_NUM_THREADS", None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, json.dumps(installed_engine_origins()), method],
        capture_output=True,
        text=True,
        env=environment,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    observation = json.loads(result.stdout.splitlines()[-1])
    assert observation == {
        "method": method,
        "pressure_exercised": sys.platform == "linux",
        "state_recovery": True,
    }
