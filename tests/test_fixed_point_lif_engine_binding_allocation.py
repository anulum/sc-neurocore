# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed fixed-point LIF allocation contracts

"""Contain historical native aborts while exercising real NumPy failures."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.engine_requirement import require_engine

require_engine()

CONSUMER = r"""
import hashlib, importlib, json, pathlib, sys
try:
    import resource
except ImportError:
    resource = None
if resource is not None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import numpy as np
import sc_neurocore_engine as engine
for name, expected in json.loads(sys.argv[2]).items():
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    assert str(path) == expected['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
assert np.dtype(np.intp).itemsize == 8, 'contract requires the 64-bit wheel profile'
mode = sys.argv[1]
maximum = int(np.iinfo(np.intp).max)
currents = np.array([-2, 3], dtype=np.int16)
currents.setflags(write=False)
neuron = engine.FixedPointLif()
assert neuron.step(0, 256, 200) == (0, 200)
before = neuron.get_state()
try:
    if mode == 'constant_dimension':
        engine.batch_lif_run(maximum + 1, 0, 0, 0)
    elif mode == 'parallel_dimension':
        engine.batch_lif_run_multi(0, maximum + 1, 0, 0, currents[:0])
    elif mode == 'constant_size':
        engine.batch_lif_run(maximum, 0, 0, 0)
    elif mode == 'parallel_size':
        engine.batch_lif_run_multi(2, maximum // 2, 0, 0, currents)
    elif mode == 'constant_memory':
        engine.batch_lif_run(maximum // 16, 0, 0, 0)
    elif mode == 'parallel_memory':
        engine.batch_lif_run_multi(2, maximum // 16, 0, 0, currents)
    else:
        raise AssertionError(mode)
except Exception as refusal:
    expected = MemoryError if mode.endswith('memory') else ValueError
    assert isinstance(refusal, expected), (type(refusal), str(refusal))
    if mode.endswith('dimension'):
        assert str(refusal) == 'array dimensions must fit numpy.intp'
    result = {'exception_base': expected.__name__,
              'exception_type': type(refusal).__name__, 'message': str(refusal)}
else:
    raise AssertionError('allocation should have refused before simulation')
assert neuron.get_state() == before
assert neuron.step(0, 256, 100) == (1, 0)
constant = engine.batch_lif_run(2, 0, 256, 3)
varying = engine.batch_lif_run_varying(0, 256, currents)
parallel = engine.batch_lif_run_multi(2, 2, 0, 256, currents)
for outputs, voltages in [(constant, [3, 6]), (varying, [-2, 1]),
                         (parallel, [[-2, -4], [3, 6]])]:
    spikes, voltage = outputs
    assert spikes.dtype == np.int32 and voltage.dtype == np.int16
    assert spikes.shape == voltage.shape
    np.testing.assert_array_equal(spikes, np.zeros(voltage.shape, dtype=np.int32))
    np.testing.assert_array_equal(voltage, voltages)
    for output in outputs:
        assert output.flags.owndata and output.flags.writeable
        assert output.flags.c_contiguous and output.flags.aligned
        assert not np.shares_memory(output, currents)
    assert not np.shares_memory(spikes, voltage)
np.testing.assert_array_equal(currents, [-2, 3])
assert not currents.flags.writeable
result['recovery_interfaces'] = ['constant', 'varying', 'parallel', 'scalar']
print(json.dumps(result))
"""


@pytest.mark.parametrize(
    "mode",
    [
        "constant_dimension",
        "parallel_dimension",
        "constant_size",
        "parallel_size",
        "constant_memory",
        "parallel_memory",
    ],
)
def test_allocation_failure_raises_and_all_interfaces_remain_usable(
    mode: str, tmp_path: Path
) -> None:
    """Real C API errors are catchable without aborting an installed consumer."""
    origins = {}
    for name in ("sc_neurocore_engine", "sc_neurocore_engine.sc_neurocore_engine"):
        module = importlib.import_module(name)
        assert module.__file__ is not None
        path = Path(module.__file__).resolve()
        origins[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", CONSUMER, mode, json.dumps(origins)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        start_new_session=True,
        check=False,
    )
    assert result.returncode == 0, (mode, result.returncode, result.stdout, result.stderr)
    observation = json.loads(result.stdout)
    expected = "MemoryError" if mode.endswith("memory") else "ValueError"
    assert observation["exception_base"] == expected
    assert observation["recovery_interfaces"] == ["constant", "varying", "parallel", "scalar"]
