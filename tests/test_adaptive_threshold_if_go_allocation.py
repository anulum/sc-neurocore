# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go adaptive-threshold allocation contracts

"""Exercise the real Go C export and Python facade under allocation pressure."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["abi", "facade"])
@pytest.mark.parametrize("reservation", [0, 1, 2])
def test_go_trace_refusal_preserves_caller_buffers_and_recovers(
    mode: str, reservation: int
) -> None:
    """Refuse each real trace reservation and recover the complete public result.

    Parameters
    ----------
    mode
        Published C ABI or its maintained Python facade.
    reservation
        First, second or third trace reservation under Linux address-space pressure.
    """
    module = importlib.import_module("sc_neurocore.accel.go.adaptive_threshold_if")
    assert module.__file__ is not None
    facade = Path(module.__file__).resolve()
    library = (facade.parent / "libadaptive_threshold_if.so").resolve()
    assert library.is_file(), "the real Go adaptive-threshold shared library is required"
    expected = {
        "facade": str(facade),
        "facade_sha256": hashlib.sha256(facade.read_bytes()).hexdigest(),
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    program = r"""
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from sc_neurocore.accel.adaptive_threshold_if import simulate_python

expected = json.loads(sys.argv[1])
mode, reservation = sys.argv[2], int(sys.argv[3])
module = importlib.import_module('sc_neurocore.accel.go.adaptive_threshold_if')
assert module.__file__ is not None
facade = Path(module.__file__).resolve()
library_path = (facade.parent/'libadaptive_threshold_if.so').resolve()
assert str(facade) == expected['facade']
assert hashlib.sha256(facade.read_bytes()).hexdigest() == expected['facade_sha256']
assert str(library_path) == expected['library']
assert hashlib.sha256(library_path.read_bytes()).hexdigest() == expected['library_sha256']
library = ctypes.CDLL(str(library_path))
function = library.adaptive_threshold_if_simulate_c
function.argtypes = [ctypes.c_int32, *([ctypes.c_double]*9), *([ctypes.c_void_p]*7)]
function.restype = ctypes.c_int32
parameters = (-63.5, -52.5, -68., -67., -49., 4.5, 8., 42., .05)
module.simulate_adaptive_threshold_if(*parameters, np.zeros(3))
drive = np.full(300_000, 1.25)
drive.setflags(write=False)
buffers = [drive, *(np.full(drive.size, -999.) for _ in range(3)),
           *(np.full(1, -999.) for _ in range(3))]
before = [hashlib.sha256(memoryview(a)).hexdigest() for a in buffers]
pressure = sys.platform == 'linux'
if pressure:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    previous = resource.getrlimit(resource.RLIMIT_AS)
    virtual = int(Path('/proc/self/statm').read_text().split()[0])*os.sysconf('SC_PAGE_SIZE')
    budget = ((1, 3, 5) if mode == 'abi' else (8, 10, 12))[reservation]*1024**2
    limit = virtual + budget
    assert previous[1] == resource.RLIM_INFINITY or limit <= previous[1]
    print(json.dumps({'phase': 'entering', 'mode': mode, 'headroom_bytes': budget,
                      'trace_bytes': drive.nbytes}), flush=True)
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
    try:
        if mode == 'abi':
            assert function(drive.size, *parameters, *(a.ctypes.data for a in buffers)) == 5
        else:
            try:
                module.simulate_adaptive_threshold_if(*parameters, drive)
            except MemoryError as refused:
                assert type(refused) is MemoryError
                assert str(refused) == 'adaptive-threshold trace buffers cannot be allocated'
            else:
                raise AssertionError('Go trace reservation must refuse pressure')
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    assert resource.getrlimit(resource.RLIMIT_AS) == previous
assert [hashlib.sha256(memoryview(a)).hexdigest() for a in buffers] == before
assert not drive.flags.writeable
golden = simulate_python(*parameters, drive[:128])
actual = module.simulate_adaptive_threshold_if(*parameters, drive[:128])
for key in ('v', 'theta', 'spikes'):
    np.testing.assert_allclose(actual[key], golden[key], rtol=0, atol=1e-12)
for key in ('v_final', 'theta_final'):
    np.testing.assert_allclose(actual[key], golden[key], rtol=0, atol=1e-12)
assert actual['spike_count'] == golden['spike_count']
assert function(128, *parameters, *(a.ctypes.data for a in buffers)) == 0
for a, key in zip(buffers[1:4], ('v', 'theta', 'spikes'), strict=True):
    np.testing.assert_allclose(a[:128], golden[key], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(a[128:], np.full(drive.size-128, -999.))
np.testing.assert_allclose(buffers[4][0], golden['v_final'], rtol=0, atol=1e-12)
np.testing.assert_allclose(buffers[5][0], golden['theta_final'], rtol=0, atol=1e-12)
assert buffers[6][0] == golden['spike_count']
assert hashlib.sha256(memoryview(drive)).hexdigest() == before[0]
print(json.dumps({'mode': mode, 'reservation': reservation, 'pressure': pressure,
                  'caller_bytes_preserved': True, 'recovery': True}), flush=True)
"""
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "GOMAXPROCS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    environment.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, json.dumps(expected), mode, str(reservation)],
        cwd=os.environ.get("TMPDIR", os.getcwd()),
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, (mode, reservation, process.stdout, process.stderr)
    observation = json.loads(process.stdout.splitlines()[-1])
    assert observation == {
        "mode": mode,
        "reservation": reservation,
        "pressure": sys.platform == "linux",
        "caller_bytes_preserved": True,
        "recovery": True,
    }
