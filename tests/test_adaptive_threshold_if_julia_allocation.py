# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia adaptive-threshold allocation contracts

"""Translate actual Julia trace allocation exhaustion and recover a complete batch."""

from __future__ import annotations

import os
import subprocess
import sys


def test_julia_trace_memory_error_preserves_drive_and_recovers() -> None:
    """Exercise the maintained facade after warming the real Julia error classifier."""
    program = r"""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from sc_neurocore.accel.adaptive_threshold_if import simulate_python
from sc_neurocore.accel.julia.neurons.adaptive_threshold_if import simulate_adaptive_threshold_if
from sc_neurocore.accel.julia.neurons._runtime import JULIA_MAIN

parameters = (-63.5, -52.5, -68., -67., -49., 4.5, 8., 42., .05)
simulate_adaptive_threshold_if(*parameters, np.zeros(3))
assert JULIA_MAIN is not None
assert JULIA_MAIN.AdaptiveThresholdIFAccel.is_memory_error(JULIA_MAIN.OutOfMemoryError())
drive = np.full(300_000, 1.25)
drive.setflags(write=False)
before = hashlib.sha256(memoryview(drive)).hexdigest()
pressure = sys.platform == 'linux'
if pressure:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    previous = resource.getrlimit(resource.RLIMIT_AS)
    virtual = int(Path('/proc/self/statm').read_text().split()[0])*os.sysconf('SC_PAGE_SIZE')
    budget = 8*1024**2
    limit = virtual + budget
    assert previous[1] == resource.RLIM_INFINITY or limit <= previous[1]
    print(json.dumps({'phase': 'entering', 'virtual_bytes': virtual,
                      'headroom_bytes': budget, 'trace_bytes': drive.nbytes}), flush=True)
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
    try:
        simulate_adaptive_threshold_if(*parameters, drive)
    except MemoryError as refused:
        assert type(refused) is MemoryError
        assert str(refused) == 'adaptive-threshold trace buffers cannot be allocated'
    else:
        raise AssertionError('real Julia trace allocation must refuse pressure')
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    assert resource.getrlimit(resource.RLIMIT_AS) == previous
assert hashlib.sha256(memoryview(drive)).hexdigest() == before
assert not drive.flags.writeable
actual = simulate_adaptive_threshold_if(*parameters, drive[:128])
expected = simulate_python(*parameters, drive[:128])
for key in ('v', 'theta', 'spikes', 'v_final', 'theta_final', 'spike_count'):
    np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
assert hashlib.sha256(memoryview(drive)).hexdigest() == before
print(json.dumps({'pressure': pressure, 'drive_preserved': True, 'recovery': True}), flush=True)
"""
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "JULIA_NUM_THREADS": "1",
        "JULIA_PKG_OFFLINE": "true",
    }
    environment.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=os.environ.get("TMPDIR", os.getcwd()),
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, (process.returncode, process.stdout, process.stderr)
