# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed Brunel allocation-pressure contracts

"""Exercise actual Linux constructor and simulation memory refusal with recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()

CONSUMER = r'''
"""Refuse real allocations without losing input bytes or seeded continuation."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys

assert sys.platform == "linux", "allocation pressure requires the Linux profile"
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
mode = sys.argv[1]
expected_origins = json.loads(sys.argv[2])
origins = {}
for name, expected in expected_origins.items():
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert str(path) == expected["path"] and digest == expected["sha256"]
    origins[name] = {"path": str(path), "sha256": digest}
import numpy as np
from sc_neurocore_engine import FixedPointBrunelNetwork

def make_network(count):
    """Build an actual recurrent or disconnected public network and retain its inputs."""
    if count == 4:
        rows = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        columns = np.array([1, 2, 3, 0], dtype=np.int64)
        weights = np.full(4, 128, dtype=np.int16)
    else:
        rows = np.zeros(count + 1, dtype=np.int64)
        columns = np.empty(0, dtype=np.int64)
        weights = np.empty(0, dtype=np.int16)
    network = FixedPointBrunelNetwork(
        count, rows, columns, weights, 20, 256, 0.7, 128, seed=17)
    return network, (rows, columns, weights)

count = 1_048_576 if mode == "working" else 4
network, network_inputs = make_network(count)
control, control_inputs = make_network(count)
np.testing.assert_array_equal(network.run(1), control.run(1))
expected_next = control.run(4)
candidate_count = 1_048_576 if mode in ("rows", "neurons") else 1
if mode in ("indices", "weights"):
    rows = np.array([0, 1_048_576], dtype=np.int64)
    columns = np.zeros(1_048_576, dtype=np.int64)
    weights = np.zeros(1_048_576, dtype=np.int16)
else:
    rows = np.zeros(candidate_count + 1, dtype=np.int64)
    columns = np.empty(0, dtype=np.int64)
    weights = np.empty(0, dtype=np.int16)
arrays = network_inputs + control_inputs + (rows, columns, weights)
for array in arrays:
    array.setflags(write=False)
before = [hashlib.sha256(memoryview(array)).hexdigest() for array in arrays]
limits = resource.getrlimit(resource.RLIMIT_AS)
vms = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
held_buffers = []
if mode == "working":
    buffer_bytes = count * np.dtype(np.int32).itemsize
    held_buffers = [np.empty(count, dtype=np.int32) for _ in range(vms // buffer_bytes + 1)]
    vms = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
budget = {"weights": 9, "neurons": 10}.get(mode, 2) * 1024 * 1024
requested_steps = vms // 2 if mode == "output" else 4
if mode == "output":
    assert requested_steps * np.dtype(np.uint32).itemsize > vms + budget
print(json.dumps({"entering": True, "mode": mode, "vms_before": vms,
                  "additional_budget_bytes": budget, "origins": origins,
                  "held_buffer_count": len(held_buffers),
                  "held_buffer_bytes": sum(array.nbytes for array in held_buffers)}), flush=True)
resource.setrlimit(resource.RLIMIT_AS, (vms + budget, limits[1]))
try:
    if mode in ("output", "working"):
        network.run(requested_steps)
    else:
        FixedPointBrunelNetwork(
            candidate_count, rows, columns, weights, 20, 256, 0.0, 128, seed=17)
except MemoryError as refusal:
    message = str(refusal)
    assert message == "Brunel simulation buffers cannot be allocated", message
else:
    raise AssertionError("real allocation pressure must refuse")
finally:
    resource.setrlimit(resource.RLIMIT_AS, limits)
assert resource.getrlimit(resource.RLIMIT_AS) == limits
assert [hashlib.sha256(memoryview(array)).hexdigest() for array in arrays] == before
assert all(not array.flags.writeable for array in arrays)
assert network.run(0).shape == (0,)
np.testing.assert_array_equal(network.run(4), expected_next)
retry, retry_inputs = make_network(4)
fresh, fresh_inputs = make_network(4)
np.testing.assert_array_equal(retry.run(8), fresh.run(8))
print(json.dumps({"mode": mode, "exception": "MemoryError", "message": message,
                  "limits_restored": True, "inputs_preserved": True,
                  "seeded_continuation_preserved": True, "constructor_recovered": True,
                  "origins": origins}), flush=True)
'''


@pytest.mark.parametrize("mode", ["output", "working", "rows", "indices", "weights", "neurons"])
def test_allocation_refusal_preserves_inputs_and_runtime(mode: str, tmp_path: Path) -> None:
    """Refuse actual allocation pressure and continue through the installed public API.

    Parameters
    ----------
    mode : str
        Output, working storage or constructor allocation profile.
    tmp_path : Path
        Owned directory for the isolated child process.
    """
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", CONSUMER, mode, json.dumps(installed_engine_origins())],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
        start_new_session=True,
        check=False,
    )
    assert result.returncode == 0, (mode, result.returncode, result.stdout, result.stderr)
    observation = json.loads(result.stdout.splitlines()[-1])
    assert observation["mode"] == mode
    assert observation["exception"] == "MemoryError"
    assert observation["message"] == "Brunel simulation buffers cannot be allocated"
    for key in (
        "limits_restored",
        "inputs_preserved",
        "seeded_continuation_preserved",
        "constructor_recovered",
    ):
        assert observation[key] is True
