# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed SCSigmaDeltaAccumulator allocation-pressure contracts

"""Exercise real Linux memory-pressure refusal through the installed batch."""

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
"""Exercise real Linux output allocation and input-layout refusals and runtime recovery."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import sysconfig

assert sys.platform == "linux", "allocation pressure requires the Linux profile"
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
mode = int(sys.argv[1])
expected_origins = json.loads(sys.argv[2])
invalid_layout = sys.argv[3] == "1"
roots = [Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")]
origins = {}
for name, expected in expected_origins.items():
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert any(path.is_relative_to(root) for root in roots)
    assert str(path) == expected["path"] and digest == expected["sha256"]
    origins[name] = {"path": str(path), "sha256": digest}
import numpy as np
from sc_neurocore_engine import SCSigmaDeltaAccumulatorNeuron
from sc_neurocore_engine.sc_neurocore_engine import py_sc_sigma_delta_accumulator_simulate

parameters = (0.0, 1.0)
drive = np.full(1_048_576, 2.0, dtype=np.float64)
drive.setflags(write=False)
before_hash = hashlib.sha256(memoryview(drive)).hexdigest()
cell = SCSigmaDeltaAccumulatorNeuron()
control = SCSigmaDeltaAccumulatorNeuron()
assert cell.step(2.0) == control.step(2.0)
before_state = cell.get_state()
limits = resource.getrlimit(resource.RLIMIT_AS)
vms_before = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
budget = mode * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (vms_before + budget, limits[1]))
print(json.dumps({"entering": True, "additional_budget_bytes": budget,
                  "vms_before": vms_before, "origins": origins}), flush=True)
try:
    py_sc_sigma_delta_accumulator_simulate(*parameters, drive[::-1] if invalid_layout else drive)
except TypeError as refusal:
    assert invalid_layout
    assert str(refusal) == "The given array is not contiguous or is misaligned."
    result = {"exception_base": "TypeError", "message": str(refusal),
              "additional_budget_bytes": budget, "vms_before": vms_before,
              "vms_after_refusal": int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")}
except MemoryError as refusal:
    assert not invalid_layout
    result = {"exception_base": "MemoryError", "exception_type": type(refusal).__name__,
              "message": str(refusal), "failed_dtype": "int32" if mode == 10 else "float64",
              "additional_budget_bytes": budget, "vms_before": vms_before,
              "vms_after_refusal": int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")}
    assert "data type " + result["failed_dtype"] in result["message"], result
else:
    raise AssertionError("allocation must fail before advancing a million samples")
finally:
    resource.setrlimit(resource.RLIMIT_AS, limits)
assert resource.getrlimit(resource.RLIMIT_AS) == limits
assert hashlib.sha256(memoryview(drive)).hexdigest() == before_hash
assert not drive.flags.writeable
assert cell.get_state() == before_state
assert cell.step(0.2) == control.step(0.2)
assert cell.get_state() == control.get_state()
retry_drive = np.array([3.25, -4.5, 0.2, 0.0])
retry = py_sc_sigma_delta_accumulator_simulate(*parameters, retry_drive)
reference = SCSigmaDeltaAccumulatorNeuron()
trace = []
for current in retry_drive:
    event = reference.step(float(current))
    trace.append((reference.get_state()["sigma"], event))
for index, name in enumerate(("sigma", "events")):
    output = retry[name]
    np.testing.assert_array_equal(output, [row[index] for row in trace])
    assert output.dtype == (np.int32 if name == "events" else np.float64)
    assert output.flags.owndata and output.flags.writeable
    assert output.flags.c_contiguous and output.flags.aligned
    assert not np.shares_memory(output, retry_drive)
assert retry["sigma_final"] == reference.get_state()["sigma"]
assert not np.shares_memory(retry["sigma"], retry["events"])
cell.reset()
assert cell.get_state() == {"sigma": 0.0}
result.update(origins=origins, limits_restored=True, inputs_preserved=True,
              class_recovered=True, batch_recovered=True, outputs_owned=True)
print(json.dumps(result), flush=True)
'''


@pytest.mark.parametrize(("budget_mib", "invalid_layout"), [(2, False), (10, False), (2, True)])
def test_allocation_pressure_refuses_and_runtime_recovers(
    budget_mib: int, invalid_layout: bool, tmp_path: Path
) -> None:
    """Catch each real output allocation failure and retry with preserved inputs."""
    origins = installed_engine_origins()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            CONSUMER,
            str(budget_mib),
            json.dumps(origins),
            str(int(invalid_layout)),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        start_new_session=True,
        check=False,
    )
    assert result.returncode == 0, (budget_mib, result.returncode, result.stdout, result.stderr)
    observation = json.loads(result.stdout.splitlines()[-1])
    assert observation["exception_base"] == ("TypeError" if invalid_layout else "MemoryError")
    if not invalid_layout:
        assert observation["failed_dtype"] == ("int32" if budget_mib == 10 else "float64")
    for key in (
        "limits_restored",
        "inputs_preserved",
        "class_recovered",
        "batch_recovered",
        "outputs_owned",
    ):
        assert observation[key] is True
