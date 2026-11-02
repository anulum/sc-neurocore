# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real kernel resource-stop accounting controls

"""Measure a real tiny OOM and refuse fabricated resource-stop accounting."""

from __future__ import annotations

import copy
import socket
import sys

import pytest

from tests.formal_native_resource_support import measure_native_unit
from tools.formal_evidence_validation import EvidencePolicy, object_fields
from tools.formal_resource_accounting import validate_resource_measurement


@pytest.fixture(scope="module")
def oom(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], EvidencePolicy]:
    """Collect an actual finite kernel OOM kill, without any proof claim."""
    root = tmp_path_factory.mktemp("actual-kernel-oom-accounting")
    policy = EvidencePolicy(
        socket.gethostname(), 128 * 1024**2, 10, "local OOM accounting-only control"
    )
    program = """import os,signal,sys,time
from pathlib import Path
def terminate(signum, frame):
    time.sleep(0.4)
    sys.exit(1)
signal.signal(signal.SIGTERM, terminate)
pid = os.fork()
if pid == 0:
    Path('/proc/self/oom_score_adj').write_text('1000')
    chunks = []
    while True:
        chunks.append(bytearray(8 * 1024**2))
else:
    _, status = os.waitpid(pid, 0)
    print('Actual child wait status:', status, flush=True)
    time.sleep(0.4)
    sys.exit(1)
"""
    measurement = measure_native_unit(
        root, [sys.executable, "-c", program], root / "native.log", policy
    )
    assert measurement["status"] == "MEMORY", measurement
    return {"measurement": measurement}, policy


def test_real_kernel_oom_requires_native_cause_and_counts(
    oom: tuple[dict[str, object], EvidencePolicy],
) -> None:
    """A genuine OOM qualifies accounting, without qualifying a proof/model."""
    receipt, policy = oom
    status, wall, peak = validate_resource_measurement(receipt, policy)
    assert status == "MEMORY" and 0 < wall <= policy.timeout_seconds
    assert 0 < peak <= policy.memory_bytes
    measurement = object_fields(receipt["measurement"])
    assert int(str(object_fields(measurement["oom_events"])["oom_kill"])) >= 1


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("status", "TIMEOUT", "stop cause"),
        ("unit_result", "exit-code", "stop cause"),
        ("unit_exec_code", 3, "exit kind"),
        ("oom_events", {"oom": 0, "oom_kill": 0, "oom_group_kill": 0}, "no actual kernel"),
        ("oom_events", {"oom": 1, "oom_kill": 0, "oom_group_kill": 0}, "no actual kernel"),
        ("oom_events", {"oom": 0, "oom_kill": 1, "oom_group_kill": 0}, "no actual kernel"),
        ("memory_limit_bytes", 64 * 1024**3, "memory_limit_bytes"),
        ("process_tree_closed", False, "process tree"),
    ],
)
def test_oom_cause_and_custody_mutations_refuse(
    oom: tuple[dict[str, object], EvidencePolicy], field: str, value: object, reason: str
) -> None:
    """Reject changed native facts while leaving the actual original untouched."""
    original, policy = oom
    receipt = copy.deepcopy(original)
    measurement = object_fields(receipt["measurement"])
    measurement[field] = value
    receipt["measurement"] = measurement
    with pytest.raises(ValueError, match=reason):
        validate_resource_measurement(receipt, policy)
