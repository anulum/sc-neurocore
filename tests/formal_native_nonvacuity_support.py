# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual separately measured native enable-audit fixture

"""Collect a real enable audit with its own finite cgroup and native accounting."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time
import uuid
from typing import TYPE_CHECKING

from tools.formal_evidence_validation import digest, object_fields, read_object
from tools.formal_log_normalization import NORMALIZER_ID, normalize_native_log

if TYPE_CHECKING:
    from tests.formal_native_evidence_support import NativeEvidence

_AUDIT = """
import json, sys
from pathlib import Path
from tools.formal_nonvacuity import check_task, native_configuration
r = Path(sys.argv[1])
primary = r / ('proof' if (r / 'proof/status').is_file() else 'proof_default')
report = check_task(primary, r / 'enable-audit')
configuration = native_configuration((primary / 'config.sby').read_text())
(r / 'native-audit.json').write_text(json.dumps({'configuration': configuration, 'report': report}))
raise SystemExit(0 if report['verdict'] == 'NONVACUOUS' else 2)
"""


def collect_native_audit(evidence: NativeEvidence) -> None:
    """Measure the public native enable check separately from its real primary.

    Parameters
    ----------
    evidence : NativeEvidence
        Successful primary fixture whose original prepared model is retained.
    """
    from tests.formal_native_evidence_support import _state

    root, policy = evidence.repository, evidence.policy
    hosted = os.environ.get("GITHUB_ACTIONS") == "true"
    control = ["sudo", "systemctl"] if hosted else ["systemctl", "--user"]
    launcher = ["sudo", "systemd-run"] if hosted else ["systemd-run", "--user"]
    unit = "scn-formal-enable-test-" + uuid.uuid4().hex + ".service"
    command = [
        *launcher,
        "--unit=" + unit,
        *(["--uid=" + str(os.getuid()), "--gid=" + str(os.getgid())] if hosted else []),
        "--property=WorkingDirectory=" + str(Path(__file__).resolve().parents[1]),
        "--property=MemoryMax=" + str(policy.memory_bytes),
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=" + str(policy.timeout_seconds) + "s",
        "--property=TimeoutStopSec=5s",
        "--property=KillMode=control-group",
        "--property=MemoryAccounting=yes",
        "--property=RemainAfterExit=yes",
        "--property=CPUQuota=100%",
        "--property=Nice=19",
        "--property=StandardOutput=append:" + str(root / "audit-unit.log"),
        "--property=StandardError=append:" + str(root / "audit-unit.log"),
        "/usr/bin/env",
        *[
            key + "=" + os.environ[key]
            for key in (
                "PATH",
                "PYTHONPATH",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONPYCACHEPREFIX",
                "TMPDIR",
            )
            if key in os.environ
        ],
        sys.executable,
        "-c",
        _AUDIT,
        str(root),
    ]
    cgroup: Path | None = None
    peak = 0
    events: dict[str, int] = {}
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        deadline = time.monotonic() + policy.timeout_seconds + 5
        while True:
            state = _state(control, unit)
            if state["ControlGroup"]:
                cgroup = Path("/sys/fs/cgroup") / state["ControlGroup"].lstrip("/")
            if cgroup is not None and (cgroup / "memory.peak").is_file():
                peak = max(peak, int((cgroup / "memory.peak").read_text()))
                events = dict(
                    (key, int(value))
                    for key, value in (
                        line.split() for line in (cgroup / "memory.events").read_text().splitlines()
                    )
                )
            if state["SubState"] in ("exited", "failed", "dead"):
                break
            assert time.monotonic() < deadline, "Enable audit exceeded its actual finite ceiling."
            time.sleep(0.05)
        assert state["Result"] == "success" and state["ExecMainStatus"] == "0", (
            root / "audit-unit.log"
        ).read_text()
        assert state["MainPID"] == "0"
        assert (
            cgroup is None
            or not (cgroup / "cgroup.procs").exists()
            or not (cgroup / "cgroup.procs").read_text().split()
        )
        start, stop = (
            int(state[name])
            for name in ("ExecMainStartTimestampMonotonic", "ExecMainExitTimestampMonotonic")
        )
        measurement: dict[str, object] = {
            "hostname": policy.hostname,
            "unit": unit,
            "status": "PASS",
            "wall_clock_source": "systemd-exec-monotonic",
            "start_monotonic_microseconds": start,
            "stop_monotonic_microseconds": stop,
            "wall_seconds": (stop - start) / 1_000_000,
            "unit_accounting": "retained",
            "native_exit_code": 0,
            "native_exit_source": "systemd-exec",
            "unit_exit_code": 0,
            "unit_result": state["Result"],
            "main_pid": int(state["MainPID"]),
            "cgroup_processes": [],
            "process_tree_closed": True,
            "peak_memory_bytes": peak,
            "memory_limit_bytes": int(state["MemoryMax"]),
            "swap_limit_bytes": int(state["MemorySwapMax"]),
            "time_limit_seconds": policy.timeout_seconds,
            "oom_events": {name: events[name] for name in ("oom", "oom_kill", "oom_group_kill")},
            "invocation_id": state["InvocationID"],
        }
    finally:
        subprocess.run([*control, "stop", unit], capture_output=True, check=True, timeout=15)
        closing = _state(control, unit)
        assert closing["ActiveState"] in ("inactive", "failed") and closing["MainPID"] == "0"
        assert (
            cgroup is None
            or not (cgroup / "cgroup.procs").exists()
            or not (cgroup / "cgroup.procs").read_text().split()
        )
    measurement["active_state"] = closing["ActiveState"]
    task = read_object(root / "native-audit.json")
    report = object_fields(task["report"])
    task["model_custody"] = {
        "origin": "retained original",
        "primary_sha256": digest(
            root
            / ("proof" if (root / "proof/status").is_file() else "proof_default")
            / "model/design_prep.il"
        ),
        "audited_sha256": report["design_prep_sha256"],
    }
    raw_log = root / ("native.log" if report["mode"] == "cover" else "enable-audit/cover.log")
    public_log = root / "public-audit.log"
    public_log.write_bytes(
        normalize_native_log(
            raw_log.read_bytes(),
            {
                "workspace": str(root),
                "unit": unit,
                "hostname": policy.hostname,
                "username": pwd.getpwuid(os.getuid()).pw_name,
            },
        )
    )
    task["log"] = {
        "path": public_log.name,
        "raw_sha256": digest(raw_log),
        "normalized_sha256": digest(public_log),
        "normalizer_id": NORMALIZER_ID,
        "normalizer_sha256": digest(
            Path(__file__).resolve().parents[1] / "tools/formal_log_normalization.py"
        ),
    }
    receipt = read_object(root / "receipt.json")
    artifact = {
        "schema_version": 1,
        "producer": "actual separately measured native enable-audit fixture",
        "receipt_sha256": digest(root / "receipt.json"),
        "source_hashes": receipt["source_hashes"],
        "toolchain": receipt["toolchain"],
        "measurement": measurement,
        "tasks": {"default": task},
    }
    (root / "nonvacuity.json").write_text(json.dumps(artifact, indent=2) + "\n")


def bind_audit(root: Path) -> dict[str, object]:
    """Bind explicit test-only review metadata to an actual measured audit.

    Parameters
    ----------
    root : Path
        Isolated fixture evidence, never a production evidence directory.

    Returns
    -------
    dict of str to object
        Separate artifact and review references for native contract controls.
    """
    artifact_path = root / "nonvacuity.json"
    artifact = read_object(artifact_path)
    artifact["receipt_sha256"] = digest(root / "receipt.json")
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    models = {
        name: object_fields(object_fields(task)["report"])["design_prep_sha256"]
        for name, task in object_fields(artifact["tasks"]).items()
    }
    review = {
        "schema_version": 1,
        "verdict": "CLEAR",
        "reviewer": "native enable-audit contract fixture; not a production attestation",
        "artifact_sha256": digest(artifact_path),
        "receipt_sha256": artifact["receipt_sha256"],
        "source_hashes": artifact["source_hashes"],
        "prepared_model_sha256": models,
    }
    review_path = root / "nonvacuity-review.json"
    review_path.write_text(json.dumps(review, indent=2) + "\n")
    return {
        "nonvacuity": artifact_path.name,
        "nonvacuity_sha256": digest(artifact_path),
        "nonvacuity_review": review_path.name,
        "nonvacuity_review_sha256": digest(review_path),
    }
