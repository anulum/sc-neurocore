# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual bounded native formal evidence fixture

"""Produce a real small SBY result; never synthesize a successful proof verdict."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import pwd
from pathlib import Path
import shutil
import socket
import subprocess
import time
import uuid

from tools.formal_evidence_validation import (
    EvidencePolicy,
    digest,
    object_fields,
    read_object,
    source_pins,
)
from tools.formal_job_inputs import resolve_job
from tools.formal_log_normalization import NORMALIZER_ID, normalize_native_log
from tests.formal_native_nonvacuity_support import bind_audit, collect_native_audit

ROOT = Path(__file__).resolve().parents[1]
JOB = Path("hdl/formal/catalogue/sc_mccullochpittsneuron.sby")


@dataclass
class NativeEvidence:
    """Retain real result files and the policy used by their qualification.

    Parameters
    ----------
    repository : Path
        Copied unchanged tracked job and collected native evidence.
    policy : EvidencePolicy
        Actual local test machine and bounded service limits.
    provenance : str
        Actual SDK provenance.
    solvers : dict of str to Path
        Actual solver executables.
    """

    repository: Path
    policy: EvidencePolicy
    provenance: str
    solvers: dict[str, Path]

    def rebind(self) -> dict[str, object]:
        """Bind a test mutation without hiding it behind an unrelated digest failure.

        Returns
        -------
        dict of str to object
            Index entry for this test-only case. This fixture metadata is
            explicitly not an independent production attestation.
        """
        receipt = self.repository / "receipt.json"
        review = self.repository / "review.json"
        data = read_object(receipt)
        log = object_fields(data["log"])
        review.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verdict": "CLEAR",
                    "reviewer": "receipt contract fixture; not a production attestation",
                    "receipt_sha256": digest(receipt),
                    "raw_log_sha256": log["raw_sha256"],
                    "normalized_log_sha256": log["normalized_sha256"],
                    "normalizer_id": log["normalizer_id"],
                    "normalizer_sha256": log["normalizer_sha256"],
                    "source_hashes": data["source_hashes"],
                    "execution_exception": None,
                    "exception_registry_sha256": None,
                }
            )
        )
        return {
            "receipt": "receipt.json",
            "receipt_sha256": digest(receipt),
            "review": "review.json",
            "review_sha256": digest(review),
            **bind_audit(self.repository),
        }


def _tool(name: str) -> Path:
    """Find an actual required native executable.

    Parameters
    ----------
    name : str
        Executable name.

    Returns
    -------
    Path
        Available native executable.
    """
    value = shutil.which(name)
    assert value is not None, f"Native formal evidence requires {name}."
    return Path(value)


def _state(control: list[str], unit: str) -> dict[str, str]:
    """Read actual primary-service accounting and closure fields.

    Parameters
    ----------
    control : list of str
        System or user systemctl invocation.
    unit : str
        Owned transient service.

    Returns
    -------
    dict of str to str
        Actual native service properties.
    """
    fields = (
        "ActiveState",
        "SubState",
        "MainPID",
        "Result",
        "ExecMainStatus",
        "ControlGroup",
        "MemoryPeak",
        "InvocationID",
        "ExecMainStartTimestampMonotonic",
        "ExecMainExitTimestampMonotonic",
        "MemoryMax",
        "MemorySwapMax",
    )
    command = [*control, "show", unit, *[f"--property={name}" for name in fields]]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def collect_native(
    repository: Path, *, cover_mode: bool = False, named_default: bool = False
) -> NativeEvidence:
    """Prove an unchanged tiny tracked job inside a real finite cgroup.

    Parameters
    ----------
    repository : Path
        Empty owned fixture directory.
    cover_mode : bool, optional
        Run the same reset property as a real native reachability task.
    named_default : bool, optional
        Use a declared task named default with task-conditional options.

    Returns
    -------
    NativeEvidence
        Real PASS, log, actual monotonic wall, cgroup peak and closed process tree.
        Its 128 MiB, sixty-second local policy must fail the production policy.
    """
    directory = repository / JOB.parent
    directory.mkdir(parents=True)
    for suffix in (".sby", ".v", "_formal.v"):
        source = ROOT / JOB.parent / (JOB.stem + suffix)
        shutil.copyfile(source, directory / source.name)
    if cover_mode:
        job = directory / JOB.name
        job.write_text(job.read_text().replace("mode bmc", "mode cover"))
        harness = directory / (JOB.stem + "_formal.v")
        text = harness.read_text()
        assert text.count("assert (") == 1
        harness.write_text(text.replace("assert (", "cover ("))
    if named_default:
        job = directory / JOB.name
        job.write_text(
            "[tasks]\ndefault\n"
            + job.read_text()
            .replace("mode ", "default: mode ")
            .replace("depth ", "default: depth ")
            .replace("smtbmc z3", "default: smtbmc z3")
        )
    inputs = resolve_job(repository, JOB)
    yosys, sby, z3 = (_tool(name) for name in ("yosys", "sby", "z3"))
    provenance = (yosys.parent.parent / "PROVENANCE.txt").read_text()
    hosted = os.environ.get("GITHUB_ACTIONS") == "true"
    control = ["sudo", "systemctl"] if hosted else ["systemctl", "--user"]
    launcher = ["sudo", "systemd-run"] if hosted else ["systemd-run", "--user"]
    policy = EvidencePolicy(
        socket.gethostname(), 128 * 1024**2, 60, "local qualification", 128 * 1024**2, 60
    )
    unit = "scn-formal-receipt-test-" + uuid.uuid4().hex + ".service"
    log = repository / "native.log"
    command = [
        *launcher,
        "--unit=" + unit,
        *(["--uid=" + str(os.getuid()), "--gid=" + str(os.getgid())] if hosted else []),
        "--property=WorkingDirectory=" + str(inputs.directory),
        "--property=MemoryMax=" + str(policy.memory_bytes),
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=60s",
        "--property=TimeoutStopSec=5s",
        "--property=KillMode=control-group",
        "--property=MemoryAccounting=yes",
        "--property=RemainAfterExit=yes",
        "--property=CPUQuota=100%",
        "--property=Nice=19",
        "--property=StandardOutput=append:" + str(log),
        "--property=StandardError=append:" + str(log),
        "/usr/bin/env",
        *[
            key + "=" + os.environ[key]
            for key in (
                "PATH",
                "PYTHONPATH",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONPYCACHEPREFIX",
            )
            if key in os.environ
        ],
        str(sby),
        "-f",
        "--prefix",
        str(repository / "proof"),
        JOB.name,
    ]
    peak = 0
    events: dict[str, int] = {}
    cgroup: Path | None = None
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        deadline = time.monotonic() + 65
        while True:
            state = _state(control, unit)
            if state["ControlGroup"]:
                cgroup = Path("/sys/fs/cgroup") / state["ControlGroup"].lstrip("/")
            if cgroup is not None and (cgroup / "memory.peak").is_file():
                peak = max(peak, int((cgroup / "memory.peak").read_text()))
                events = {
                    key: int(value)
                    for key, value in (
                        line.split() for line in (cgroup / "memory.events").read_text().splitlines()
                    )
                }
            if state.get("MemoryPeak", "").isdigit():
                peak = max(peak, int(state["MemoryPeak"]))
            if state["SubState"] in ("exited", "failed", "dead"):
                break
            assert time.monotonic() < deadline, "Actual tiny SBY job exceeded its finite ceiling."
            time.sleep(0.05)
        processes = (
            (cgroup / "cgroup.procs").read_text().split()
            if cgroup is not None and (cgroup / "cgroup.procs").is_file()
            else []
        )
        assert state["MainPID"] == "0" and not processes, state
        assert state["Result"] == "success" and state["ExecMainStatus"] == "0", log.read_text()
        primary = repository / ("proof_default" if named_default else "proof")
        assert (primary / "status").read_text().split()[0] == "PASS", log.read_text()
        wall = (
            int(state["ExecMainExitTimestampMonotonic"])
            - int(state["ExecMainStartTimestampMonotonic"])
        ) / 1_000_000
        receipt = {
            "schema_version": 1,
            "producer": "actual native receipt test producer",
            "job": JOB.as_posix(),
            "source_hashes": inputs.source_hashes(),
            "task_statuses": {"default": "PASS"},
            "command": {
                "program": "sby",
                "options": ["-f"],
                "output_option": "--prefix",
                "directory": JOB.parent.as_posix(),
                "argument": JOB.name,
            },
            "toolchain": {
                **source_pins(provenance),
                "binary_sha256": {"yosys": digest(yosys), "sby": digest(sby)},
                "yosys_version": subprocess.check_output([str(yosys), "-V"], text=True).strip(),
                "sby_version": subprocess.check_output([str(sby), "--version"], text=True).strip(),
                "solver": {
                    "name": "z3",
                    "sha256": digest(z3),
                    "version": subprocess.check_output([str(z3), "--version"], text=True).strip(),
                },
            },
            "measurement": {
                "hostname": policy.hostname,
                "unit": unit,
                "wall_clock_source": "systemd-exec-monotonic",
                "start_monotonic_microseconds": int(state["ExecMainStartTimestampMonotonic"]),
                "stop_monotonic_microseconds": int(state["ExecMainExitTimestampMonotonic"]),
                "status": "PASS",
                "native_exit_code": int(state["ExecMainStatus"]),
                "native_exit_source": "systemd-exec",
                "unit_exit_code": int(state["ExecMainStatus"]),
                "unit_accounting": "retained",
                "unit_result": state["Result"],
                "main_pid": int(state["MainPID"]),
                "cgroup_processes": processes,
                "process_tree_closed": True,
                "wall_seconds": wall,
                "peak_memory_bytes": peak,
                "memory_limit_bytes": int(state["MemoryMax"]),
                "swap_limit_bytes": int(state["MemorySwapMax"]),
                "time_limit_seconds": 60,
                "oom_events": {
                    name: events[name] for name in ("oom", "oom_kill", "oom_group_kill")
                },
                "invocation_id": state["InvocationID"],
            },
        }
    finally:
        subprocess.run([*control, "stop", unit], capture_output=True, check=True, timeout=15)
        closing = _state(control, unit)
        assert closing["ActiveState"] == "inactive" and closing["MainPID"] == "0", closing
        assert (
            cgroup is None
            or not (cgroup / "cgroup.procs").exists()
            or not (cgroup / "cgroup.procs").read_text().split()
        )
    public_log = repository / "public-native.log"
    public_log.write_bytes(
        normalize_native_log(
            log.read_bytes(),
            {
                "workspace": str(repository),
                "unit": unit,
                "hostname": policy.hostname,
                "username": pwd.getpwuid(os.getuid()).pw_name,
            },
        )
    )
    receipt["log"] = {
        "path": public_log.name,
        "raw_sha256": digest(log),
        "normalized_sha256": digest(public_log),
        "normalizer_id": NORMALIZER_ID,
        "normalizer_sha256": digest(ROOT / "tools/formal_log_normalization.py"),
    }
    measurement = object_fields(receipt["measurement"])
    measurement["active_state"] = closing["ActiveState"]
    receipt["measurement"] = measurement
    (repository / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result = NativeEvidence(repository, policy, provenance, {"z3": z3})
    collect_native_audit(result)
    return result


def copy_case(original: NativeEvidence, repository: Path) -> NativeEvidence:
    """Copy only actual proof inputs and small evidence for an isolated control.

    Parameters
    ----------
    original : NativeEvidence
        Actual measured fixture.
    repository : Path
        Empty owned case directory.

    Returns
    -------
    NativeEvidence
        Isolated inputs and evidence without duplicated solver build output.
    """
    directory = repository / JOB.parent
    directory.mkdir(parents=True)
    for source in (original.repository / JOB.parent).iterdir():
        shutil.copyfile(source, directory / source.name)
    for name in (
        "receipt.json",
        "native.log",
        "public-native.log",
        "nonvacuity.json",
        "public-audit.log",
    ):
        shutil.copyfile(original.repository / name, repository / name)
    assert read_object(repository / "receipt.json")["task_statuses"] == {"default": "PASS"}
    return NativeEvidence(repository, original.policy, original.provenance, original.solvers)
