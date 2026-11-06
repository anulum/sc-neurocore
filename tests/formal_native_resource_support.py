# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual bounded resource-stop accounting fixture

"""Measure real local resource stops; never manufacture a native proof outcome."""

from __future__ import annotations

from dataclasses import replace
import json
import os
import pwd
import shutil
import socket
from pathlib import Path
import subprocess
import time
import uuid

from tests.formal_native_evidence_support import JOB, ROOT, NativeEvidence, _state, _tool
from tests.formal_native_nonvacuity_support import (
    bind_audit,
    collect_native_audit,
    read_cgroup_file,
)
from tools.formal_evidence_validation import (
    EvidencePolicy,
    digest,
    object_fields,
    read_object,
    source_pins,
)
from tools.formal_job_inputs import resolve_job
from tools.formal_log_normalization import NORMALIZER_ID, normalize_native_log
from tools.formal_receipt_identity import solver_configuration
from tools.formal_resource_policy import PRIMARY, YICES_RETRY


def measure_native_unit(
    directory: Path,
    argv: list[str],
    log: Path,
    policy: EvidencePolicy,
    *,
    oom_policy: str = "stop",
) -> dict[str, object]:
    """Collect actual cgroup and systemd termination facts for an owned unit.

    Parameters
    ----------
    directory : Path
        Actual native working directory in the isolated qualification fixture.
    argv : list of str
        Real executable and arguments; no synthetic status producer is inserted.
    log : Path
        Retained actual stdout and stderr file.
    policy : EvidencePolicy
        Actual local host, finite cgroup cap and runtime ceiling.
    oom_policy : str, optional
        Native OOM action; the accounting-only child-OOM control uses continue.

    Returns
    -------
    dict of str to object
        Actual unit identity, exit kind, monotonic clock, memory and closure.
        A resource stop or ERROR is never described as a successful proof.
    """
    hosted = os.environ.get("GITHUB_ACTIONS") == "true"
    control = ["sudo", "systemctl"] if hosted else ["systemctl", "--user"]
    launcher = ["sudo", "systemd-run"] if hosted else ["systemd-run", "--user"]
    unit = "scn-formal-resource-test-" + uuid.uuid4().hex + ".service"
    slice_unit = "scn_formal_resource_" + uuid.uuid4().hex + ".slice"
    command = [
        *launcher,
        "--unit=" + unit,
        "--slice=" + slice_unit,
        *(["--uid=" + str(os.getuid()), "--gid=" + str(os.getgid())] if hosted else []),
        "--property=WorkingDirectory=" + str(directory),
        "--property=MemoryMax=" + str(policy.memory_bytes),
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=" + str(policy.timeout_seconds) + "s",
        "--property=TimeoutStopSec=5s",
        "--property=KillMode=control-group",
        "--property=MemoryAccounting=yes",
        "--property=RemainAfterExit=yes",
        "--property=CPUQuota=100%",
        "--property=Nice=19",
        "--property=TasksMax=128",
        "--property=OOMPolicy=" + oom_policy,
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
                "TMPDIR",
            )
            if key in os.environ
        ],
        *argv,
    ]
    peak = 0
    events: dict[str, int] = {}
    cgroup: Path | None = None
    try:
        subprocess.run([*control, "start", slice_unit], capture_output=True, check=True, timeout=15)
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        slice_state = _state(control, slice_unit)
        event_group = Path("/sys/fs/cgroup") / slice_state["ControlGroup"].lstrip("/")
        deadline = time.monotonic() + policy.timeout_seconds + 7
        while True:
            state = _state(control, unit)
            if state["ControlGroup"]:
                cgroup = Path("/sys/fs/cgroup") / state["ControlGroup"].lstrip("/")
            if cgroup is not None:
                if (reading := read_cgroup_file(cgroup, "memory.peak")) is not None:
                    peak = max(peak, int(reading))
                if (reading := read_cgroup_file(event_group, "memory.events")) is not None:
                    events = dict(
                        (name, int(value))
                        for name, value in (line.split() for line in reading.splitlines())
                    )
            if state.get("MemoryPeak", "").isdigit():
                peak = max(peak, int(state["MemoryPeak"]))
            if state["SubState"] in ("exited", "failed", "dead"):
                break
            assert time.monotonic() < deadline, (
                "Owned native resource fixture exceeded its ceiling."
            )
            time.sleep(0.01)
        processes = (
            (read_cgroup_file(cgroup, "cgroup.procs") or "").split() if cgroup is not None else []
        )
        assert state["MainPID"] == "0" and not processes, state
        events = dict(
            (name, int(value))
            for name, value in (
                line.split() for line in (event_group / "memory.events").read_text().splitlines()
            )
        )
        start = int(state["ExecMainStartTimestampMonotonic"])
        stop = int(state["ExecMainExitTimestampMonotonic"])
        status = {"timeout": "TIMEOUT", "oom-kill": "MEMORY", "success": "PASS"}.get(
            state["Result"], "ERROR"
        )
        measurement: dict[str, object] = {
            "host_profile": policy.host_profile,
            "unit": unit,
            "status": status,
            "unit_accounting": "retained",
            "unit_result": state["Result"],
            "native_exit_source": "systemd-exec",
            "wall_clock_source": "systemd-exec-monotonic",
            "unit_exit_code": int(state["ExecMainStatus"]),
            "native_exit_code": int(state["ExecMainStatus"]),
            "unit_exec_code": int(state["ExecMainCode"]),
            "start_monotonic_microseconds": start,
            "stop_monotonic_microseconds": stop,
            "wall_seconds": (stop - start) / 1_000_000,
            "peak_memory_bytes": peak,
            "memory_limit_bytes": int(state["MemoryMax"]),
            "swap_limit_bytes": int(state["MemorySwapMax"]),
            "time_limit_seconds": policy.timeout_seconds,
            "main_pid": int(state["MainPID"]),
            "active_state": state["ActiveState"],
            "cgroup_processes": processes,
            "process_tree_closed": True,
            "oom_events": {name: events[name] for name in ("oom", "oom_kill", "oom_group_kill")},
            "invocation_id": state["InvocationID"],
            "oom_event_cgroup": str(event_group),
            "oom_event_scope": "exclusive single-unit slice",
        }
    finally:
        subprocess.run([*control, "stop", unit], capture_output=True, check=True, timeout=15)
        subprocess.run([*control, "stop", slice_unit], capture_output=True, check=True, timeout=15)
        closing = _state(control, unit)
        assert closing["MainPID"] == "0" and closing["ActiveState"] in ("inactive", "failed"), (
            closing
        )
    return measurement


def collect_native_resources(repository: Path) -> NativeEvidence:
    """Run both actual solvers to a native time ceiling and audit their model.

    Parameters
    ----------
    repository : Path
        Empty owned test repository. The tracked HDL is copied unchanged;
        the explicit test configuration uses depth 1000000 to exhaust time.

    Returns
    -------
    NativeEvidence
        Two real TIMEOUT attempts and a separately measured enable audit,
        under a local 128 MiB policy. Never production ML350 evidence.
    """
    policy = EvidencePolicy(
        "local resource qualification",
        128 * 1024**2,
        4,
        "local resource qualification",
        128 * 1024**2,
        20,
    )
    yosys, sby, z3, yices = (_tool(name) for name in ("yosys", "sby", "z3", "yices-smt2"))
    provenance = (yosys.parent.parent / "PROVENANCE.txt").read_text()
    solvers = {"z3": z3, "yices-smt2": yices}
    directory = repository / JOB.parent
    directory.mkdir(parents=True)
    for suffix in (".sby", ".v", "_formal.v"):
        source = ROOT / JOB.parent / (JOB.stem + suffix)
        raw = source.read_bytes()
        if suffix == ".sby":
            assert raw.count(b"depth 4") == 1
            raw = raw.replace(b"depth 4", b"depth 1000000")
        (directory / source.name).write_bytes(raw)
    definitions: list[dict[str, object]] = []
    for solver, contract, label in (
        ("z3", PRIMARY, "primary"),
        ("yices-smt2", YICES_RETRY, "yices"),
    ):
        executed_root = repository if solver == "z3" else repository / "yices-source"
        configuration: dict[str, object] = {}
        if solver != "z3":
            destination = executed_root / JOB.parent
            destination.mkdir(parents=True)
            for source in directory.iterdir():
                shutil.copyfile(source, destination / source.name)
            variant = solver_configuration((directory / JOB.name).read_text(), solver)
            (destination / JOB.name).write_text(variant)
            config = repository / "executed-yices.sby"
            config.write_text(variant)
            configuration["configuration"] = {"path": config.name, "sha256": digest(config)}
        inputs = resolve_job(executed_root, JOB)
        work = repository / ("proof" if solver == "z3" else "proof-yices")
        raw_log = repository / (label + "-native.log")
        measurement = measure_native_unit(
            inputs.directory, [str(sby), "-f", "--prefix", str(work), JOB.name], raw_log, policy
        )
        assert measurement["status"] == "TIMEOUT", raw_log.read_text()[-3000:]
        status = (work / "status").read_text().split()[0] if (work / "status").is_file() else None
        public_log = repository / (label + "-public.log")
        public_log.write_bytes(
            normalize_native_log(
                raw_log.read_bytes(),
                {
                    "workspace": str(repository),
                    "unit": str(measurement["unit"]),
                    "hostname": socket.gethostname(),
                    "username": pwd.getpwuid(os.getuid()).pw_name,
                },
            )
        )
        receipt: dict[str, object] = {
            "schema_version": 1,
            "producer": "actual local resource fixture; not production evidence",
            "job": JOB.as_posix(),
            "source_hashes": inputs.source_hashes(),
            "task_statuses": {"default": status},
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
                    "name": solver,
                    "sha256": digest(solvers[solver]),
                    "version": subprocess.check_output(
                        [str(solvers[solver]), "--version"], text=True
                    ).strip(),
                },
            },
            "measurement": measurement,
            "log": {
                "path": public_log.name,
                "raw_sha256": digest(raw_log),
                "normalized_sha256": digest(public_log),
                "normalizer_id": NORMALIZER_ID,
                "normalizer_sha256": digest(ROOT / "tools/formal_log_normalization.py"),
            },
        }
        path = repository / ("receipt.json" if solver == "z3" else "yices-receipt.json")
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        definitions.append(
            {
                "resource_contract": contract,
                "receipt": path.name,
                "review": label + "-review.json",
                **configuration,
            }
        )
    (repository / "attempt-definitions.json").write_text(json.dumps(definitions))
    evidence = NativeEvidence(repository, policy, provenance, solvers)
    collect_native_audit(
        replace(
            evidence,
            policy=replace(
                policy,
                memory_bytes=policy.audit_memory_bytes,
                timeout_seconds=policy.audit_timeout_seconds,
            ),
        ),
        resource_parent=True,
    )
    bind_resource_row(evidence)
    return evidence


def bind_resource_row(evidence: NativeEvidence) -> dict[str, object]:
    """Refresh test-only review bindings around genuine retained native facts.

    Parameters
    ----------
    evidence : NativeEvidence
        Actual native TIMEOUT pair and separate non-vacuity audit.

    Returns
    -------
    dict of str to object
        Test-only row entry. Rebinding never changes any measured outcome,
        model, native log, clock, accounting or audit result.
    """
    root = evidence.repository
    definitions = json.loads((root / "attempt-definitions.json").read_text())
    assert isinstance(definitions, list)
    entries: list[dict[str, object]] = []
    summaries = []
    for definition in definitions:
        attempt = object_fields(definition)
        receipt_path = root / str(attempt["receipt"])
        receipt = read_object(receipt_path)
        log = object_fields(receipt["log"])
        configuration = object_fields(attempt.get("configuration", {}))
        review = {
            "schema_version": 1,
            "verdict": "CLEAR",
            "reviewer": "resource contract fixture; not independent production review",
            "resource_contract": attempt["resource_contract"],
            "receipt_sha256": digest(receipt_path),
            "raw_log_sha256": log["raw_sha256"],
            "normalized_log_sha256": log["normalized_sha256"],
            "normalizer_id": log["normalizer_id"],
            "normalizer_sha256": log["normalizer_sha256"],
            "source_hashes": receipt["source_hashes"],
            "execution_exception": None,
            "exception_registry_sha256": None,
            "configuration_sha256": configuration.get("sha256"),
        }
        review_path = root / str(attempt["review"])
        review_path.write_text(json.dumps(review, indent=2) + "\n")
        attempt.update(receipt_sha256=digest(receipt_path), review_sha256=digest(review_path))
        entries.append(attempt)
        summaries.append(
            {key: attempt[key] for key in ("resource_contract", "receipt_sha256", "review_sha256")}
            | {"native_status": object_fields(receipt["measurement"])["status"]}
        )
    audit = bind_audit(root)
    scopes: dict[str, object] = {}
    for name, data in object_fields(read_object(root / "nonvacuity.json")["tasks"]).items():
        options = object_fields(object_fields(object_fields(data)["configuration"])["options"])
        scopes[name] = {"mode": options["mode"], "depth": int(str(options["depth"]))}
    notes = root / "release-notes.md"
    tokens = [
        JOB.as_posix(),
        "UNPROVEN-RESOURCE",
        str(audit["nonvacuity_sha256"]),
        *[str(row["receipt_sha256"]) for row in summaries],
        *[
            f"{name}: mode={object_fields(scope)['mode']}; depth={object_fields(scope)['depth']}"
            for name, scope in scopes.items()
        ],
    ]
    notes.write_text(" | ".join(tokens) + "\n")
    entry = {
        "outcome": "UNPROVEN-RESOURCE",
        "resource_contract": YICES_RETRY,
        "attempts": entries,
        **audit,
        "nonvacuity_parent_sha256": entries[0]["receipt_sha256"],
        "release_notes": {"path": notes.name, "sha256": digest(notes)},
    }
    review = {
        "schema_version": 1,
        "verdict": "CLEAR",
        "outcome": "UNPROVEN-RESOURCE",
        "job": JOB.as_posix(),
        "resource_contract": YICES_RETRY,
        "attempts": summaries,
        "reviewer": "row contract fixture; not independent production review",
        "depth_scope": scopes,
        **{
            key: entry[key]
            for key in (
                "nonvacuity_sha256",
                "nonvacuity_review_sha256",
                "nonvacuity_parent_sha256",
                "release_notes",
            )
        },
    }
    path = root / "row-review.json"
    path.write_text(json.dumps(review, indent=2) + "\n")
    entry.update(review=path.name, review_sha256=digest(path))
    return entry
