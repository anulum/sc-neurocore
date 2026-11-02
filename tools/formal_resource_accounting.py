# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained native resource-exhaustion accounting

"""Distinguish a real bounded resource stop from proof or tool failure."""

from __future__ import annotations

import re

from tools.formal_evidence_validation import EvidencePolicy, finite_number, object_fields, require

STOP_GRACE_SECONDS = 15


def validate_resource_measurement(
    receipt: dict[str, object], policy: EvidencePolicy
) -> tuple[str, float, int]:
    """Require actual kernel/unit accounting for one exhausted proof attempt.

    Parameters
    ----------
    receipt : dict of str to object
        Actual receipt with retained systemd exit and cgroup accounting.
    policy : EvidencePolicy
        Explicitly indexed and reviewed host, cap and primary runtime limit.

    Returns
    -------
    tuple of str, float and int
        Actual TIMEOUT or MEMORY outcome, monotonic wall and cgroup peak.
        These are incomplete results, never successful proofs.

    Raises
    ------
    ValueError
        If accounting, stop cause, closure, clock, cap or OOM evidence differs.
    """
    measurement = object_fields(receipt.get("measurement"))
    status = measurement.get("status")
    require(status in ("TIMEOUT", "MEMORY"), "Resource receipt is not TIMEOUT or MEMORY.")
    for name, expected in {
        "hostname": policy.hostname,
        "unit_accounting": "retained",
        "native_exit_source": "systemd-exec",
        "wall_clock_source": "systemd-exec-monotonic",
        "active_state": "failed",
        "cgroup_processes": [],
    }.items():
        require(measurement.get(name) == expected, f"Resource {name} does not qualify.")
    require(measurement.get("process_tree_closed") is True, "Resource process tree is not closed.")
    for name, expected_limit in {
        "main_pid": 0,
        "memory_limit_bytes": policy.memory_bytes,
        "swap_limit_bytes": 0,
        "time_limit_seconds": policy.timeout_seconds,
    }.items():
        value = measurement.get(name)
        require(
            type(value) is int and value == expected_limit, f"Resource {name} does not qualify."
        )
    invocation = measurement.get("invocation_id")
    require(
        isinstance(invocation, str) and re.fullmatch(r"[0-9a-f]{32}", invocation) is not None,
        "Resource invocation identity is missing or invalid.",
    )
    code, unit_code = measurement.get("native_exit_code"), measurement.get("unit_exit_code")
    require(
        type(code) is int and type(unit_code) is int and code == unit_code and code > 0,
        "Resource native and retained unit exit codes differ or claim success.",
    )
    execution_kind = measurement.get("unit_exec_code")
    require(
        type(execution_kind) is int
        and (execution_kind == 1 or (execution_kind == 2 and code in (9, 15))),
        "Resource main process has an unexplained signal or exit kind.",
    )
    expected_result = "timeout" if status == "TIMEOUT" else "oom-kill"
    require(measurement.get("unit_result") == expected_result, "Resource unit stop cause differs.")
    start, stop = (
        measurement.get("start_monotonic_microseconds"),
        measurement.get("stop_monotonic_microseconds"),
    )
    require(
        type(start) is int and type(stop) is int and start > 0 and stop > start,
        "Resource monotonic clock stamps are invalid.",
    )
    wall = finite_number(measurement.get("wall_seconds"), "Resource wall_seconds")
    require(
        wall == (int(str(stop)) - int(str(start))) / 1_000_000,
        "Resource wall differs from its actual monotonic stamps.",
    )
    require(
        0 < wall <= policy.timeout_seconds + STOP_GRACE_SECONDS,
        "Resource wall exceeds its admitted primary and stop budgets.",
    )
    if status == "TIMEOUT":
        require(
            wall >= policy.timeout_seconds, "Resource timeout stopped before its admitted ceiling."
        )
    peak = finite_number(measurement.get("peak_memory_bytes"), "Resource peak_memory_bytes")
    require(
        type(measurement.get("peak_memory_bytes")) is int and 0 < peak <= policy.memory_bytes,
        "Resource cgroup peak is outside its admitted cap.",
    )
    events = object_fields(measurement.get("oom_events"))
    for name in ("oom", "oom_kill", "oom_group_kill"):
        value = events.get(name)
        require(type(value) is int and value >= 0, f"Resource {name} must be a nonnegative count.")
    if status == "TIMEOUT":
        require(
            all(events[name] == 0 for name in ("oom", "oom_kill", "oom_group_kill")),
            "Resource timeout also reports an OOM.",
        )
    else:
        require(
            int(str(events["oom"])) > 0
            and (int(str(events["oom_kill"])) > 0 or int(str(events["oom_group_kill"])) > 0),
            "Resource MEMORY has no actual kernel OOM kill.",
        )
    return str(status), wall, int(peak)
