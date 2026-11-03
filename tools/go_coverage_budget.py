# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live hosted-runner native coverage budget derivation

"""Read actual Linux capacity and derive caller-policy budgets without admitting resources.

No cgroup, affinity, cooling or service state is written. Fractions are explicit
reviewed policy inputs; observed host capacity and remaining time stay separate.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import TypedDict


class CPUQuota(TypedDict):
    """Retain an actual or explicitly copied kernel quota input and its parsed ratio."""

    path: str
    raw: str | None
    quota_cpus: float | None


class BudgetProfile(TypedDict):
    """Keep observed capacity, caller policy and derived CLI values separately typed."""

    recorded_epoch: float
    job_started_epoch: float
    job_timeout_seconds: float
    remaining_job_seconds: float
    runtime_fraction_policy: float
    stop_fraction_policy: float
    affinity: list[int]
    cgroup_membership: str
    visible_cpu_quotas: list[CPUQuota]
    effective_cpu_capacity: float
    host_mem_available_bytes: int
    allocation_disk_free_bytes: int
    jobs: int
    seconds: int
    stop_seconds: float
    resource_admission_claimed: bool
    memory_or_disk_reservation_claimed: bool


def cgroup_cpu_rows(group: Path, root: Path) -> list[CPUQuota]:
    """Read every visible ancestor quota, retaining absent controller files explicitly.

    Parameters
    ----------
    group : pathlib.Path
        Actual current cgroup directory, or an explicit copied parser input.
    root : pathlib.Path
        Visible cgroup hierarchy root; traversal cannot escape this directory.

    Returns
    -------
    list of dict
        Raw file values and parsed finite CPU limits, without invented root quotas.

    Raises
    ------
    ValueError
        The hierarchy or a present kernel-format quota value is invalid.
    """
    group, root = group.resolve(strict=True), root.resolve(strict=True)
    if not group.is_relative_to(root):
        raise ValueError("Current cgroup is outside its visible hierarchy")
    rows: list[CPUQuota] = []
    current = group
    while True:
        path = current / "cpu.max"
        row: CPUQuota = {"path": str(path), "raw": None, "quota_cpus": None}
        if path.exists():
            raw = path.read_text().strip()
            fields = raw.split()
            if len(fields) != 2 or not fields[1].isdigit() or int(fields[1]) <= 0:
                raise ValueError("Malformed actual cpu.max period")
            if fields[0] != "max" and (not fields[0].isdigit() or int(fields[0]) <= 0):
                raise ValueError("Malformed actual cpu.max quota")
            row["raw"] = raw
            if fields[0] != "max":
                row["quota_cpus"] = int(fields[0]) / int(fields[1])
        rows.append(row)
        if current == root:
            return rows
        current = current.parent


def resolve_cgroup_membership(root: Path, membership: str) -> Path:
    """Validate an actual hierarchy and unified-membership snapshot without inventing one.

    Parameters
    ----------
    root : pathlib.Path
        Actual visible cgroup root, or an owned negative parser-input view.
    membership : str
        Actual ``/proc/self/cgroup`` contents, or a damaged copy for refusal checks.

    Returns
    -------
    pathlib.Path
        Existing unified group; the quota reader enforces its hierarchy boundary.

    Raises
    ------
    RuntimeError, OSError
        The controller declaration or one absolute unified membership is absent.
    """
    if not (root / "cgroup.controllers").is_file():
        raise RuntimeError("The live runner does not expose a qualified cgroup-v2 hierarchy")
    locations = [
        line.removeprefix("0::") for line in membership.splitlines() if line.startswith("0::")
    ]
    if len(locations) != 1 or not locations[0].startswith("/"):
        raise RuntimeError("Actual unified cgroup membership is missing or ambiguous")
    return (root / locations[0].lstrip("/")).resolve(strict=True)


def effective_cpu_capacity(affinity: list[int], quotas: list[CPUQuota]) -> float:
    """Combine actual parsed CPU observations, refusing an empty affinity snapshot.

    Parameters
    ----------
    affinity : list of int
        Actual available CPUs, or an explicitly damaged observation copy.
    quotas : list of dict
        Validated visible ancestor quota rows from the kernel-format reader.

    Returns
    -------
    float
        Smallest actual affinity/quota capacity; this grants no resources.

    Raises
    ------
    RuntimeError
        The supplied affinity snapshot contains no actually available CPU.
    """
    if not affinity:
        raise RuntimeError("The native producer has no actually available CPU")
    return min(
        [
            float(len(affinity)),
            *(row["quota_cpus"] for row in quotas if row["quota_cpus"] is not None),
        ]
    )


def parse_memory_headroom(memory: str, disk_free: int) -> int:
    """Validate actual memory units and positive observed memory/disk headroom.

    Parameters
    ----------
    memory : str
        Real ``/proc/meminfo`` text, or an owned damaged parser-input copy.
    disk_free : int
        Actual allocation free bytes, or a zero-headroom negative input.

    Returns
    -------
    int
        Host available memory bytes; no reservation or admission is made.

    Raises
    ------
    RuntimeError, ValueError
        Available memory is absent, ambiguous, malformed or nonpositive, or the
        allocation has no observed free disk bytes.
    """
    available = [
        line.split()[1:] for line in memory.splitlines() if line.startswith("MemAvailable:")
    ]
    if len(available) != 1 or len(available[0]) != 2 or available[0][1] != "kB":
        raise RuntimeError("Actual available memory is missing or has unknown units")
    memory_bytes = int(available[0][0]) * 1024
    if memory_bytes <= 0 or disk_free <= 0:
        raise RuntimeError("The native producer has no observed memory or disk headroom")
    return memory_bytes


def derive_profile(
    job_timeout_seconds: float,
    job_started_epoch: float,
    allocation: Path,
    runtime_fraction: float,
    stop_fraction: float,
) -> BudgetProfile:
    """Derive finite native CLI values from the live runner and explicit caller policy.

    Parameters
    ----------
    job_timeout_seconds : float
        Existing enclosing job's actual configured timeout, passed by its owner.
    job_started_epoch : float
        Recorded wall-clock start of that enclosing job, before toolchain setup.
    allocation : pathlib.Path
        Existing admitted output parent used for the actual disk-free observation.
    runtime_fraction, stop_fraction : float
        Reviewed policy fractions; neither value is a hardware-capacity constant.

    Returns
    -------
    dict
        Actual observations, policy inputs, derived jobs/seconds/stop_seconds and
        an explicit false resource-admission claim.

    Raises
    ------
    ValueError, RuntimeError, OSError
        Inputs or live capacity are invalid, time is exhausted, or Linux cgroup-v2
        membership cannot be resolved without guessing.
    """
    now = time.time()
    if (
        not all(
            math.isfinite(value)
            for value in (
                job_timeout_seconds,
                job_started_epoch,
                runtime_fraction,
                stop_fraction,
            )
        )
        or job_timeout_seconds <= 0
        or job_started_epoch <= 0
        or job_started_epoch > now
        or not 0 < runtime_fraction < 1
        or not 0 < stop_fraction < 1
    ):
        raise ValueError(
            "Finite job times and explicit fractions between zero and one are required"
        )
    remaining = job_started_epoch + job_timeout_seconds - now
    seconds = math.floor(remaining * runtime_fraction)
    if seconds < 1:
        raise ValueError("The enclosing job has no remaining native-gate budget")
    if sys.platform != "linux":
        raise RuntimeError("The hosted native gate requires actual Linux cgroup-v2 observations")
    root = Path("/sys/fs/cgroup").resolve(strict=True)
    membership = Path("/proc/self/cgroup").read_text()
    group = resolve_cgroup_membership(root, membership)
    quotas = cgroup_cpu_rows(group, root)
    affinity = sorted(os.sched_getaffinity(0))
    effective = effective_cpu_capacity(affinity, quotas)
    disk = shutil.disk_usage(allocation.resolve(strict=True))
    memory_bytes = parse_memory_headroom(Path("/proc/meminfo").read_text(), disk.free)
    return {
        "recorded_epoch": now,
        "job_started_epoch": job_started_epoch,
        "job_timeout_seconds": job_timeout_seconds,
        "remaining_job_seconds": remaining,
        "runtime_fraction_policy": runtime_fraction,
        "stop_fraction_policy": stop_fraction,
        "affinity": affinity,
        "cgroup_membership": membership,
        "visible_cpu_quotas": quotas,
        "effective_cpu_capacity": effective,
        "host_mem_available_bytes": memory_bytes,
        "allocation_disk_free_bytes": disk.free,
        "jobs": max(1, math.floor(effective)),
        "seconds": seconds,
        "stop_seconds": seconds * stop_fraction,
        "resource_admission_claimed": False,
        "memory_or_disk_reservation_claimed": False,
    }
