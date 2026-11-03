# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live native coverage budget and quota-format contracts

"""Check live capacity separately from explicit kernel-format parser controls."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
import time

import pytest

from tools.go_coverage_budget import (
    cgroup_cpu_rows,
    derive_profile,
    effective_cpu_capacity,
    parse_memory_headroom,
    resolve_cgroup_membership,
)


def test_live_profile_derives_all_cli_values_from_actual_capacity(native_job: Path) -> None:
    """The current allocation supplies actual time, CPU, memory and disk observations."""
    inputs = json.loads((native_job / "producer-inputs.json").read_text())
    profile = derive_profile(inputs["seconds"], inputs["started_epoch"], native_job, 1 / 4, 1 / 200)
    assert profile["resource_admission_claimed"] is False
    assert profile["memory_or_disk_reservation_claimed"] is False
    assert profile["jobs"] == max(1, math.floor(float(profile["effective_cpu_capacity"])))
    assert profile["seconds"] == math.floor(float(profile["remaining_job_seconds"]) / 4)
    assert profile["stop_seconds"] == float(profile["seconds"]) / 200
    assert profile["visible_cpu_quotas"] and profile["affinity"]
    assert float(profile["host_mem_available_bytes"]) > 0
    assert float(profile["allocation_disk_free_bytes"]) > 0


@pytest.mark.parametrize(
    "fault", ["timeout", "nan", "future", "start", "expired", "runtime", "stop"]
)
def test_invalid_job_or_policy_cannot_create_a_budget(native_job: Path, fault: str) -> None:
    """Invalid times and fractions refuse before any host admission could be inferred."""
    inputs = json.loads((native_job / "producer-inputs.json").read_text())
    seconds, started, runtime, stop = (
        float(inputs["seconds"]),
        float(inputs["started_epoch"]),
        1 / 4,
        1 / 200,
    )
    if fault == "timeout":
        seconds = 0
    elif fault == "nan":
        seconds = float("nan")
    elif fault == "future":
        started = time.time() + seconds
    elif fault == "start":
        started = 0
    elif fault == "expired":
        started = time.time() - seconds - 1
    elif fault == "runtime":
        runtime = 1
    else:
        stop = 0
    with pytest.raises(ValueError):
        derive_profile(seconds, started, native_job, runtime, stop)


def test_quota_parser_retains_nested_and_absent_controller_inputs(tmp_path: Path) -> None:
    """Kernel-format examples qualify parsing, without claiming observed host capacity."""
    root = tmp_path / "visible-root"
    parent, leaf = root / "parent", root / "parent/leaf"
    leaf.mkdir(parents=True)
    (parent / "cpu.max").write_text("max 100000\n")
    (leaf / "cpu.max").write_text("50000 100000\n")
    rows = cgroup_cpu_rows(leaf, root)
    assert [row["quota_cpus"] for row in rows] == [0.5, None, None]
    assert [row["raw"] for row in rows] == ["50000 100000", "max 100000", None]
    with pytest.raises(ValueError, match="outside"):
        cgroup_cpu_rows(tmp_path, root)


@pytest.mark.parametrize(
    "raw", ["", "max", "max invalid", "max 0", "0 100000", "bad 100000", "-1 100000"]
)
def test_malformed_quota_configuration_cannot_become_capacity(tmp_path: Path, raw: str) -> None:
    """Invalid protocol fields cannot silently enlarge the producer's effective CPU allowance."""
    (tmp_path / "cpu.max").write_text(raw)
    with pytest.raises(ValueError, match="cpu.max"):
        cgroup_cpu_rows(tmp_path, tmp_path)


@pytest.mark.parametrize("fault", ["controller", "missing", "duplicate", "relative"])
def test_damaged_actual_membership_snapshot_cannot_resolve_a_group(
    tmp_path: Path, fault: str
) -> None:
    """Corrupt copied kernel metadata without changing the real hierarchy or claiming capacity."""
    root = Path("/sys/fs/cgroup")
    membership = Path("/proc/self/cgroup").read_text()
    assert (root / "cgroup.controllers").is_file()
    if fault == "controller":
        with pytest.raises(RuntimeError, match="qualified cgroup"):
            resolve_cgroup_membership(tmp_path, membership)
        return
    (tmp_path / "cgroup.controllers").write_bytes((root / "cgroup.controllers").read_bytes())
    if fault == "missing":
        membership = "\n".join(
            line for line in membership.splitlines() if not line.startswith("0::")
        )
    elif fault == "duplicate":
        membership = membership.rstrip("\n") + "\n" + membership
    else:
        actual = next(line for line in membership.splitlines() if line.startswith("0::"))
        membership = membership.replace(actual, actual.replace("0::/", "0::", 1))
    with pytest.raises(RuntimeError, match="membership"):
        resolve_cgroup_membership(tmp_path, membership)


def test_empty_copy_of_actual_affinity_cannot_become_cpu_capacity() -> None:
    """Clearing a genuine live affinity snapshot must refuse, without replacing the OS reader."""
    root = Path("/sys/fs/cgroup").resolve(strict=True)
    group = resolve_cgroup_membership(root, Path("/proc/self/cgroup").read_text())
    quotas = cgroup_cpu_rows(group, root)
    damaged = sorted(os.sched_getaffinity(0))
    assert damaged
    damaged.clear()
    with pytest.raises(RuntimeError, match="available CPU"):
        effective_cpu_capacity(damaged, quotas)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "units", "memory-zero", "disk-zero"])
def test_damaged_actual_memory_or_disk_observations_cannot_become_headroom(
    native_job: Path, fault: str
) -> None:
    """Only damaged copies are used; host meminfo, free space and live admission are unchanged."""
    memory = Path("/proc/meminfo").read_text()
    disk_free = shutil.disk_usage(native_job).free
    available = next(line for line in memory.splitlines() if line.startswith("MemAvailable:"))
    if fault == "missing":
        memory = memory.replace(available, "")
    elif fault == "duplicate":
        memory += available + "\n"
    elif fault == "units":
        memory = memory.replace(available, available.replace("kB", "unknown-unit"))
    elif fault == "memory-zero":
        memory = memory.replace(available, "MemAvailable: 0 kB")
    else:
        disk_free = 0
    with pytest.raises(RuntimeError, match="memory|headroom"):
        parse_memory_headroom(memory, disk_free)
