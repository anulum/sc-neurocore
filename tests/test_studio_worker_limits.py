# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Embedded Studio worker limit acceptance

"""Real process jobs verify limits before task import and refusal on allocation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_worker_limits import StudioWorkerLimits


def _manager(root: Path, *, max_data_bytes: int) -> StudioJobManager:
    return StudioJobManager(
        root=root,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=10.0,
        worker_limits=StudioWorkerLimits(
            max_data_bytes=max_data_bytes,
            max_cpu_seconds=8,
            max_open_files=256,
            max_file_bytes=4 * 1024**2,
        ),
    )


def test_limits_are_active_before_task_module_import(tmp_path: Path) -> None:
    """The public manager launches a task whose import snapshots kernel limits."""
    manager = _manager(tmp_path / "jobs", max_data_bytes=256 * 1024**2)
    record = manager.submit_process_task(
        kind="analysis",
        owner="operator",
        request_id="limits",
        task_path="tests.studio_worker_limit_tasks:report_import_limits",
        payload={},
    )
    result = manager.wait(record.job_id, timeout_seconds=15)

    assert result.status == "completed", result.error
    assert result.result is not None
    assert result.result["RLIMIT_DATA"] == 256 * 1024**2
    assert result.result["RLIMIT_CPU"] == 8
    assert result.result["RLIMIT_NOFILE"] == 256
    assert result.result["RLIMIT_FSIZE"] == 4 * 1024**2
    assert result.result["RLIMIT_CORE"] == 0


def test_allocation_over_limit_fails_without_success_result(tmp_path: Path) -> None:
    """Kernel refusal reaches the durable failure record through a real worker."""
    manager = _manager(tmp_path / "jobs", max_data_bytes=64 * 1024**2)
    record = manager.submit_process_task(
        kind="analysis",
        owner="operator",
        request_id="allocation",
        task_path="tests.studio_worker_limit_tasks:allocate_beyond_data_limit",
        payload={"bytes": 128 * 1024**2},
    )
    result = manager.wait(record.job_id, timeout_seconds=15)

    assert result.status == "failed"
    # The kernel's refusal raises without a message, so the worker leaves no
    # fault text and the record keeps the supervisor's exit report.
    assert result.error == "Studio process worker exited with 1."
    assert result.result is None


def test_synthesis_child_respects_inherited_worker_ceiling(tmp_path: Path) -> None:
    """A real Yosys child runs when its requested limit exceeds the worker hard limit."""
    manager = _manager(tmp_path / "jobs", max_data_bytes=2 * 1024**3)
    record = manager.submit_process_task(
        kind="analysis",
        owner="operator",
        request_id="nested-eda",
        task_path="sc_neurocore.studio.platform.synthesis_process:run_synthesis_process_task",
        payload={
            "eda_process_cpu_seconds": 30.0,
            "eda_process_memory_bytes": 1024**3,
            "target": "ice40",
            "verilog": "module t(input wire a, output wire y); assign y = a; endmodule",
        },
    )
    result = manager.wait(record.job_id, timeout_seconds=15)

    assert result.status == "completed", result.error
    assert result.result is not None
    assert result.result["success"] is True


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limit_is_refused_before_manager_start(value: object) -> None:
    """A malformed limit cannot silently become an unlimited worker setting."""
    with pytest.raises(ValueError, match="positive integer"):
        StudioWorkerLimits(
            max_data_bytes=cast(int, value),
            max_open_files=256,
            max_file_bytes=1024,
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_cpu_limit_is_refused(value: object) -> None:
    """CPU ceilings cannot silently become disabled or fractional."""
    with pytest.raises(ValueError, match="positive integer"):
        StudioWorkerLimits(
            max_data_bytes=1024,
            max_open_files=256,
            max_file_bytes=1024,
            max_cpu_seconds=cast(int, value),
        )


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_invalid_timeout_is_refused(value: float) -> None:
    """A malformed job timeout cannot become an unbounded CPU budget."""
    limits = StudioWorkerLimits(max_data_bytes=1024, max_open_files=256, max_file_bytes=1024)
    with pytest.raises(ValueError, match="finite and positive"):
        limits.worker_arguments(value)


def test_default_limits_reach_a_real_worker(tmp_path: Path) -> None:
    """The normal manager path uses the host default when no override is supplied."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=10.0,
    )
    record = manager.submit_process_task(
        kind="analysis",
        owner="operator",
        request_id="defaults",
        task_path="tests.studio_worker_limit_tasks:report_import_limits",
        payload={},
    )
    result = manager.wait(record.job_id, timeout_seconds=15)

    assert result.status == "completed", result.error
    assert result.result is not None
    assert result.result["RLIMIT_DATA"] == StudioWorkerLimits.for_host().max_data_bytes
    assert result.result["RLIMIT_NOFILE"] == 4096
