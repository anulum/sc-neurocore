# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benchmark measurement context helpers

from __future__ import annotations

import os
from pathlib import Path


def load_average() -> list[float] | None:
    """Return the current host load averages when the platform exposes them."""

    return list(os.getloadavg()) if hasattr(os, "getloadavg") else None


def _cpu_indices() -> list[int]:
    cpu_root = Path("/sys/devices/system/cpu")
    indices: list[int] = []
    if not cpu_root.exists():
        return indices
    for path in cpu_root.iterdir():
        name = path.name
        if name.startswith("cpu") and name[3:].isdigit():
            indices.append(int(name[3:]))
    return sorted(indices)


def _read_text(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _cpu_governors() -> list[str]:
    governors = {
        value
        for cpu in _cpu_indices()
        if (value := _read_text(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")))
    }
    return sorted(governors)


def _cpu_frequency_mhz() -> dict[str, float] | None:
    values: list[float] = []
    for cpu in _cpu_indices():
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
        raw = _read_text(base / "cpuinfo_cur_freq") or _read_text(base / "scaling_cur_freq")
        if raw is None:
            continue
        try:
            values.append(float(raw) / 1000.0)
        except ValueError:
            continue
    if not values:
        return None
    return {
        "min_observed_mhz": min(values),
        "max_observed_mhz": max(values),
        "sampled_cpu_count": float(len(values)),
    }


def measurement_context(load_average_before: list[float] | None) -> dict[str, object]:
    """Return reproducibility metadata for local benchmark artefacts."""

    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    return {
        "host_load": "workstation under concurrent load during capture",
        "cpu_isolation": (
            "taskset affinity only when launched with taskset; no kernel-reserved "
            "isolated cores were detected on this workstation"
        ),
        "cpu_affinity": affinity,
        "load_average_before": load_average_before,
        "load_average_after": load_average(),
        "cpu_governor": _cpu_governors() or "unavailable",
        "cpu_frequency_mhz": _cpu_frequency_mhz() or "unavailable",
        "concurrent_load_status": (
            "non-exclusive workstation run; other heavy jobs may have been active"
        ),
        "timing_interpretation": (
            "use timing medians as local regression context only, not final throughput claims"
        ),
        "production_rerun_requirement": (
            "rerun on reserved isolated cores with recorded affinity, governor, "
            "frequency, versions, and host-load evidence before publishing performance claims"
        ),
    }
