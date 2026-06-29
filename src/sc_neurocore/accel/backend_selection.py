# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Data-driven polyglot backend ordering from recorded benchmarks

"""Reorder the polyglot dispatch order from measured benchmarks for the running host.

The repository ships per-backend throughput benchmarks under
``benchmarks/results/bench_*.json``, each carrying a ``kernel`` id, a ``meta.cpu``
host model string, and a ``backends`` map of ``median_call_ms`` per language. This
module reads those files and, for a given kernel on the running host's CPU, returns
the compiled backends ordered fastest-measured-first, falling back to the static
:data:`~sc_neurocore.accel.backend_order.FASTEST_FIRST_BACKENDS` when no measurement
matches the host+kernel. The always-available floor (the static order's last entry,
e.g. ``"python"`` or ``"numpy"``) is always preserved as the final tier so the
dispatch loop never loses its guaranteed fallback.

This closes the gap the SOTA register flagged: ~240 benchmark JSONs on disk while
the dispatch order was a hardcoded tuple that never consulted them.
"""

from __future__ import annotations

import functools
import json
import platform
from pathlib import Path
from typing import Any

from .backend_order import FASTEST_FIRST_BACKENDS

#: Repository root (``src/sc_neurocore/accel/`` → repo root is three parents up from src).
_REPO_ROOT = Path(__file__).resolve().parents[3]
#: Directory holding the recorded per-backend benchmark JSONs.
_RESULTS_DIR = _REPO_ROOT / "benchmarks" / "results"


def current_cpu() -> str:
    """Return the host CPU model string, matching the benchmark harness convention.

    Mirrors ``benchmarks/bench_*.py::_cpu_model`` so a live host matches a stored
    ``meta.cpu`` exactly: the ``/proc/cpuinfo`` ``model name`` line, else
    :func:`platform.processor`, else ``"unknown"``.
    """
    try:
        info = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        info = ""
    for line in info.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _backend_speed_order(backends: dict[str, Any]) -> list[str]:
    """Return backend names that ran, sorted by ascending ``median_call_ms``."""
    timed: list[tuple[float, str]] = []
    for name, entry in backends.items():
        if not isinstance(entry, dict):
            continue
        if not (entry.get("available") and entry.get("used")):
            continue
        ms = entry.get("median_call_ms")
        if not isinstance(ms, (int, float)):
            continue
        timed.append((float(ms), name))
    timed.sort(key=lambda pair: pair[0])
    return [name for _ms, name in timed]


@functools.cache
def measured_orders() -> dict[str, dict[str, tuple[str, ...]]]:
    """Build ``{cpu: {kernel: (backends fastest-measured-first)}}`` from the JSONs.

    Cached: the on-disk benchmarks are immutable for the life of the process.
    Malformed or non-comparison files (no ``backends`` / ``kernel`` / ``meta.cpu``)
    are skipped silently — they are simply not a usable measurement.
    """
    table: dict[str, dict[str, tuple[str, ...]]] = {}
    if not _RESULTS_DIR.is_dir():
        return table
    for path in sorted(_RESULTS_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        backends = raw.get("backends")
        kernel = raw.get("kernel")
        cpu = raw.get("meta", {}).get("cpu") if isinstance(raw.get("meta"), dict) else None
        if not (isinstance(backends, dict) and isinstance(kernel, str) and isinstance(cpu, str)):
            continue
        order = _backend_speed_order(backends)
        if order:
            table.setdefault(cpu, {})[kernel] = tuple(order)
    return table


def select_backend_order(
    kernel: str,
    *,
    static: tuple[str, ...] = FASTEST_FIRST_BACKENDS,
    cpu: str | None = None,
) -> tuple[str, ...]:
    """Return the dispatch order for ``kernel``, reordered from measured benchmarks.

    Compiled backends measured on this host's CPU lead, fastest-first; any backend
    in ``static`` without a measurement keeps its static position after them; the
    floor (``static[-1]``) is always last. With no matching measurement the static
    order is returned verbatim.
    """
    if not static:
        return static
    floor = static[-1]
    host = cpu if cpu is not None else current_cpu()
    measured = measured_orders().get(host, {}).get(kernel)
    if not measured:
        return static
    compiled_fast = [name for name in measured if name != floor and name in static]
    rest = [name for name in static if name != floor and name not in compiled_fast]
    return (*compiled_fast, *rest, floor)


__all__ = ["current_cpu", "measured_orders", "select_backend_order"]
