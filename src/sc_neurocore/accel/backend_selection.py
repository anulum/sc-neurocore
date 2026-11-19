# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Data-driven polyglot backend ordering from recorded benchmarks

"""Reorder the polyglot dispatch order from current measurements for the running host.

The package ships ``measured_backend_orders.json`` beside this module. ``tools/generate_backend_speed_table.py``
writes it from the repository's benchmark records and admits a record only while the coverage census calls it
``current`` or ``reviewed``: a measurement whose bound sources have changed since it was taken orders nothing. Each
entry names a kernel, the measuring host's CPU model, that host's x86-64 psABI level (``"unrecorded"`` when the record
does not state its flags) and the lanes that ran, fastest first. The file is read through the package's resources,
so a source checkout and an installed wheel order lanes the same way.

For a kernel on the running host, :func:`select_backend_order` uses the entry measured on the same CPU model. An
entry with a recorded level applies only when this host is at that level or above; an unrecorded entry applies on the
exact model match alone. With no applicable entry the static order is returned verbatim. The always-available floor
(the static order's last entry, such as ``"python"`` or ``"numpy"``) is always kept as the final tier, so the dispatch
loop never loses its guaranteed fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
import functools
from importlib import resources
import json
from pathlib import Path
import platform
from typing import NamedTuple

from .backend_order import FASTEST_FIRST_BACKENDS
from .host_isa import LEVELS, at_least, host_level

#: The packaged table, beside this module.
TABLE_RESOURCE = "measured_backend_orders.json"

#: The level of an entry whose record does not state its host's CPU flags.
UNRECORDED = "unrecorded"


class MeasuredOrder(NamedTuple):
    """One admitted measurement of a kernel on one host model and level.

    Attributes
    ----------
    kernel : str
        The kernel identifier the benchmark record names.
    cpu_model : str
        The measuring host's CPU model string.
    isa_level : str
        The measuring host's psABI level, or ``"unrecorded"``.
    order : tuple of str
        The lanes that ran, fastest first.
    """

    kernel: str
    cpu_model: str
    isa_level: str
    order: tuple[str, ...]


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


def _read_table() -> str:
    """Return the packaged table's text."""
    return resources.files(__package__).joinpath(TABLE_RESOURCE).read_text(encoding="utf-8")


def _entry(item: object) -> MeasuredOrder | None:
    """Return one well-formed table entry, or ``None`` for anything else."""
    if not isinstance(item, Mapping):
        return None
    kernel, cpu, level, order = (
        item.get(key) for key in ("kernel", "cpu_model", "isa_level", "order")
    )
    if not (isinstance(kernel, str) and isinstance(cpu, str) and isinstance(order, list)):
        return None
    if level != UNRECORDED and level not in LEVELS:
        return None
    if not order or not all(isinstance(name, str) for name in order):
        return None
    return MeasuredOrder(kernel, cpu, str(level), tuple(order))


@functools.cache
def measured_orders() -> tuple[MeasuredOrder, ...]:
    """Return the admitted measurements of the packaged table.

    Cached: the packaged table is immutable for the life of the process. An absent or unreadable table gives no
    entries, so dispatch falls back to the static order; the repository's drift test and the installed-wheel test
    keep the table present and current.

    Returns
    -------
    tuple of MeasuredOrder
        The well-formed entries, in the table's order.
    """
    try:
        payload = json.loads(_read_table())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return ()
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return ()
    return tuple(entry for entry in (_entry(item) for item in entries) if entry is not None)


@functools.cache
def _running_level() -> str:
    """Return the running host's psABI level, read once per process."""
    return host_level()


def _applies(entry: MeasuredOrder, kernel: str, cpu: str, level: str) -> bool:
    """Say whether an entry describes this kernel on this host."""
    if entry.kernel != kernel or entry.cpu_model != cpu:
        return False
    return entry.isa_level == UNRECORDED or at_least(level, entry.isa_level)


def select_backend_order(
    kernel: str,
    *,
    static: tuple[str, ...] = FASTEST_FIRST_BACKENDS,
    cpu: str | None = None,
    isa_level: str | None = None,
) -> tuple[str, ...]:
    """Return the dispatch order for ``kernel``, reordered from current measurements of this host.

    Compiled backends measured on this host's CPU model lead, fastest-first; any backend in ``static`` without a
    measurement keeps its static position after them; the floor (``static[-1]``) is always last. Among applicable
    entries one with a recorded level is preferred to an unrecorded one, and a higher recorded level to a lower one.
    With no applicable entry the static order is returned verbatim.

    Parameters
    ----------
    kernel : str
        The kernel identifier, as the benchmark records name it.
    static : tuple of str
        The static order, floor last.
    cpu : str, optional
        The host CPU model; the running host's when omitted.
    isa_level : str, optional
        The host's psABI level or ``"unknown"``; the running host's when omitted.

    Returns
    -------
    tuple of str
        The dispatch order.
    """
    if not static:
        return static
    floor = static[-1]
    host = cpu if cpu is not None else current_cpu()
    level = isa_level if isa_level is not None else _running_level()
    applicable = [entry for entry in measured_orders() if _applies(entry, kernel, host, level)]
    if not applicable:
        return static
    applicable.sort(
        key=lambda entry: -1 if entry.isa_level == UNRECORDED else LEVELS.index(entry.isa_level),
        reverse=True,
    )
    measured = applicable[0].order
    compiled_fast = [name for name in measured if name != floor and name in static]
    rest = [name for name in static if name != floor and name not in compiled_fast]
    return (*compiled_fast, *rest, floor)


__all__ = [
    "MeasuredOrder",
    "TABLE_RESOURCE",
    "current_cpu",
    "measured_orders",
    "select_backend_order",
]
