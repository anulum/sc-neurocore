# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The host's x86-64 instruction-set level

"""Read the running host's x86-64 psABI micro-architecture level from Linux CPU flags.

The x86-64 psABI names four levels, each a superset of the one before: ``x86-64-v1`` (the baseline), ``x86-64-v2``,
``x86-64-v3`` and ``x86-64-v4``. A library built for one level may execute instructions a lower-level host lacks,
which ends the process; a measured backend order taken on one level does not describe another. This module answers
one question, "which level is this host?", from the flag names Linux prints in ``/proc/cpuinfo``.

The flag names are Linux's, not the psABI document's: LZCNT is printed as ``abm``, SSE3 as ``pni`` and the
LAHF/SAHF pair as ``lahf_lm``. OSXSAVE, which the psABI lists for ``x86-64-v3``, is not printed by Linux and is not
required; AVX and AVX2 appear only when the kernel has enabled their state. A host that is not x86-64 Linux, or whose
flags lack even the baseline, is ``"unknown"``: no level is ever guessed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import platform

#: The psABI levels, lowest first.
LEVELS: tuple[str, ...] = ("x86-64-v1", "x86-64-v2", "x86-64-v3", "x86-64-v4")

#: The flags each level adds to the one below it, under their Linux ``/proc/cpuinfo`` names.
LEVEL_FLAGS: Mapping[str, frozenset[str]] = {
    "x86-64-v1": frozenset({"cmov", "cx8", "fpu", "fxsr", "mmx", "syscall", "sse", "sse2"}),
    "x86-64-v2": frozenset({"cx16", "lahf_lm", "popcnt", "pni", "sse4_1", "sse4_2", "ssse3"}),
    "x86-64-v3": frozenset({"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe"}),
    "x86-64-v4": frozenset({"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}),
}

#: The machine names under which the Linux x86-64 flag names apply.
_X86_64_MACHINES = frozenset({"x86_64", "amd64"})

#: Where Linux prints the CPU flags.
CPUINFO = Path("/proc/cpuinfo")


def level_from_flags(flags: Iterable[str]) -> str:
    """Return the highest psABI level whose flags, and those of every level below it, are all present.

    Parameters
    ----------
    flags : iterable of str
        CPU flag names as Linux prints them in ``/proc/cpuinfo``.

    Returns
    -------
    str
        One of :data:`LEVELS`, or ``"unknown"`` when the baseline flags are not all present.
    """
    present = frozenset(flags)
    reached = "unknown"
    for level in LEVELS:
        if not LEVEL_FLAGS[level] <= present:
            break
        reached = level
    return reached


def _first_flags_line(text: str) -> frozenset[str] | None:
    """Return the flag names of the first ``flags`` line of ``/proc/cpuinfo`` text, or ``None`` when there is none."""
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() == "flags":
            return frozenset(value.split())
    return None


def host_level(
    *,
    cpuinfo: Path = CPUINFO,
    machine: str | None = None,
    system: str | None = None,
) -> str:
    """Return the running host's psABI level, or ``"unknown"``.

    Parameters
    ----------
    cpuinfo : Path
        The file to read; the running host's ``/proc/cpuinfo`` unless a test names another.
    machine, system : str, optional
        The machine and operating system names; :func:`platform.machine` and :func:`platform.system` when omitted.

    Returns
    -------
    str
        One of :data:`LEVELS`, or ``"unknown"`` on a host that is not x86-64 Linux, whose ``cpuinfo`` cannot be read
        or has no flags line, or whose flags lack the baseline. It never raises for those cases.
    """
    machine_name = (platform.machine() if machine is None else machine).lower()
    system_name = platform.system() if system is None else system
    if system_name != "Linux" or machine_name not in _X86_64_MACHINES:
        return "unknown"
    try:
        text = cpuinfo.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unknown"
    flags = _first_flags_line(text)
    if flags is None:
        return "unknown"
    return level_from_flags(flags)


def at_least(host: str, required: str) -> bool:
    """Say whether a host level meets a required level.

    Parameters
    ----------
    host : str
        The host's level, one of :data:`LEVELS` or ``"unknown"``.
    required : str
        The level a record or a library needs; one of :data:`LEVELS`.

    Returns
    -------
    bool
        ``True`` when ``host`` is ``required`` or above it. An ``"unknown"`` host meets no level.

    Raises
    ------
    ValueError
        When ``required`` is not one of :data:`LEVELS`, or ``host`` is neither a level nor ``"unknown"``.
    """
    if required not in LEVELS:
        raise ValueError(f"{required!r} is not a level; expected one of {LEVELS}")
    if host == "unknown":
        return False
    if host not in LEVELS:
        raise ValueError(f"{host!r} is not a level; expected one of {LEVELS} or 'unknown'")
    return LEVELS.index(host) >= LEVELS.index(required)


__all__ = ["CPUINFO", "LEVELS", "LEVEL_FLAGS", "at_least", "host_level", "level_from_flags"]
