# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The running host's CPU model, read one way everywhere

"""Read the running host's CPU model string by one rule.

Backend dispatch, the dense IF comparison and Studio's benchmark contribution all compare this
string with the ``meta.cpu`` a benchmark recorded, so they read it the same way. The module
imports only the standard library: the conversion package uses it without loading the
acceleration package or any native runtime.
"""

from __future__ import annotations

import platform
from pathlib import Path

#: The Linux CPU description whose ``model name`` line names the model.
CPUINFO = Path("/proc/cpuinfo")


def cpu_model(cpuinfo: Path = CPUINFO) -> str:
    """Return the host CPU model string, matching the benchmark harness convention.

    Parameters
    ----------
    cpuinfo : Path
        The CPU description to read; the running host's ``/proc/cpuinfo`` unless a test
        names another file.

    Returns
    -------
    str
        The value of the first ``model name`` line that has one; else
        :func:`platform.processor`; else ``"unknown"``. A file that is absent or unreadable
        counts as having no ``model name`` line, and bytes that are not UTF-8 are replaced, so
        a bad byte elsewhere keeps the model line and this never raises.
    """
    try:
        info = cpuinfo.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # A bad byte elsewhere in the file must not cost the model line.
        try:
            info = cpuinfo.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            info = ""
    except OSError:
        info = ""
    for line in info.splitlines():
        if line.startswith("model name"):
            _, separator, value = line.partition(":")
            if separator and value.strip():
                return value.strip()
    return platform.processor() or "unknown"


__all__ = ["CPUINFO", "cpu_model"]
