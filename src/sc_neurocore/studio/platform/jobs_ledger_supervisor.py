# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job supervisor identity

"""Who is supervising a job, and is that process still there.

A lease is only useful if its holder can be checked. The identity carries the
host, the process id and a token that changes when a process id is reused, so a
recycled id cannot inherit another supervisor's jobs. Liveness is answered with
three values, not two: a foreign host cannot be probed from here, and saying
"unknown" is more honest than declaring a running job dead.
"""

from __future__ import annotations

import os
import socket


def supervisor_identity(pid: int | None = None) -> str:
    """Return host/PID/start identity for this process or an observed local child.

    Omit pid for the current process. An explicit pid is a caller observation,
    not an authenticated assertion. Unavailable metadata yields start token0;
    registration must refuse that unknown identity instead of certifying it.
    """
    target = os.getpid() if pid is None else pid
    return f"{socket.gethostname()}:{target}:{_process_start_token(target)}"


def _process_start_token(pid: int | None = None) -> str:
    """Return a token that changes when a process id is reused."""
    target = os.getpid() if pid is None else pid
    try:
        with open(f"/proc/{target}/stat", encoding="utf-8", errors="replace") as handle:
            fields = handle.read().rsplit(")", 1)[-1].split()
        return str(fields[19])
    except OSError:
        # Exited and reaped, or no Linux process metadata: unknown generation.
        return "0"


def supervisor_is_alive(identity: str) -> bool | None:
    """Return whether a supervisor is running, or ``None`` when unknowable.

    Parameters
    ----------
    identity : str
        An identity produced by :func:`supervisor_identity`.

    Returns
    -------
    bool or None
        ``True`` when the process is running, ``False`` when it provably is
        not, and ``None`` when this host cannot tell — a malformed identity, a
        different host, or a platform without process metadata. A zero or
        malformed start token is unknown, not evidence of process death.
        A different UID may deny the signal probe; readable matching proc
        metadata still proves this exact process generation is alive.
    """
    try:
        host, pid_text, token = identity.split(":", 2)
    except ValueError:
        return None
    if host != socket.gethostname():
        return None
    if not token.isascii() or not token.isdecimal() or token.startswith("0"):
        return None
    try:
        pid = int(pid_text)
    except ValueError:
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Distinct storage/API UIDs may deny signals while exposing /proc stat.
        # The matching process-start token below still proves this generation.
        pass
    except OverflowError:
        return None
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as handle:
            fields = handle.read().rsplit(")", 1)[-1].split()
        state, started = fields[0], fields[19]
    except FileNotFoundError:
        # A process can disappear between kill(0) and reading its metadata.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return None
        return None
    except OSError:
        # For example a hidden or unreadable proc entry of another identity.
        return None
    return state not in {"Z", "X", "x"} and started == token


__all__ = ["supervisor_identity", "supervisor_is_alive"]
