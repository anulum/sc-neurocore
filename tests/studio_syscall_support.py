# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — kernel-held system calls for deterministic races

"""Hold chosen system calls of one child interpreter at an exact point.

The kernel's seccomp user notification suspends a thread inside a chosen call
until a notifier answers. The notifier here is a thread of the same child,
started before the filter so it is not filtered itself. For each held call it
may first perform a real competing action, for example replacing a directory
or committing a row through another connection, and then let the call
proceed; or it answers with an errno, which the product sees exactly as a
kernel refusal. Product code is not replaced or instrumented.

Install only inside a child interpreter: a seccomp filter cannot be removed.
Only calls made without the interpreter lock can be held: SQLite and most
``os`` file functions release it, but ``os.kill`` and ``os.killpg`` do not, so
refuse those with :func:`tests.studio_seccomp_support.install_refusals`.
"""

from __future__ import annotations

from collections.abc import Callable
import ctypes
from dataclasses import dataclass
import json
import os
import struct
import sys
import threading
from typing import NoReturn

import coverage

from tests.studio_seccomp_support import SECCOMP_AVAILABLE

__all__ = ["SECCOMP_AVAILABLE", "HeldCall", "finish", "hold_system_calls"]

# x86_64 numbers from ``asm/unistd_64.h``.
_SYSCALLS: dict[str, int] = {
    "read": 0,
    "write": 1,
    "pread64": 17,
    "pwrite64": 18,
    "kill": 62,
    "fcntl": 72,
    "fchmodat": 268,
    "fchmodat2": 452,
    "fsync": 74,
    "fdatasync": 75,
    "rename": 82,
    "mkdir": 83,
    "rmdir": 84,
    "getdents64": 217,
    "openat": 257,
    "mkdirat": 258,
    "unlinkat": 263,
    "renameat": 264,
    "renameat2": 316,
}
_NAMES = {number: name for name, number in _SYSCALLS.items()}
_SECCOMP = 317
_SET_MODE_FILTER = 1
_NEW_LISTENER = 1 << 3
_PR_SET_NO_NEW_PRIVS = 38
_AUDIT_ARCH_X86_64 = 0xC000003E
_RETURN_ALLOW = 0x7FFF0000
_RETURN_NOTIFY = 0x7FC00000
_RECEIVE = 0xC0502100
_SEND = 0xC0182101
_CONTINUE = 1


@dataclass(frozen=True)
class HeldCall:
    """One system call the kernel is holding for an answer.

    ``thread`` is the native ID of the calling thread.
    """

    name: str
    arguments: tuple[int, ...]
    thread: int

    def text(self, index: int) -> str:
        """Return the NUL-terminated string argument ``index`` of the held thread."""
        raw = ctypes.string_at(self.arguments[index])
        return os.fsdecode(raw)

    def memory(self, index: int, length: int) -> bytes:
        """Return ``length`` bytes the pointer argument ``index`` addresses."""
        return ctypes.string_at(self.arguments[index], length)

    def descriptor_path(self, index: int) -> str:
        """Return what descriptor argument ``index`` currently names."""
        return os.readlink(f"/proc/self/fd/{ctypes.c_int(self.arguments[index]).value}")


class _Notification(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("pid", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("nr", ctypes.c_int32),
        ("arch", ctypes.c_uint32),
        ("ip", ctypes.c_uint64),
        ("args", ctypes.c_uint64 * 6),
    ]


class _Response(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("val", ctypes.c_int64),
        ("error", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


class _FilterProgram(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("instructions", ctypes.c_void_p)]


def _instruction(code: int, true_jump: int, false_jump: int, value: int) -> bytes:
    return struct.pack("HBBI", code, true_jump, false_jump, value)


def _program(names: list[str]) -> bytes:
    program = [
        _instruction(0x20, 0, 0, 4),
        _instruction(0x15, 1, 0, _AUDIT_ARCH_X86_64),
        _instruction(0x06, 0, 0, _RETURN_ALLOW),
        _instruction(0x20, 0, 0, 0),
    ]
    for name in names:
        program.append(_instruction(0x15, 0, 1, _SYSCALLS[name]))
        program.append(_instruction(0x06, 0, 0, _RETURN_NOTIFY))
    program.append(_instruction(0x06, 0, 0, _RETURN_ALLOW))
    return b"".join(program)


def hold_system_calls(names: list[str], decide: Callable[[HeldCall], int | None]) -> None:
    """Route ``names`` of the calling thread, and threads it starts, to ``decide``.

    ``decide`` runs on the notifier thread for every held call. Returning
    ``None`` lets the call proceed unchanged; returning an errno fails it with
    that error. An exception in ``decide`` lets the call proceed and is
    re-raised in no thread, so ``decide`` should record what it observed.

    Raises
    ------
    OSError
        The kernel refused ``no_new_privs`` or the filter.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    libc.syscall.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_void_p]
    libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
    listener: list[int] = []
    ready = threading.Event()

    def serve() -> None:
        ready.wait()
        notification = _Notification()
        while True:
            ctypes.memset(ctypes.byref(notification), 0, ctypes.sizeof(notification))
            if libc.ioctl(listener[0], _RECEIVE, ctypes.byref(notification)) != 0:
                continue
            held = HeldCall(
                _NAMES.get(notification.nr, str(notification.nr)),
                tuple(notification.args),
                notification.pid,
            )
            try:
                error = decide(held)
            except Exception:
                error = None
            response = _Response(id=notification.id, val=0, error=0, flags=_CONTINUE)
            if error is not None:
                response = _Response(id=notification.id, val=0, error=-error, flags=0)
            libc.ioctl(listener[0], _SEND, ctypes.byref(response))

    # Started before the filter exists, so the notifier itself is never held.
    threading.Thread(target=serve, name="syscall-notifier", daemon=True).start()
    code = _program(names)
    buffer = ctypes.create_string_buffer(code, len(code))
    header = _FilterProgram(len(code) // 8, ctypes.cast(buffer, ctypes.c_void_p))
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    descriptor = libc.syscall(_SECCOMP, _SET_MODE_FILTER, _NEW_LISTENER, ctypes.addressof(header))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    listener.append(descriptor)
    ready.set()


def finish(result: dict[str, object]) -> NoReturn:
    """Print ``result`` and leave a child whose calls may still be held.

    Interpreter shutdown stops the notifier thread before late clean-up, such
    as SQLite closing its connections, runs; a held call there would never be
    answered. Coverage data is saved while the notifier still answers, then the
    process exits without that shutdown.
    """
    print(json.dumps(result), flush=True)
    sys.stderr.flush()
    measurement = coverage.Coverage.current()
    if measurement is not None:
        measurement.stop()
        measurement.save()
    os._exit(0)
