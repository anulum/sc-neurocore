# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — real kernel refusals for Studio fault paths

"""Make the kernel refuse chosen system calls inside one child interpreter.

A test runs product code in a fresh interpreter that first installs a seccomp
filter. The kernel then answers the named calls with the chosen errno, as it
does under a hardened service manager, a security module or a kernel without
the call. Nothing in the product is replaced, and the test process itself is
never filtered: the filter is installed with ``no_new_privs`` in the child and
is inherited only by that child's own descendants.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]

# x86_64 system call numbers from ``asm/unistd_64.h``.
_SYSCALLS: dict[str, int] = {
    "clone": 56,
    "execve": 59,
    "kill": 62,
    "prctl": 157,
    "fchmodat": 268,
    "pidfd_send_signal": 424,
    "pidfd_open": 434,
    "clone3": 435,
    "fchmodat2": 452,
}
# ``clone`` flags glibc uses for a new thread when ``clone3`` is unavailable.
THREAD_CLONE_FLAGS = 0x3D0F00
_AUDIT_ARCH_X86_64 = 0xC000003E
_LOAD_WORD = 0x20
_JUMP_EQUAL = 0x15
_RETURN = 0x06
_ALLOW = 0x7FFF0000
_ERRNO = 0x00050000
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2

SECCOMP_AVAILABLE = platform.system() == "Linux" and platform.machine() == "x86_64"


@dataclass(frozen=True)
class Refusal:
    """Answer ``syscall`` with ``error``, optionally only when one argument matches.

    ``first_argument`` is compared with argument ``argument_index`` (low 32 bits).
    """

    syscall: str
    error: int
    first_argument: int | None = None
    argument_index: int = 0


def _instruction(code: int, true_jump: int, false_jump: int, value: int) -> bytes:
    return struct.pack("HBBI", code, true_jump, false_jump, value)


def _program(refusals: list[Refusal]) -> bytes:
    """Build a classic BPF program that allows every call not refused."""
    program = [
        _instruction(_LOAD_WORD, 0, 0, 4),
        _instruction(_JUMP_EQUAL, 1, 0, _AUDIT_ARCH_X86_64),
        _instruction(_RETURN, 0, 0, _ALLOW),
    ]
    for refusal in refusals:
        number = _SYSCALLS[refusal.syscall]
        program.append(_instruction(_LOAD_WORD, 0, 0, 0))
        if refusal.first_argument is None:
            program.append(_instruction(_JUMP_EQUAL, 0, 1, number))
        else:
            program.append(_instruction(_JUMP_EQUAL, 0, 3, number))
            program.append(_instruction(_LOAD_WORD, 0, 0, 16 + 8 * refusal.argument_index))
            program.append(_instruction(_JUMP_EQUAL, 0, 1, refusal.first_argument))
        program.append(_instruction(_RETURN, 0, 0, _ERRNO | refusal.error))
    program.append(_instruction(_RETURN, 0, 0, _ALLOW))
    return b"".join(program)


class _FilterProgram(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("instructions", ctypes.c_void_p)]


def install_refusals(refusals: list[Refusal]) -> None:
    """Install the filter in the calling process; call only inside a child.

    Raises
    ------
    OSError
        The kernel refused ``no_new_privs`` or the filter.
    """
    code = _program(refusals)
    buffer = ctypes.create_string_buffer(code, len(code))
    header = _FilterProgram(len(code) // 8, ctypes.cast(buffer, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    for arguments in (
        (_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0),
        (_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.addressof(header), 0, 0),
    ):
        if libc.prctl(*arguments) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))


def run_child(
    source: str, *, arguments: tuple[str, ...] = (), expected_returncode: int = 0
) -> dict[str, object]:
    """Run ``source`` in a fresh interpreter and return its last-line JSON object.

    The child imports the repository's ``src`` and ``tests`` packages;
    ``sys.argv[1:]`` holds ``arguments``. A child expected to die with a
    signal or exit code returns ``{}`` after the code is checked.
    """
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(REPOSITORY / "src"), str(REPOSITORY))),
    }
    completed = subprocess.run(
        [sys.executable, "-c", source, *arguments],
        capture_output=True,
        check=False,
        cwd=REPOSITORY,
        env=environment,
        text=True,
        timeout=120.0,
    )
    assert completed.returncode == expected_returncode, completed.stderr
    if expected_returncode != 0:
        return {}
    result = json.loads(completed.stdout.splitlines()[-1])
    assert isinstance(result, dict)
    return result


def run_refused(
    source: str, refusals: list[Refusal], *, arguments: tuple[str, ...] = ()
) -> dict[str, object]:
    """Run ``source`` in a child that can install ``refusals``; return its JSON result.

    ``source`` does its own setup, then calls ``install_refusals(REFUSALS)``
    at the point the refusals must start, and prints one JSON object as its
    last output line. ``sys.argv[1:]`` holds ``arguments``.
    """
    prelude = (
        "import json, sys\n"
        "from tests.studio_seccomp_support import Refusal, install_refusals\n"
        "REFUSALS = [Refusal(*item) for item in json.loads(sys.argv.pop(1))]\n"
    )
    encoded = json.dumps(
        [[r.syscall, r.error, r.first_argument, r.argument_index] for r in refusals]
    )
    return run_child(prelude + source, arguments=(encoded, *arguments))
