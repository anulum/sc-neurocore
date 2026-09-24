# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Non-overwriting purge namespace moves

"""Atomic destination exclusion for Linux purge staging and restoration."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path


def sync_directory(path: Path) -> None:
    """Persist directory entry changes or propagate the OS error to recovery."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def move_without_replace(source: Path, destination: Path) -> bool:
    """Move a directory atomically, returning False if the target already exists.

    Requires libc/kernel/filesystem renameat2 RENAME_NOREPLACE support. Other
    errors propagate to the caller's recovery handling; never fall back to a
    check-then-rename operation that could overwrite a concurrent destination.
    This excludes destination replacement, not concurrent source substitution.
    """
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, "Atomic non-overwriting rename is unavailable") from exc
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return True
    code = ctypes.get_errno()
    if code in {errno.EEXIST, errno.ENOTEMPTY}:
        return False
    raise OSError(code, os.strerror(code), str(source), None, str(destination))
