# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native event-recording decoding

"""Read one native SHD row in a fresh process with no imported Julia runtime."""

from __future__ import annotations

import ctypes
import os
import struct
import sys
import threading
import time


class _Recording(ctypes.Structure):
    """Match the owning native shared-library SHD result structure."""

    _fields_ = [
        ("events", ctypes.POINTER(ctypes.c_double)),
        ("value_count", ctypes.c_size_t),
        ("label", ctypes.c_int64),
    ]


def _read(library_path: str, recording_path: str, index: int, maximum: int) -> None:
    """Load one operator library, verify its result, write bytes, and free ownership."""
    library = ctypes.CDLL(library_path)
    library.shd_read_c.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(_Recording),
    ]
    library.shd_read_c.restype = ctypes.c_int
    library.shd_free_c.argtypes = [ctypes.POINTER(_Recording)]
    library.shd_free_c.restype = None
    recording = _Recording()
    try:
        code = library.shd_read_c(
            os.fsencode(recording_path), index, maximum, ctypes.byref(recording)
        )
        if code != 0:
            raise RuntimeError("native SHD reader refused the recording")
        if recording.value_count % 4 or recording.value_count > maximum // 8:
            raise RuntimeError("native SHD reader returned an invalid event size")
        if recording.value_count and not recording.events:
            raise RuntimeError("native SHD reader returned no event buffer")
        data = (
            ctypes.string_at(recording.events, recording.value_count * 8)
            if recording.value_count
            else b""
        )
        sys.stdout.buffer.write(
            struct.pack("<4sQq", b"SHD1", recording.value_count, recording.label)
        )
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    finally:
        library.shd_free_c(ctypes.byref(recording))


def _guard(parent: int) -> None:
    """Stop the complete native process if its parent dies or its lifetime expires."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and os.getppid() == parent:
        time.sleep(0.1)
    os._exit(124)


if __name__ == "__main__":
    try:
        threading.Thread(target=_guard, args=(int(sys.argv[5]),), daemon=True).start()
        _read(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
