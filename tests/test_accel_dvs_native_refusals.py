# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — An interrupted native DVS read leaves no reader behind

"""An interruption while a native DVS reader runs kills and reaps that reader.

The reader is a real operator command that blocks. A real interval timer
interrupts the parent while it waits, as a stop request would.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from sc_neurocore.accel.dvs_native import read_native_recording


class _Interrupted(Exception):
    """Raised by the timer's handler in the waiting parent."""


@pytest.fixture
def interval_interrupt() -> Iterator[None]:
    """Deliver one real SIGALRM shortly after the read starts, then restore the handler."""

    def interrupt(_number: int, _frame: object) -> None:
        raise _Interrupted

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def test_an_interrupted_read_kills_and_reaps_its_reader(
    tmp_path: Path, interval_interrupt: None
) -> None:
    """The reader is gone when the interruption reaches the caller."""
    identity = tmp_path / "reader.pid"
    reader = tmp_path / "blocking-reader"
    reader.write_text(f"#!/bin/sh\necho $$ > {identity}\nexec sleep 60\n", encoding="utf-8")
    reader.chmod(0o700)
    with pytest.raises(_Interrupted):
        read_native_recording(tmp_path / "recording.npy", 1024, reader, backend="go")
    deadline = time.monotonic() + 5.0
    while not identity.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(ProcessLookupError):
        os.kill(int(identity.read_text()), 0)
