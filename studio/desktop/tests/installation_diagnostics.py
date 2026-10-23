# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Main-thread installer diagnostics

"""Report slow real-runtime tests without sampling Python frames from a native thread."""

from __future__ import annotations

import os
import signal
import traceback
from collections.abc import Generator
from pathlib import Path
from types import FrameType

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    """Record a slow test's main-thread stack while retaining its original outcome.

    Parameters
    ----------
    item : pytest.Item
        Actual scenario whose normal setup, call and teardown are observed.

    Yields
    ------
    None
        Control to pytest's original test protocol. Its return or exception is
        preserved, and the previous signal handler and timer are restored.

    Returns
    -------
    object
        Original pytest protocol result.

    Notes
    -----
    The installer lane supplies SC_NEUROCORE_DESKTOP_STACK_LOG and an independent
    outer timeout. A Python signal can be deferred during C-only execution;
    this observer never replaces that timeout or the fatal-error handler.
    """
    with Path(os.environ["SC_NEUROCORE_DESKTOP_STACK_LOG"]).open("a", buffering=1) as output:

        def dump(signum: int, frame: FrameType | None) -> None:
            """Log the interrupted main-thread stack without changing the scenario.

            Parameters
            ----------
            signum : int
                Actual SIGALRM delivered by the scenario's diagnostic timer.
            frame : types.FrameType or None
                Main-thread Python frame at normal signal-handler dispatch.
            """
            output.write(f"45-second installer diagnostic: {item.nodeid}\n")
            traceback.print_stack(frame, file=output)
            output.flush()

        previous = signal.signal(signal.SIGALRM, dump)
        timer = signal.setitimer(signal.ITIMER_REAL, 45.0)
        try:
            return (yield)
        finally:
            signal.setitimer(signal.ITIMER_REAL, *timer)
            signal.signal(signal.SIGALRM, previous)
