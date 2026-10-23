# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native desktop configuration refusal

"""Show actual persistence refusal in the native window without losing user state."""

from __future__ import annotations

import fcntl
import os
import time

import pytest

from sc_neurocore.studio.desktop.runtime import DesktopBackend
from studio.desktop.tests.test_window import _close_button, _message, _shown, _wait
from tests.test_studio_jobs_guard_cleanup import _children, _runs

pytestmark = pytest.mark.usefixtures("application")


def test_readonly_configuration_is_visible_and_native_close_preserves_the_profile(
    desktop_backend: DesktopBackend,
) -> None:
    """Display a real configuration write refusal and close its admitted backend.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Actual child owner and private profile used by the native window.
    """
    previous = DesktopBackend(desktop_backend.environ)
    try:
        previous.start()
        configuration = previous.paths.config / "desktop.json"
        original = configuration.read_bytes()
    finally:
        closed = previous.close()
        assert closed is not None and closed.reaped and closed.returncode == 0
    with _shown(desktop_backend) as window:
        deadline = time.monotonic() + 30
        children: list[int] = []
        while not children:
            children = [
                pid
                for pid in _children(os.getpid())
                if _runs(pid, "sc_neurocore.studio.desktop.backend")
            ]
            assert time.monotonic() < deadline
            time.sleep(0.005)
        desktop_backend.paths.config.chmod(0o500)
        try:
            _wait(lambda: _message(window, "Studio could not save its desktop configuration."))
            assert window.isVisible() and window.instance is None and window.view is None
            assert desktop_backend.returncode == 0
            assert configuration.read_bytes() == original
            with (desktop_backend.paths.state / "backend.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _close_button(window)
            assert configuration.read_bytes() == original
        finally:
            desktop_backend.paths.config.chmod(0o700)
