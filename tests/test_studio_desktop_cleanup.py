# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop cleanup under actual signal refusal

"""Keep an actual live backend owned when the kernel refuses cleanup signals."""

from __future__ import annotations

import errno
import fcntl
import signal
import threading
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop.runtime import DesktopBackend
from sc_neurocore.studio.platform.jobs_reaper import ReapReport
from tests.studio_seccomp_support import Refusal, install_refusals
from tests.test_studio_desktop_runtime import _environment, _request


def test_refused_cleanup_retains_the_backend_until_unrestricted_retry(tmp_path: Path) -> None:
    """Return an honest unreaped report and retry the same real child from its owner.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Private profile retaining the actual backend configuration and user data.
    """
    backend = DesktopBackend(_environment(tmp_path))
    reports: list[ReapReport | None] = []

    def refused_close() -> None:
        """Refuse real termination syscalls only in this short-lived cleanup thread."""
        install_refusals(
            [
                Refusal("kill", errno.EPERM, signal.SIGTERM, argument_index=1),
                Refusal("kill", errno.EPERM, signal.SIGKILL, argument_index=1),
            ]
        )
        reports.append(backend.close())

    try:
        instance = backend.start()
        configuration = backend.paths.config / "desktop.json"
        original = configuration.read_bytes()
        worker = threading.Thread(target=refused_close, daemon=True)
        worker.start()
        worker.join(timeout=40)
        assert not worker.is_alive()
        (refused,) = reports
        assert refused is not None and not refused.reaped
        assert refused.outcome == "unreaped" and instance.pid in refused.survivors
        assert backend.returncode is None
        assert _request(instance.origin, "/api/health") == {"status": "ok"}
        assert configuration.read_bytes() == original
        with (
            (backend.paths.state / "backend.lock").open("a+b") as lock,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        collected = backend.close()
        assert collected is not None and collected.reaped and collected.returncode == 0
    assert backend.returncode == 0 and configuration.read_bytes() == original
    assert backend.close() is collected
    with (backend.paths.state / "backend.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
