# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — High-level Python binding targeting AER Interconnect Router

"""Supervise the Go AER UDP router: build it on demand, start it and stop it.

The router itself is the Go service under ``accel/go/services/aer_router``.
This module only runs its build, starts the binary in the background and
terminates it. Progress is reported through the module logger, not printed.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

_LOG = logging.getLogger(__name__)

_PACKAGE_ROUTER_DIR = (
    Path(__file__).resolve().parent.parent / "accel" / "go" / "services" / "aer_router"
)


class AERRoutingDaemon:
    """Supervise the Go AER UDP mesh router that links several FPGA boards.

    Parameters
    ----------
    port : int, optional
        UDP port the router is reported to listen on.
    router_dir : pathlib.Path, optional
        Directory holding the router's ``main.go`` and its ``aer_router``
        binary; the router shipped with the package when omitted.
    """

    def __init__(self, port: int = 9000, router_dir: Path | None = None) -> None:
        """Remember the port and router directory; nothing is built or started yet."""
        self._router_dir = _PACKAGE_ROUTER_DIR if router_dir is None else router_dir
        self._port = port
        self._process: subprocess.Popen[bytes] | None = None

    def start(self, build: bool = True) -> None:
        """Build the router if asked, then start it in the background.

        Parameters
        ----------
        build : bool, optional
            Run ``go build`` in the router directory before starting.

        Raises
        ------
        subprocess.CalledProcessError
            The Go build fails.
        OSError
            The router binary cannot be started.
        """
        if build:
            _LOG.info("Building the Go AER router in %s", self._router_dir)
            subprocess.run(
                ["go", "build", "-o", "aer_router", "main.go"],
                cwd=str(self._router_dir),
                check=True,
            )

        _LOG.info("Starting the AER router on UDP port %d", self._port)
        self._process = subprocess.Popen(
            ["./aer_router"],
            cwd=str(self._router_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)

    def stop(self) -> None:
        """Terminate the running router and wait up to two seconds for it to exit."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=2.0)
            self._process = None
            _LOG.info("Stopped the AER router")
