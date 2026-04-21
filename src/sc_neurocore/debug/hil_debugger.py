# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — High-level HIL Debugger wrapper

from sc_neurocore.debug.hil_server import HILServerDaemon


class HILDebugger:
    """High-level wrapper for the HIL telemetry server."""

    def __init__(self, port: int = 8081) -> None:
        self.daemon = HILServerDaemon(port=port)

    def start(self) -> bool:
        """Starts the HIL debugger server."""
        return self.daemon.start()

    def stop(self) -> None:
        """Stops the HIL debugger server."""
        self.daemon.stop()

    @property
    def is_running(self) -> bool:
        """Returns True if the server is active."""
        return self.daemon.is_running

    @property
    def url(self) -> str:
        """Returns the base URL for the active telemetry server."""
        return f"http://localhost:{self.daemon.port}"
