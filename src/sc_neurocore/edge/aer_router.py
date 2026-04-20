# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — High-level Python binding targeting AER Interconnect Router

import subprocess
import time
from pathlib import Path


class AERRoutingDaemon:
    """Orchestrates the Go-based AER UDP mesh multi-FPGA router pipeline dynamically."""

    def __init__(self, port: int = 9000):
        self._router_dir = Path(__file__).resolve().parent.parent / "accel" / "go" / "services" / "aer_router"
        self._port = port
        self._process = None

    def start(self, build: bool = True):
        if build:
            print("[AER Router] Natively compiling robust Go pipeline...")
            subprocess.run(["go", "build", "-o", "aer_router", "main.go"], cwd=str(self._router_dir), check=True)
            
        print(f"[AER Router] Spawning background listener on port {self._port}...")
        self._process = subprocess.Popen(
            ["./aer_router"],
            cwd=str(self._router_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)

    def stop(self):
        """Tears down the active background UDP topology safely."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=2.0)
            self._process = None
            print("[AER Router] Daemon successfully shut down.")
