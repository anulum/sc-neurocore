# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HIL Debugger Server Daemon

"""Hardware-in-the-Loop server orchestrator.

Spawns and manages the standalone high-performance Go-based WebSocket
telemetry server for real-time SC debugging and visualization.
"""

from __future__ import annotations

import http.client
import os
import subprocess
import sysconfig
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class HILServerDaemon:
    """Manages the background execution of the Go HIL Debugger service."""

    port: int = 8081
    _process: Optional[subprocess.Popen[bytes]] = None
    _go_dir: Path = Path(__file__).parent.parent / "accel" / "go" / "services" / "hil_debugger"

    def __post_init__(self) -> None:
        if self._go_dir.exists():
            return
        # Fallback for installed package
        installed_bin = Path(sysconfig.get_path("scripts")) / "hil_debugger"
        if installed_bin.exists():
            self._go_dir = installed_bin.parent
            return
        raise FileNotFoundError(
            "HIL Debugger Go binary not found. "
            "Run: cd accel/go/services/hil_debugger && go build -o hil_debugger main.go"
        )

    def start(self, build: bool = True) -> bool:
        """Compile and start the standalone HIL Debugger service."""
        if self._process and self._process.poll() is None:
            return True  # Already running

        if build and self._go_dir.is_dir():  # only build from source
            print("[HIL Daemon] Compiling high-performance Go telemetry server...")
            try:
                subprocess.run(
                    ["go", "build", "-o", "hil_debugger", "main.go"],
                    cwd=str(self._go_dir),
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"[HIL Daemon] Build failed: {e.stderr.decode()}")
                return False

        bin_path = self._go_dir / "hil_debugger"
        if not bin_path.exists():
            print(f"[HIL Daemon] Binary {bin_path} not found.")
            return False

        env = os.environ.copy()
        env["HIL_PORT"] = str(self.port)

        self._process = subprocess.Popen(
            [str(bin_path)],
            cwd=str(self._go_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for the service to bind and report health
        return self._wait_for_ready()

    def _wait_for_ready(self, timeout_sec: int = 5) -> bool:
        start_time = time.time()
        assert self._process is not None, "_wait_for_ready called before start()"
        while time.time() - start_time < timeout_sec:
            if self._process.poll() is not None:
                err = (
                    self._process.stderr.read().decode()
                    if self._process.stderr
                    else "unknown crash"
                )
                print(f"[HIL Daemon] Server crashed: {err}")
                return False
            conn = http.client.HTTPConnection("localhost", self.port, timeout=0.5)
            try:
                conn.request("GET", "/health")
                response = conn.getresponse()
                if response.status == 200:
                    print(f"[HIL Daemon] Server ready on port {self.port}.")
                    return True
            except (ConnectionError, TimeoutError, OSError):
                pass
            finally:
                conn.close()
            time.sleep(0.1)
        print("[HIL Daemon] Timeout waiting for readiness.")
        self.stop()
        return False

    def stop(self) -> None:
        """Gracefully terminate the background HIL debugger process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            print("[HIL Daemon] Server stopped.")

    @property
    def is_running(self) -> bool:
        """Returns True if the daemon process is running."""
        return self._process is not None and self._process.poll() is None
