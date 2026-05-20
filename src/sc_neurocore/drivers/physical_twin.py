# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware-in-the-loop twin bridge

from __future__ import annotations

import json
import logging
import socket
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PhysicalTwinBridge:
    """Synchronise software neuron state with an explicit twin backend.

    ``mode="EMULATION"`` is a deterministic local noise model for development
    and CI. ``mode="TCP"`` opens a JSON-line request/response connection for a
    real hardware-twin service. The class never marks itself connected to
    physical hardware unless a TCP exchange actually succeeds.
    """

    def __init__(
        self,
        ip: str = "192.168.2.99",
        port: int = 5000,
        *,
        mode: str = "EMULATION",
        timeout_s: float = 1.0,
        seed: int = 42,
        noise_sigma: float = 0.01,
        divergence_threshold: float = 0.1,
    ) -> None:
        mode = mode.upper()
        if mode not in {"EMULATION", "TCP"}:
            raise ValueError("PhysicalTwinBridge mode must be 'EMULATION' or 'TCP'")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative")
        if divergence_threshold <= 0:
            raise ValueError("divergence_threshold must be positive")

        self.ip = ip
        self.port = port
        self.mode = mode
        self.timeout_s = timeout_s
        self.noise_sigma = noise_sigma
        self.divergence_threshold = divergence_threshold
        self.connected = mode == "EMULATION"
        self._rng = np.random.default_rng(seed)

    def sync_step(self, sw_v_mem: float, sw_spike: int) -> float:
        """Send software state and return the twin membrane voltage."""
        if self.mode == "TCP":
            return self._sync_step_tcp(sw_v_mem, sw_spike)

        hw_v_mem = sw_v_mem + float(self._rng.normal(0.0, self.noise_sigma))
        self._log_divergence(sw_v_mem, hw_v_mem)
        return hw_v_mem

    def _sync_step_tcp(self, sw_v_mem: float, sw_spike: int) -> float:
        request = (
            json.dumps(
                {"v_mem": float(sw_v_mem), "spike": int(sw_spike)},
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )

        try:
            with socket.create_connection((self.ip, self.port), timeout=self.timeout_s) as sock:
                sock.sendall(request)
                response = next(sock.makefile("r", encoding="utf-8"))
        except StopIteration as exc:
            self.connected = False
            raise ConnectionError("hardware twin closed connection without a reply") from exc
        except OSError as exc:
            self.connected = False
            raise ConnectionError(f"hardware twin connection failed: {exc}") from exc

        hw_v_mem = self._parse_reply(response)
        self.connected = True
        self._log_divergence(sw_v_mem, hw_v_mem)
        return hw_v_mem

    @staticmethod
    def _parse_reply(response: str) -> float:
        try:
            payload: Any = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("hardware twin reply is not valid JSON") from exc

        value = payload.get("v_mem") if isinstance(payload, dict) else None
        if not isinstance(value, int | float):
            raise ValueError("hardware twin reply missing numeric 'v_mem'")
        return float(value)

    def _log_divergence(self, sw_v_mem: float, hw_v_mem: float) -> None:
        diff = abs(sw_v_mem - hw_v_mem)
        if diff > self.divergence_threshold:
            logger.warning(
                "hardware twin divergence detected: software_v_mem=%.6f hardware_v_mem=%.6f diff=%.6f",
                sw_v_mem,
                hw_v_mem,
                diff,
            )
