# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_serve_server.py

"""Mock networks and HTTP helpers for the spike-server tests."""

from __future__ import annotations

import json
from typing import Any
import time
import urllib.request
import numpy as np
import pytest
from sc_neurocore.serve import SpikeServer


class MockNetwork:
    """Mock network with step() interface."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.call_count = 0

    def step(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Count the call and return each input halved."""
        self.call_count += 1
        return {k: v * 0.5 for k, v in inputs.items()}

    def reset(self) -> None:
        """Clear the call count."""
        self.call_count = 0


class MockPopulation:
    """Mock population with a label, a size and zero voltages."""

    def __init__(self, label: str, n: int) -> None:
        """Store the label and size and zero the voltages."""
        self.label = label
        self.n = n
        self.voltages = np.zeros(n)

    def step_all(self, currents: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Spike where the current exceeds 0.5."""
        return (currents > 0.5).astype(np.int8)


class MockPopNetwork:
    """Mock network with populations interface."""

    def __init__(self) -> None:
        """Hold an excitatory and an inhibitory mock population."""
        self.populations = [MockPopulation("exc", 3), MockPopulation("inh", 2)]


_HTTP_PORT = 18901


def _post(path: str, data: object) -> Any:
    """POST JSON to the test server and return the decoded reply, retrying twice."""
    body = json.dumps(data).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{_HTTP_PORT}{path}",
                data=body,
                headers={"Content-Type": "application/json", "Connection": "close"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except (ConnectionError, OSError):
            if attempt == 2:
                raise
            time.sleep(0.2)
    raise AssertionError("unreachable: the last attempt returns or re-raises")


def _get(path: str) -> Any:
    """GET from the test server and return the decoded reply, retrying twice."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{_HTTP_PORT}{path}",
                headers={"Connection": "close"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except (ConnectionError, OSError):
            if attempt == 2:
                raise
            time.sleep(0.2)
    raise AssertionError("unreachable: the last attempt returns or re-raises")


__all__ = [
    "json",
    "time",
    "urllib",
    "np",
    "pytest",
    "SpikeServer",
    "MockNetwork",
    "MockPopulation",
    "MockPopNetwork",
    "_HTTP_PORT",
    "_post",
    "_get",
]
