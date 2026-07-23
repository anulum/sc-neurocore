# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_serve_server.py

from __future__ import annotations

import json
import time
import urllib.request
import numpy as np
import pytest
from sc_neurocore.serve import SpikeServer
class MockNetwork:
    """Mock network with step() interface."""

    def __init__(self):
        self.call_count = 0

    def step(self, inputs: dict) -> dict:
        self.call_count += 1
        return {k: v * 0.5 for k, v in inputs.items()}

    def reset(self):
        self.call_count = 0
class MockPopulation:
    def __init__(self, label, n):
        self.label = label
        self.n = n
        self.voltages = np.zeros(n)

    def step_all(self, currents):
        return (currents > 0.5).astype(np.int8)
class MockPopNetwork:
    """Mock network with populations interface."""

    def __init__(self):
        self.populations = [MockPopulation("exc", 3), MockPopulation("inh", 2)]
_HTTP_PORT = 18901
@pytest.fixture(scope="module")
def running_server():
    net = MockNetwork()
    server = SpikeServer(net, host="127.0.0.1", port=_HTTP_PORT)
    server.start(blocking=False)
    time.sleep(0.5)
    yield server
    server.stop()
def _post(path, data):
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
def _get(path):
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

__all__ = ['json', 'time', 'urllib', 'np', 'pytest', 'SpikeServer', 'MockNetwork', 'MockPopulation', 'MockPopNetwork', '_HTTP_PORT', 'running_server', '_post', '_get']
