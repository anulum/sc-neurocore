# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.serve (SpikeServer)

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


class TestSpikeServerStep:
    def test_step_with_sc_network(self):
        net = MockNetwork()
        server = SpikeServer(net)
        result = server.step({"input": [1.0, 2.0, 3.0]})
        assert "outputs" in result
        assert "timestep" in result
        assert result["timestep"] == 1
        np.testing.assert_array_almost_equal(
            result["outputs"]["input"],
            [0.5, 1.0, 1.5],
        )

    def test_step_increments_timestep(self):
        net = MockNetwork()
        server = SpikeServer(net)
        server.step({"input": [1.0]})
        server.step({"input": [2.0]})
        result = server.step({"input": [3.0]})
        assert result["timestep"] == 3

    def test_step_with_population_network(self):
        net = MockPopNetwork()
        server = SpikeServer(net)
        result = server.step({"exc": [1.0, 0.0, 1.0]})
        assert "outputs" in result
        assert "exc" in result["outputs"]

    def test_step_unsupported_network(self):
        server = SpikeServer(object())
        with pytest.raises(TypeError, match="Unsupported"):
            server.step({"input": [1.0]})


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


class TestSpikeServerHTTP:
    def test_health(self, running_server):
        result = _get("/health")
        assert result["status"] == "ok"

    def test_info_get(self, running_server):
        result = _get("/info")
        assert "timestep" in result
        assert result["type"] == "MockNetwork"

    def test_step_endpoint(self, running_server):
        result = _post("/step", {"inputs": {"input": [1.0, 2.0]}})
        assert "outputs" in result
        assert result["timestep"] >= 1

    def test_reset_endpoint(self, running_server):
        time.sleep(0.2)
        result = _post("/reset", {})
        assert result["status"] == "reset"
        assert result["timestep"] == 0

    def test_info_post(self, running_server):
        result = _post("/info", {})
        assert "timestep" in result

    def test_not_found_post(self, running_server):
        try:
            _post("/nonexistent", {})
            pytest.fail("Should have raised")
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_not_found_get(self, running_server):
        try:
            _get("/nonexistent")
            pytest.fail("Should have raised")
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_bad_json(self, running_server):
        req = urllib.request.Request(
            f"http://127.0.0.1:{_HTTP_PORT}/step",
            data=b"not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            pytest.fail("Should have raised")
        except urllib.error.HTTPError as e:
            assert e.code == 400
