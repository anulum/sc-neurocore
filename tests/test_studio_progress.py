# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio WebSocket progress streaming

from __future__ import annotations

import queue

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.progress import (
    _characterize_with_progress,
    _heatmap_with_progress,
    _scan_with_progress,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


# --- Characterise with progress ---


class TestCharacteriseProgress:
    def test_produces_progress_and_complete(self):
        q: queue.Queue = queue.Queue()

        def sim_fn(**kw):
            import numpy as np

            n = 100
            return {
                "time": list(range(n)),
                "states": {"v": list(np.zeros(n))},
                "spikes": [],
                "spike_count": 0,
                "stats": {"rate_hz": 0, "isi_mean_ms": None, "isi_cv": None, "isi_histogram": None},
                "dt": 0.1,
                "n_steps": n,
                "current_trace": list(np.zeros(n)),
            }

        _characterize_with_progress(sim_fn, {"current": 10.0, "params": {"a": 1.0}}, q)

        messages = []
        while not q.empty():
            messages.append(q.get_nowait())

        types = [m["type"] for m in messages]
        assert "progress" in types
        assert types[-1] == "complete"

        complete = messages[-1]
        assert "result" in complete
        assert "fi_curve" in complete["result"]
        assert "pattern" in complete["result"]

    def test_progress_has_pct(self):
        q: queue.Queue = queue.Queue()

        def sim_fn(**kw):
            import numpy as np

            n = 50
            return {
                "time": list(range(n)),
                "states": {"v": list(np.zeros(n))},
                "spikes": [],
                "spike_count": 0,
                "stats": {"rate_hz": 0, "isi_mean_ms": None, "isi_cv": None, "isi_histogram": None},
                "dt": 0.1,
                "n_steps": n,
                "current_trace": list(np.zeros(n)),
            }

        _characterize_with_progress(sim_fn, {"current": 5.0}, q)

        progress_msgs = []
        while not q.empty():
            m = q.get_nowait()
            if m["type"] == "progress":
                progress_msgs.append(m)

        assert len(progress_msgs) > 0
        for m in progress_msgs:
            assert "pct" in m
            assert 0 <= m["pct"] <= 100
            assert "msg" in m


# --- Heatmap with progress ---


class TestHeatmapProgress:
    def test_heatmap_progress(self):
        q: queue.Queue = queue.Queue()

        def sim_fn(**kw):
            import numpy as np

            n = 50
            return {
                "time": list(range(n)),
                "states": {"v": list(np.zeros(n))},
                "spikes": [],
                "spike_count": 0,
                "stats": {"rate_hz": 0, "isi_mean_ms": None, "isi_cv": None, "isi_histogram": None},
                "dt": 0.1,
                "n_steps": n,
                "current_trace": list(np.zeros(n)),
            }

        _heatmap_with_progress(
            sim_fn,
            {"params": {"a": 1.0}, "current": 10.0},
            "a",
            [0.5, 1.0, 1.5],
            "a",
            [0.5, 1.0, 1.5],
            q,
        )

        messages = []
        while not q.empty():
            messages.append(q.get_nowait())

        types = [m["type"] for m in messages]
        assert "progress" in types
        assert types[-1] == "complete"

        complete = messages[-1]
        assert "rates" in complete["result"]
        assert len(complete["result"]["rates"]) == 3


# --- Scan with progress ---


class TestScanProgress:
    def test_scan_produces_results(self):
        q: queue.Queue = queue.Queue()
        _scan_with_progress(q)

        messages = []
        while not q.empty():
            messages.append(q.get_nowait())

        types = [m["type"] for m in messages]
        assert "progress" in types
        assert types[-1] == "complete"

        complete = messages[-1]
        assert isinstance(complete["result"], list)
        assert len(complete["result"]) > 0
        assert "name" in complete["result"][0]


# --- WebSocket endpoint ---


class TestWebSocketEndpoint:
    def test_ws_unknown_op(self, client):
        with client.websocket_connect("/ws/progress") as ws:
            ws.send_json({"op": "nonexistent"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Unknown op" in msg["msg"]

    def test_ws_invalid_json(self, client):
        with client.websocket_connect("/ws/progress") as ws:
            ws.send_text("not json at all")
            msg = ws.receive_json()
            assert msg["type"] == "error"

    def test_ws_characterize_streams_progress(self, client):
        with client.websocket_connect("/ws/progress") as ws:
            ws.send_json(
                {
                    "op": "characterize",
                    "config": {"name": "LIFNeuron", "current": 10.0, "duration": 50.0},
                }
            )
            messages = []
            for _ in range(100):
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] in ("complete", "error"):
                    break

            types = [m["type"] for m in messages]
            assert "complete" in types or "error" in types
            if "complete" in types:
                progress_count = types.count("progress")
                assert progress_count > 0
