# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio progress characterise progress

"""Focused suite: TestCharacteriseProgress from former test_studio_progress.py."""

from __future__ import annotations

from tests.studio_progress_support import *  # noqa: F403


class TestCharacteriseProgress:
    def test_produces_progress_and_complete(self) -> None:
        q: queue.Queue[dict[str, Any]] = queue.Queue()

        def sim_fn(**kw: Any) -> dict[str, Any]:
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

    def test_progress_has_pct(self) -> None:
        q: queue.Queue[dict[str, Any]] = queue.Queue()

        def sim_fn(**kw: Any) -> dict[str, Any]:
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
