# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio progress heatmap progress

"""Focused suite: TestHeatmapProgress from former test_studio_progress.py."""

from __future__ import annotations

from tests.studio_progress_support import *  # noqa: F403

class TestHeatmapProgress:
    def test_heatmap_progress(self) -> None:
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

