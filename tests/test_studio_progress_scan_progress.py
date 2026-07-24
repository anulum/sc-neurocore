# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio progress scan progress

"""Focused suite: TestScanProgress from former test_studio_progress.py."""

from __future__ import annotations

from tests.studio_progress_support import *  # noqa: F403


class TestScanProgress:
    def test_scan_produces_results(self) -> None:
        q: queue.Queue[dict[str, Any]] = queue.Queue()
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
