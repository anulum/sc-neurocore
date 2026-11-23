# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSleepPhase from former test_meta_plasticity.py

"""Focused suite: TestSleepPhase from former test_meta_plasticity.py."""

from __future__ import annotations

from sc_neurocore.meta_plasticity.meta_plasticity import (
    SleepPhase,
)


class TestSleepPhase:
    """Replay of experiences during sleep."""

    def test_record_and_buffer_size(self) -> None:
        """Recorded experiences fill the buffer."""
        sp = SleepPhase()
        sp.record({"novelty": 0.5})
        sp.record({"novelty": 0.8})
        assert sp.buffer_size == 2

    def test_sleep_replays(self) -> None:
        """Sleep replays the requested experiences and then wakes."""
        sp = SleepPhase(consolidation_rounds=3)
        for i in range(5):
            sp.record({"novelty": float(i) / 4})
        calls = []
        replays = sp.sleep(lambda m: calls.append(m))
        assert replays == 3
        assert len(calls) == 3
        assert not sp.is_sleeping

    def test_empty_buffer_no_replay(self) -> None:
        """An empty buffer replays nothing."""
        sp = SleepPhase()
        replays = sp.sleep(lambda m: None)
        assert replays == 0
