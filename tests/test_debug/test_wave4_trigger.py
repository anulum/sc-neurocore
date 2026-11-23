# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrigger from former test_wave4.py

"""Focused suite: TestTrigger from former test_wave4.py."""

from __future__ import annotations

from sc_neurocore.debug.hil_client import (
    SpikeEvent,
    TriggerCondition,
    TriggerLog,
)


class TestTrigger:
    """Correlation triggers and their log."""

    def test_armed_trigger(self) -> None:
        """An armed trigger fires above its level only."""
        tc = TriggerCondition(min_correlation=0.5, armed=True)
        assert tc.evaluate(SpikeEvent(correlation=0.6))
        assert not tc.evaluate(SpikeEvent(correlation=0.3))

    def test_disarmed(self) -> None:
        """A disarmed trigger never fires."""
        tc = TriggerCondition(min_correlation=0.5, armed=False)
        assert not tc.evaluate(SpikeEvent(correlation=0.9))

    def test_trigger_log(self) -> None:
        """The trigger log counts its two entries."""
        tl = TriggerLog()
        tl.fire(SpikeEvent(sequence=1))
        tl.fire(SpikeEvent(sequence=2))
        assert tl.count == 2
