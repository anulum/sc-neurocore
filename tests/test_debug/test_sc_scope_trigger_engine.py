# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTriggerEngine from former test_sc_scope.py

"""Focused suite: TestTriggerEngine from former test_sc_scope.py."""

from __future__ import annotations

from sc_scope_support import *  # noqa: F403

class TestTriggerEngine:
    def test_density_above(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.DENSITY_ABOVE, threshold=0.9, layer_id=0))
        # High density sample
        words = np.array([0xFFFF_FFFF] * 8, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        events = te.evaluate(s)
        assert len(events) == 1
        assert events[0].trigger_type == TriggerType.DENSITY_ABOVE

    def test_density_below(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.DENSITY_BELOW, threshold=0.1, layer_id=0))
        words = np.array([0] * 8, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        events = te.evaluate(s)
        assert len(events) == 1

    def test_no_trigger_when_disabled(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.DENSITY_ABOVE, enabled=False))
        words = np.array([0xFFFF_FFFF] * 8, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert len(te.evaluate(s)) == 0

    def test_wrong_layer_skipped(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.DENSITY_ABOVE, threshold=0.5, layer_id=1))
        words = np.array([0xFFFF_FFFF] * 8, dtype=np.uint32)
        s = BitstreamSample(0, layer_id=0, neuron_id=0, words=words)
        assert len(te.evaluate(s)) == 0

    def test_event_count(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.SPIKE_DETECTED, layer_id=0))
        rng = np.random.default_rng(42)
        for i in range(5):
            words = rng.integers(1, 0xFFFF_FFFF, size=4, dtype=np.uint32)
            s = BitstreamSample(i * 100, 0, 0, words)
            te.evaluate(s)
        assert te.event_count > 0

    def test_clear(self):
        te = TriggerEngine()
        te.add_trigger(TriggerCondition(TriggerType.SPIKE_DETECTED, layer_id=0))
        words = np.array([0xFFFF_FFFF] * 4, dtype=np.uint32)
        te.evaluate(BitstreamSample(0, 0, 0, words))
        te.clear()
        assert te.event_count == 0
