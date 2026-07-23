# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHilClientBranchCoverage from former test_wave4.py

"""Focused suite: TestHilClientBranchCoverage from former test_wave4.py."""

from __future__ import annotations

from wave4_support import *  # noqa: F403

class TestHilClientBranchCoverage:
    """Edge and accessor branches across the HIL telemetry components."""

    def test_ring_buffer_head_and_capacity_accessors(self):
        rb = SpikeRingBuffer(capacity=4)
        assert rb.capacity == 4
        assert rb.head == 0
        rb.push(SpikeEvent(timestamp=1))
        rb.push(SpikeEvent(timestamp=2))
        assert rb.head == 2

    def test_layer_aggregator_all_returns_independent_copies(self):
        agg = LayerAggregator()
        agg.record(SpikeEvent(layer_id="L1", correlation=0.3, precision=0.9))
        snapshot = agg.all()
        assert set(snapshot) == {"L1"}
        # Mutating the snapshot must not bleed back into the aggregator state.
        snapshot["L1"]["event_count"] = 999
        assert agg.all()["L1"]["event_count"] == 1

    def test_layer_aggregator_means_handle_zero_event_count(self):
        empty = {"event_count": 0, "sum_correlation": 0.0, "sum_precision": 0.0}
        assert LayerAggregator.mean_correlation(empty) == 0.0
        assert LayerAggregator.mean_precision(empty) == 0.0

    def test_layer_aggregator_mean_precision_divides_when_populated(self):
        ls = {"event_count": 2, "sum_correlation": 1.0, "sum_precision": 1.4}
        assert LayerAggregator.mean_precision(ls) == pytest.approx(0.7)

    def test_correlation_window_mean_and_max_empty_return_zero(self):
        win = CorrelationWindow(size=8)
        assert win.mean() == 0.0
        assert win.max() == 0.0

    def test_trigger_condition_layer_mismatch_does_not_fire(self):
        trig = TriggerCondition(min_correlation=0.5, layer_id="L1")
        # Event belongs to a different layer → trigger must not fire.
        assert trig.evaluate(SpikeEvent(layer_id="L2", correlation=0.9)) is False

    def test_rate_limiter_available_reflects_remaining_tokens(self):
        rl = RateLimiter(capacity=2)
        assert rl.available == 2
        rl.allow()
        assert rl.available == 1
