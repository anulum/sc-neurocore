# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEventDrivenInterface from former test_analog_bridge_extended.py

"""Focused suite: TestEventDrivenInterface from former test_analog_bridge_extended.py."""

from __future__ import annotations

from analog_bridge_extended_support import *  # noqa: F403

class TestEventDrivenInterface(unittest.TestCase):
    """AER conversion and rate-coding contract checks."""

    def setUp(self) -> None:
        """Create a one-microsecond event interface."""
        self.iface = EventDrivenInterface(clock_period_us=1.0)

    def test_bitstream_to_events(self) -> None:
        """Set bits become ordered AER events with matching timestamps."""
        bs = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        events = self.iface.bitstream_to_events(42, bs)
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0].neuron_id, 42)
        self.assertAlmostEqual(events[0].timestamp_us, 0.0)
        self.assertAlmostEqual(events[1].timestamp_us, 2.0)

    def test_zero_bitstream(self) -> None:
        """An all-zero bitstream emits no AER events."""
        bs = np.zeros(100, dtype=np.uint8)
        events = self.iface.bitstream_to_events(1, bs)
        self.assertEqual(len(events), 0)

    def test_events_to_current_shape(self) -> None:
        """AER events produce a current trace with the requested duration."""
        events = [AEREvent(0, 5.0), AEREvent(0, 15.0)]
        current = self.iface.events_to_current(events, duration_us=50.0, tau_syn=5.0)
        self.assertEqual(len(current), 50)

    def test_events_produce_positive_current(self) -> None:
        """Excitatory AER events add positive synaptic current."""
        events = [AEREvent(0, 0.0)]
        current = self.iface.events_to_current(events, duration_us=20.0, tau_syn=5.0)
        self.assertGreater(current[0], 0)

    def test_current_decays(self) -> None:
        """The synaptic current kernel decays after an event."""
        events = [AEREvent(0, 0.0)]
        current = self.iface.events_to_current(events, duration_us=50.0, tau_syn=5.0)
        self.assertGreater(current[0], current[-1])

    def test_rate_code(self) -> None:
        """Rate coding reports event frequency in hertz."""
        events = [AEREvent(0, i * 10.0) for i in range(100)]
        rate = self.iface.rate_code(events, window_us=1000.0)
        self.assertAlmostEqual(rate, 100_000.0, delta=1.0)  # 100 events in 1ms = 100kHz

    def test_rate_code_empty(self) -> None:
        """Empty event lists report zero firing rate."""
        self.assertEqual(self.iface.rate_code([], window_us=100.0), 0.0)
