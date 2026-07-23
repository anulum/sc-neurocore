# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBCIStudio from former test_bci_studio.py

"""Focused suite: TestBCIStudio from former test_bci_studio.py."""

from __future__ import annotations

from bci_studio_support import *  # noqa: F403

class TestBCIStudio(unittest.TestCase):
    def setUp(self):
        self.studio = BCIStudio(channels=64, lr=0.01)

    def test_process_frame(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.studio.process_frame(raw, reward=0.5)
        self.assertIn("command", result)
        self.assertIn("latency_ms", result)
        self.assertIn("spikes", result)
        self.assertIn("compression_ratio", result)
        self.assertIn("feedback_bytes", result)

    def test_latency_under_budget(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.studio.process_frame(raw, reward=0.0)
        self.assertLess(result["latency_ms"], 10.0)

    def test_large_weight_shift_records_adaptation_event(self):
        # A fully-spiking frame with a strong reward moves every weight, so the
        # aggregate shift clears the adaptation threshold (0.01 * channels).
        raw = np.tile([1.0, -1.0], 32).astype(np.float32)  # all 64 channels spike
        before = self.studio.metrics.adaptation_events
        self.studio.process_frame(raw, reward=2.0)
        self.assertEqual(self.studio.metrics.adaptation_events, before + 1)

    def test_session_lifecycle(self):
        self.studio.start_session()
        for _ in range(10):
            raw = np.random.randn(64).astype(np.float32)
            self.studio.process_frame(raw, reward=0.5)
        metrics = self.studio.stop_session()
        self.assertEqual(metrics.total_frames, 10)
        self.assertGreater(metrics.total_spikes, 0)

    def test_session_summary(self):
        self.studio.start_session()
        raw = np.random.randn(64).astype(np.float32)
        self.studio.process_frame(raw)
        metrics = self.studio.stop_session()
        summary = metrics.summary()
        self.assertIn("Frames:", summary)
        self.assertIn("Latency:", summary)

    def test_feedback_bytes(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.studio.process_frame(raw)
        self.assertEqual(result["feedback_bytes"], 16)

    def test_zero_input_no_stim(self):
        raw = np.zeros(64, dtype=np.float32)
        result = self.studio.process_frame(raw, reward=0.0)
        self.assertEqual(result["command"], FPGAFeedbackController.COMMAND_NOP)

    def test_profiler_integration(self):
        for _ in range(20):
            raw = np.random.randn(64).astype(np.float32)
            self.studio.process_frame(raw)
        self.assertGreater(self.studio.profiler.mean, 0)
        self.assertTrue(self.studio.profiler.budget_met)
