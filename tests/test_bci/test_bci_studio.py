# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BCI Studio Tests

from __future__ import annotations

import unittest

import numpy as np

from sc_neurocore.bci_studio.bci_studio import (
    SpikeCodec,
    OnlineLearner,
    FPGAFeedbackController,
    LatencyProfiler,
    BCIStudio,
)


class TestSpikeCodec(unittest.TestCase):
    def setUp(self):
        self.codec = SpikeCodec()

    def test_encode_decode_roundtrip(self):
        spikes = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8)
        encoded = self.codec.encode(spikes)
        decoded = self.codec.decode(encoded)
        np.testing.assert_array_equal(spikes, decoded)

    def test_empty_array(self):
        spikes = np.array([], dtype=np.uint8)
        encoded = self.codec.encode(spikes)
        self.assertEqual(encoded, b"")

    def test_all_zeros(self):
        spikes = np.zeros(100, dtype=np.uint8)
        encoded = self.codec.encode(spikes)
        decoded = self.codec.decode(encoded)
        np.testing.assert_array_equal(spikes, decoded)

    def test_all_ones(self):
        spikes = np.ones(100, dtype=np.uint8)
        encoded = self.codec.encode(spikes)
        decoded = self.codec.decode(encoded)
        np.testing.assert_array_equal(spikes, decoded)

    def test_compression_ratio_sparse(self):
        spikes = np.zeros(1000, dtype=np.uint8)
        spikes[::100] = 1  # very sparse
        ratio = self.codec.compression_ratio(spikes)
        self.assertGreater(ratio, 1.0)

    def test_decode_returns_empty_for_truncated_header(self):
        # An RLE stream shorter than the 4-byte length header carries no spikes.
        self.assertEqual(self.codec.decode(b"\x00\x00").size, 0)

    def test_compression_ratio_of_empty_array_is_unity(self):
        # An empty spike array compresses to nothing, so the ratio is defined as 1.0.
        self.assertEqual(self.codec.compression_ratio(np.array([], dtype=np.uint8)), 1.0)

    def test_compression_ratio_dense(self):
        rng = np.random.default_rng(42)
        spikes = rng.integers(0, 2, size=1000, dtype=np.uint8)
        ratio = self.codec.compression_ratio(spikes)
        self.assertGreater(ratio, 0)


class TestOnlineLearner(unittest.TestCase):
    def setUp(self):
        self.learner = OnlineLearner(num_weights=64, lr=0.1)

    def test_initial_weights(self):
        np.testing.assert_array_equal(self.learner.weights, np.ones(64, dtype=np.float32))

    def test_positive_reward_potentiates(self):
        spikes = np.zeros(64, dtype=np.uint8)
        spikes[:10] = 1
        old_w = self.learner.weights[0]
        self.learner.step(spikes, reward=1.0)
        # Spiking channels should be potentiated (after decay)
        self.assertGreater(self.learner.weights[0], old_w * 0.9)

    def test_negative_reward_depresses(self):
        spikes = np.zeros(64, dtype=np.uint8)
        spikes[:10] = 1
        self.learner.step(spikes, reward=-1.0)
        # Spiking channels with negative reward get depressed
        self.assertLess(self.learner.weights[0], 1.0)

    def test_weights_clipped(self):
        for _ in range(100):
            spikes = np.ones(64, dtype=np.uint8)
            self.learner.step(spikes, reward=1.0)
        self.assertTrue(np.all(self.learner.weights <= 10.0))
        self.assertTrue(np.all(self.learner.weights >= 0.01))

    def test_update_counter(self):
        spikes = np.zeros(64, dtype=np.uint8)
        self.learner.step(spikes, reward=0.0)
        self.assertEqual(self.learner.updates, 1)


class TestFPGAFeedbackController(unittest.TestCase):
    def setUp(self):
        self.ctrl = FPGAFeedbackController()

    def test_serialize_deserialize(self):
        packet = self.ctrl.serialize(command=1, channel=42, amplitude=0.75, timestamp_us=12345.0)
        result = self.ctrl.deserialize(packet)
        self.assertEqual(result["command"], 1)
        self.assertEqual(result["channel"], 42)
        self.assertAlmostEqual(result["amplitude"], 0.75, places=4)
        self.assertAlmostEqual(result["timestamp_us"], 12345.0, places=1)

    def test_packet_size(self):
        packet = self.ctrl.serialize(command=0)
        self.assertEqual(len(packet), 16)  # DMA-aligned


class TestLatencyProfiler(unittest.TestCase):
    def test_empty_profiler(self):
        p = LatencyProfiler()
        self.assertEqual(p.mean, 0.0)

    def test_budget_met(self):
        p = LatencyProfiler()
        for _ in range(100):
            p.record(0.5)
        self.assertTrue(p.budget_met)

    def test_budget_exceeded(self):
        p = LatencyProfiler()
        for _ in range(100):
            p.record(15.0)
        self.assertFalse(p.budget_met)

    def test_percentiles(self):
        p = LatencyProfiler()
        for i in range(100):
            p.record(float(i))
        self.assertGreater(p.p95, p.p50)
        self.assertGreater(p.p99, p.p95)


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


if __name__ == "__main__":
    unittest.main()
