# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BCI Primitives Tests

import unittest

import numpy as np

from sc_neurocore.bci_studio.bci_primitives import BCIClosedLoopEngine


class TestBCIClosedLoopEngine(unittest.TestCase):
    def setUp(self):
        self.engine = BCIClosedLoopEngine(channels=64)

    def test_process_returns_dict(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.5)
        self.assertIsInstance(result, dict)

    def test_result_has_required_keys(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.5)
        self.assertIn("command", result)
        self.assertIn("latency_ms", result)
        self.assertIn("spikes", result)

    def test_command_is_binary(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.5)
        self.assertIn(result["command"], (0, 1))

    def test_latency_positive(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.5)
        self.assertGreater(result["latency_ms"], 0.0)

    def test_spikes_count_non_negative(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.5)
        self.assertGreaterEqual(result["spikes"], 0)

    def test_zero_input_produces_no_spikes(self):
        raw = np.zeros(64, dtype=np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.0)
        self.assertEqual(result["spikes"], 0)
        self.assertEqual(result["command"], 0)

    def test_channels_attribute(self):
        self.assertEqual(self.engine.channels, 64)

    def test_weights_shape(self):
        self.assertEqual(self.engine.weights.shape, (64,))


if __name__ == "__main__":
    unittest.main()
