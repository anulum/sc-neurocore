# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sc_neurocore.synapses.stochastic_stdp import StochasticSTDPSynapse
from sc_neurocore.utils.fsm_activations import TanhFSM
from sc_neurocore.utils.decorrelators import ShufflingDecorrelator

class TestAdvancements(unittest.TestCase):
    def test_stdp_potentiation(self):
        # Create synapse with low weight
        synapse = StochasticSTDPSynapse(w_min=0.0, w_max=1.0, w=0.1, learning_rate=0.2, seed=42)
        initial_w = synapse.w

        # Force correlated firing (LTP)
        # PRE = 1, POST = 1
        for _ in range(100):
            synapse.process_step(pre_bit=1, post_bit=1)

        print(f"STDP Potentiation: {initial_w} -> {synapse.w}")
        self.assertGreater(synapse.w, initial_w)

    def test_stdp_depression(self):
        # Create synapse with high weight
        synapse = StochasticSTDPSynapse(w_min=0.0, w_max=1.0, w=0.9, learning_rate=0.2, seed=42)
        initial_w = synapse.w

        # Force anti-correlated firing (LTD)
        # PRE = 1, POST = 0
        for _ in range(100):
            synapse.process_step(pre_bit=1, post_bit=0)

        print(f"STDP Depression: {initial_w} -> {synapse.w}")
        self.assertLess(synapse.w, initial_w)

    def test_tanh_fsm(self):
        fsm = TanhFSM(states=16)

        # Test Saturation High
        ones = np.ones(20, dtype=np.uint8)
        out_high = fsm.process(ones)
        self.assertEqual(out_high[-1], 1)
        self.assertEqual(fsm.state, 15)

        # Test Saturation Low
        zeros = np.zeros(40, dtype=np.uint8)
        out_low = fsm.process(zeros)
        self.assertEqual(out_low[-1], 0)
        self.assertEqual(fsm.state, 0)

    def test_decorrelator(self):
        decorr = ShufflingDecorrelator(window_size=10, seed=42)
        bits = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)

        out = decorr.process(bits)

        # Mean should be identical
        self.assertEqual(bits.mean(), out.mean())

        # But order should change (statistically likely)
        # With seed 42, we expect some change.
        self.assertFalse(np.array_equal(bits, out))

if __name__ == '__main__':
    unittest.main()
