# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEmitAnalogConfig from former test_analog_bridge.py

"""Focused suite: TestEmitAnalogConfig from former test_analog_bridge.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from analog_bridge_support import *  # noqa: F403


class TestEmitAnalogConfig(unittest.TestCase):
    """Configuration emission checks for bridge node descriptors."""

    def setUp(self) -> None:
        """Create a bridge with a four-bit DAC range."""
        self.bridge = AnalogBridge(g_range=(0.0, 100.0), v_range=(-80.0, -40.0), dac_res=4)

    def test_weight_node_produces_synapse(self) -> None:
        """SC weight descriptors produce synapse DAC entries."""
        nodes = [MockNode("SC_WEIGHT", "s1", probability=0.5)]
        config = self.bridge.emit_analog_config(nodes)
        self.assertIn("s1", config["synapses"])
        self.assertIn("dac", config["synapses"]["s1"])
        self.assertIn("g_ns", config["synapses"]["s1"])

    def test_neuron_node_produces_neuron(self) -> None:
        """LIF membrane descriptors produce neuron DAC entries."""
        nodes = [MockNode("LIF_MEMBRANE", "n1", threshold=0.25)]
        config = self.bridge.emit_analog_config(nodes)
        self.assertIn("n1", config["neurons"])
        self.assertIn("dac", config["neurons"]["n1"])
        self.assertIn("v_mv", config["neurons"]["n1"])

    def test_quantization_error_tracked(self) -> None:
        """Synapse entries include quantization error metadata."""
        nodes = [MockNode("SC_WEIGHT", "s1", probability=0.33)]
        config = self.bridge.emit_analog_config(nodes)
        self.assertIn("s1", config["errors"])
        self.assertIsInstance(config["errors"]["s1"], float)

    def test_empty_nodes(self) -> None:
        """An empty node collection emits empty synapse and neuron maps."""
        config = self.bridge.emit_analog_config([])
        self.assertEqual(len(config["synapses"]), 0)
        self.assertEqual(len(config["neurons"]), 0)

    def test_probability_zero_maps_to_g_min(self) -> None:
        """A zero stochastic weight maps to minimum conductance."""
        bridge = AnalogBridge(g_range=(0.0, 100.0), v_range=(-80.0, -40.0), dac_res=10)
        nodes = [MockNode("SC_WEIGHT", "s0", probability=0.0)]
        config = bridge.emit_analog_config(nodes)
        self.assertAlmostEqual(config["synapses"]["s0"]["g_ns"], 0.0, places=1)

    def test_probability_one_maps_to_g_max(self) -> None:
        """A unit stochastic weight maps to maximum conductance."""
        bridge = AnalogBridge(g_range=(0.0, 100.0), v_range=(-80.0, -40.0), dac_res=10)
        nodes = [MockNode("SC_WEIGHT", "s1", probability=1.0)]
        config = bridge.emit_analog_config(nodes)
        self.assertAlmostEqual(config["synapses"]["s1"]["g_ns"], 100.0, places=1)
