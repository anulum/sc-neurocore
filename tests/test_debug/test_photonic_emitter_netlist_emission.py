# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNetlistEmission from former test_photonic_emitter.py

"""Focused suite: TestNetlistEmission from former test_photonic_emitter.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from photonic_emitter_support import *  # noqa: F403

class TestNetlistEmission(unittest.TestCase):
    def test_basic_netlist(self):
        emitter = PhotonicEmitter()
        nodes = [
            MockNode("SC_AND", "m1", ["laser_a", "laser_b"], "bus_1"),
            MockNode("LIF_MEMBRANE", "n1", ["bus_1"], "spike_out"),
        ]
        netlist = emitter.emit_lumerical_netlist(MockGraph(nodes))
        self.assertIn("MZI_MODULATOR", netlist)
        self.assertIn("RESONANT_CAVITY", netlist)
        self.assertIn("m1", netlist)
        self.assertIn("n1", netlist)

    def test_pdk_in_header(self):
        emitter = PhotonicEmitter(target_pdk="imec_si_ph_v2")
        nodes = [MockNode("SC_AND", "m1", ["a", "b"], "c")]
        netlist = emitter.emit_lumerical_netlist(MockGraph(nodes))
        self.assertIn("imec_si_ph_v2", netlist)

    def test_and_gate_connections(self):
        emitter = PhotonicEmitter()
        nodes = [MockNode("SC_AND", "m1", ["input_a", "input_b"], "out")]
        netlist = emitter.emit_lumerical_netlist(MockGraph(nodes))
        self.assertIn("CONNECT m1:in1 input_a", netlist)
        self.assertIn("CONNECT m1:in2 input_b", netlist)

    def test_lif_threshold(self):
        emitter = PhotonicEmitter()
        nodes = [MockNode("LIF_MEMBRANE", "n1", ["in"], "out")]
        netlist = emitter.emit_lumerical_netlist(MockGraph(nodes))
        self.assertIn("Q_factor 15000", netlist)

    def test_empty_graph(self):
        emitter = PhotonicEmitter()
        netlist = emitter.emit_lumerical_netlist(MockGraph([]))
        self.assertIn("SC-NeuroCore", netlist)
