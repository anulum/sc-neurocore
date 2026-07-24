# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCascadeSemantics from former test_tinysc_ports.py

"""Focused suite: TestCascadeSemantics from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403


class TestCascadeSemantics:
    def test_two_layer_cascade_output_size(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=4, n_outputs=4, threshold=1))
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2, threshold=1))
        result = net.run([0.9, 0.9, 0.9, 0.9])
        assert len(result) == 2

    def test_cascade_deterministic(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=4, n_outputs=4, threshold=1))
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2, threshold=1))
        r1 = net.run([0.5, 0.5, 0.5, 0.5])
        r2 = net.run([0.5, 0.5, 0.5, 0.5])
        assert r1 == r2

    def test_three_layer_cascade(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=8, n_outputs=4, threshold=1))
        net.add_layer(SCLayer(n_inputs=4, n_outputs=4, threshold=1))
        net.add_layer(SCLayer(n_inputs=4, n_outputs=1, threshold=1))
        result = net.run([0.9] * 8)
        assert len(result) == 1
