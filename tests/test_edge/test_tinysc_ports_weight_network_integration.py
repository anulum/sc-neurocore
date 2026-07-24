# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWeightNetworkIntegration from former test_tinysc_ports.py

"""Focused suite: TestWeightNetworkIntegration from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403


class TestWeightNetworkIntegration:
    def test_export_import_roundtrip(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2, threshold=1))
        r1 = net.run([0.9, 0.9, 0.9, 0.9])

        blob = serialize_weights(net.export_weights())
        loaded = deserialize_weights(blob)
        net2 = SCNetwork.from_weights(loaded, bit_length=256)
        r2 = net2.run([0.9, 0.9, 0.9, 0.9])
        assert r1 == r2

    def test_export_preserves_structure(self):
        net = SCNetwork(bit_length=512)
        net.add_layer(SCLayer(n_inputs=32, n_outputs=8, threshold=100))
        net.add_layer(SCLayer(n_inputs=8, n_outputs=2, threshold=50))
        exported = net.export_weights()
        assert len(exported) == 2
        assert exported[0][0] == 32  # n_inputs
        assert exported[0][1] == 8  # n_outputs
        assert exported[1][2] == 50  # threshold
