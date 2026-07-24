# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWeights from former test_tinysc_ports.py

"""Focused suite: TestWeights from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403


class TestWeights:
    def test_header_roundtrip(self):
        h = WeightHeader(n_layers=3)
        data = h.to_bytes()
        h2 = WeightHeader.from_bytes(data)
        assert h2.magic == WEIGHT_MAGIC
        assert h2.n_layers == 3

    def test_validate(self):
        h = WeightHeader()
        assert h.validate()
        h.magic = 0xDEAD
        assert not h.validate()

    def test_serialize_roundtrip(self):
        weights = [
            (4, 2, 512, [[0xAAAA_AAAA], [0x5555_5555]]),
        ]
        blob = serialize_weights(weights)
        layers = deserialize_weights(blob)
        assert len(layers) == 1
        lh, rows = layers[0]
        assert lh.n_inputs == 4
        assert lh.n_outputs == 2
        assert rows[0] == [0xAAAA_AAAA]
        assert rows[1] == [0x5555_5555]

    def test_multi_layer(self):
        weights = [
            (32, 4, 256, [[0xFF] * 1 for _ in range(4)]),
            (4, 2, 128, [[0x0F] * 1 for _ in range(2)]),
        ]
        blob = serialize_weights(weights)
        layers = deserialize_weights(blob)
        assert len(layers) == 2

    def test_invalid_magic_raises(self):
        data = b"\x00" * 16
        with pytest.raises(ValueError, match="Invalid weight blob"):
            deserialize_weights(data)
