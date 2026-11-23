# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWeights from former test_tinysc_ports.py

"""Focused suite: TestWeights from former test_tinysc_ports.py."""

from __future__ import annotations

import pytest
from sc_neurocore.edge.weights import (
    deserialize_weights,
    serialize_weights,
    WEIGHT_MAGIC,
    WeightHeader,
)


class TestWeights:
    """The weight blob format of the tiny SC port."""

    def test_header_roundtrip(self) -> None:
        """A header reads back its magic and layer count."""
        h = WeightHeader(n_layers=3)
        data = h.to_bytes()
        h2 = WeightHeader.from_bytes(data)
        assert h2.magic == WEIGHT_MAGIC
        assert h2.n_layers == 3

    def test_validate(self) -> None:
        """A header validates only with the right magic."""
        h = WeightHeader()
        assert h.validate()
        h.magic = 0xDEAD
        assert not h.validate()

    def test_serialize_roundtrip(self) -> None:
        """A serialised layer reads back its sizes."""
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

    def test_multi_layer(self) -> None:
        """Two serialised layers read back as two."""
        weights = [
            (32, 4, 256, [[0xFF] * 1 for _ in range(4)]),
            (4, 2, 128, [[0x0F] * 1 for _ in range(2)]),
        ]
        blob = serialize_weights(weights)
        layers = deserialize_weights(blob)
        assert len(layers) == 2

    def test_invalid_magic_raises(self) -> None:
        """A blob with the wrong magic is refused."""
        data = b"\x00" * 16
        with pytest.raises(ValueError, match="Invalid weight blob"):
            deserialize_weights(data)
