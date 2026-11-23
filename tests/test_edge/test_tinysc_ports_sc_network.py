# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCNetwork from former test_tinysc_ports.py

"""Focused suite: TestSCNetwork from former test_tinysc_ports.py."""

from __future__ import annotations

from typing import Any, cast

import pytest
from sc_neurocore.edge.bitstream import (
    MASK32,
)
from sc_neurocore.edge.sc_network import (
    SCLayer,
    SCNetwork,
)


class TestSCNetwork:
    """Configuration, checks and serialisation of the tiny SC network."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_inputs": 0, "n_outputs": 1},
            {"n_inputs": 1, "n_outputs": 0},
            {"n_inputs": 1, "n_outputs": 1, "threshold": -1},
            {"n_inputs": 65, "n_outputs": 1},
            {"n_inputs": 1, "n_outputs": 65},
            {"n_inputs": 1, "n_outputs": 1, "sc_mode": "bipolar"},
        ],
    )
    def test_layer_invalid_configuration(self, kwargs: dict[str, Any]) -> None:
        """Each invalid layer configuration is refused."""
        with pytest.raises(ValueError):
            SCLayer(**kwargs)

    def test_layer_invalid_weight_shape(self) -> None:
        """Weights without one row per output are refused."""
        with pytest.raises(ValueError, match="one row"):
            SCLayer(n_inputs=1, n_outputs=2, weights=[[0xFFFF_FFFF]])

    def test_layer_rejects_invalid_weight_word(self) -> None:
        """A weight word that is not unsigned is refused."""
        with pytest.raises(ValueError, match="unsigned"):
            SCLayer(n_inputs=1, n_outputs=1, weights=[[MASK32 + 1]])

    def test_layer_rejects_invalid_input_words(self) -> None:
        """Input words of the wrong count, sign or bit length are refused."""
        layer = SCLayer(n_inputs=4, n_outputs=1)
        with pytest.raises(ValueError, match="input_words"):
            layer.forward([], 32)
        with pytest.raises(ValueError, match="unsigned"):
            layer.forward([-1], 32)
        with pytest.raises(ValueError, match="bit_length"):
            layer.forward([0], 0)

    def test_network_invalid_configuration(self) -> None:
        """A zero bit length or a bipolar mode is refused."""
        with pytest.raises(ValueError, match="bit_length"):
            SCNetwork(bit_length=0)
        with pytest.raises(ValueError, match="unipolar"):
            SCNetwork(sc_mode=cast(Any, "bipolar"))

    def test_network_rejects_invalid_probabilities(self) -> None:
        """A probability above one is refused."""
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=2, n_outputs=1))
        with pytest.raises(ValueError, match="probabilities"):
            net.run([0.5, 1.1])

    def test_network_rejects_input_length_mismatch(self) -> None:
        """Too few inputs are refused."""
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=2, n_outputs=1))
        with pytest.raises(ValueError, match="n_inputs"):
            net.run([0.5])

    def test_add_layer_rejects_sc_mode_mismatch(self) -> None:
        """A layer of another mode is refused."""
        net = SCNetwork(bit_length=256, sc_mode="unipolar")
        layer = SCLayer(n_inputs=1, n_outputs=1, sc_mode="unipolar")
        layer.sc_mode = cast(Any, "bipolar")  # force mismatch branch post-init
        with pytest.raises(ValueError, match="sc_mode"):
            net.add_layer(layer)

    def test_empty_network(self) -> None:
        """An empty network maps no input to no output."""
        net = SCNetwork(bit_length=256)
        assert net.run([]) == []

    def test_single_layer(self) -> None:
        """One layer gives its output size."""
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2, threshold=1))
        result = net.run([0.9, 0.9, 0.9, 0.9])
        assert len(result) == 2

    def test_layer_count(self) -> None:
        """The network counts its layers and neurons."""
        net = SCNetwork()
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2))
        net.add_layer(SCLayer(n_inputs=2, n_outputs=1))
        assert net.layer_count == 2
        assert net.total_neurons == 3

    def test_from_weights_roundtrip(self) -> None:
        """A network rebuilt from its weights keeps its layers and seed."""

        class LayerHeader:
            def __init__(self, n_inputs: int, n_outputs: int, threshold: int) -> None:
                self.n_inputs = n_inputs
                self.n_outputs = n_outputs
                self.threshold = threshold

        source = SCNetwork(bit_length=256)
        source.add_layer(SCLayer(n_inputs=2, n_outputs=1, threshold=2, weights=[[0xFFFF_FFFF]]))
        exported = source.export_weights()
        layers_data = [
            (LayerHeader(n_inputs, n_outputs, threshold), rows)
            for n_inputs, n_outputs, threshold, rows in exported
        ]
        restored = SCNetwork.from_weights(layers_data, bit_length=256, lfsr_seed=0x1234)
        assert restored.layer_count == 1
        assert restored.layers[0].weights == [[0xFFFF_FFFF]]
        assert restored.lfsr_seed == 0x1234
