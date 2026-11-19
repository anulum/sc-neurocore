# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Edge SCNetwork contract tests

"""Contract tests for the edge stochastic-computing network runner."""

from __future__ import annotations

from typing import cast

import pytest

from sc_neurocore.edge.bitstream import MASK32
from sc_neurocore.edge.sc_network import SCLayer, SCNetwork


class _MismatchedEncodedNetwork(SCNetwork):
    """Network that simulates a corrupted encoder return shape."""

    def encode_inputs(self, probabilities: list[float]) -> list[list[int]]:
        """Return streams with incompatible packed-word widths."""
        return [[0x1, 0x2], [0x4]]


class _InvalidWordEncodedNetwork(SCNetwork):
    """Network that simulates a corrupted encoder word value."""

    def encode_inputs(self, probabilities: list[float]) -> list[list[int]]:
        """Return a stream containing a word outside the u32 domain."""
        return [[MASK32 + 1], [0x0]]


class _NoEncodedStreamsNetwork(SCNetwork):
    """Network that simulates an encoder dropping all input streams."""

    def encode_inputs(self, probabilities: list[float]) -> list[list[int]]:
        """Return no encoded streams despite a non-empty input vector."""
        return []


class _ZeroWidthEncodedNetwork(SCNetwork):
    """Network that simulates an encoder returning zero-width streams."""

    def encode_inputs(self, probabilities: list[float]) -> list[list[int]]:
        """Return one encoded stream with no packed words."""
        return [[]]


def test_network_rejects_nonintegral_bit_length() -> None:
    """Reject bit lengths that cannot map to a fixed packed-word count."""
    with pytest.raises(ValueError, match="integer"):
        SCNetwork(bit_length=cast(int, 64.5))


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), float("-inf")])
def test_run_rejects_nonfinite_probabilities_before_encoding(probability: float) -> None:
    """Reject non-finite probabilities before the LFSR threshold conversion."""
    net = SCNetwork(bit_length=64)
    net.add_layer(SCLayer(n_inputs=2, n_outputs=1, threshold=1))

    with pytest.raises(ValueError, match="finite"):
        net.run([0.25, probability])


@pytest.mark.parametrize("probability", [cast(float, True), cast(float, "0.5")])
def test_run_rejects_probability_aliases_before_encoding(probability: float) -> None:
    """Reject bool and non-real aliases instead of coercing them to thresholds."""
    net = SCNetwork(bit_length=64)
    net.add_layer(SCLayer(n_inputs=2, n_outputs=1, threshold=1))

    with pytest.raises(ValueError, match="finite real"):
        net.run([0.25, probability])


def test_run_rejects_mismatched_encoded_stream_widths() -> None:
    """Reject corrupted encoder streams before layer execution."""
    net = _MismatchedEncodedNetwork(bit_length=64)
    net.add_layer(SCLayer(n_inputs=2, n_outputs=1, threshold=1))

    with pytest.raises(ValueError, match="same word width"):
        net.run([0.25, 0.50])


def test_run_rejects_missing_encoded_streams_at_layer_boundary() -> None:
    """Surface a clear layer-boundary error when encoding drops all streams."""
    net = _NoEncodedStreamsNetwork(bit_length=64)
    net.add_layer(SCLayer(n_inputs=2, n_outputs=1, threshold=1))

    with pytest.raises(ValueError, match="input_words"):
        net.run([0.25, 0.50])


def test_run_rejects_zero_width_encoded_streams() -> None:
    """Reject encoded streams that carry no packed words."""
    net = _ZeroWidthEncodedNetwork(bit_length=64)
    net.add_layer(SCLayer(n_inputs=1, n_outputs=1, threshold=1))

    with pytest.raises(ValueError, match="must not be empty"):
        net.run([0.25])


def test_run_rejects_encoded_words_outside_u32_domain() -> None:
    """Reject corrupted encoded words before masking can hide them."""
    net = _InvalidWordEncodedNetwork(bit_length=64)
    net.add_layer(SCLayer(n_inputs=2, n_outputs=1, threshold=1))

    with pytest.raises(ValueError, match="unsigned 32-bit"):
        net.run([0.25, 0.50])


def test_layer_rejects_weight_rows_that_do_not_match_input_width() -> None:
    """Reject rows whose packed weight width cannot cover all inputs."""
    with pytest.raises(ValueError, match="words_per_input"):
        SCLayer(n_inputs=33, n_outputs=1, weights=[[0]])


def test_layer_rejects_nonintegral_weight_words() -> None:
    """Reject weight words that would fail later in bitwise layer execution."""
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        SCLayer(n_inputs=1, n_outputs=1, weights=[[cast(int, 1.5)]])


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [([1.0, 0.0], [True, False]), ([0.0, 1.0], [False, True])],
)
def test_run_keeps_input_identity(probabilities: list[float], expected: list[bool]) -> None:
    """Each mask selects its own input across all 33 logical clocks."""
    network = SCNetwork(bit_length=33)
    network.add_layer(SCLayer(n_inputs=2, n_outputs=2, threshold=33, weights=[[1], [2]]))
    assert network.run(probabilities) == expected


def test_cascade_keeps_spike_position_and_logical_length() -> None:
    """The second layer measures only the logical clocks of its selected spike."""
    network = SCNetwork(bit_length=33)
    network.add_layer(SCLayer(n_inputs=2, n_outputs=2, threshold=33, weights=[[1], [2]]))
    network.add_layer(SCLayer(n_inputs=2, n_outputs=2, threshold=33, weights=[[2], [1]]))
    assert network.run([1.0, 0.0]) == [False, True]
    assert network.run([0.0, 1.0]) == [True, False]


def test_network_mask_crosses_packed_input_word_boundary() -> None:
    """A bit in the second mask word selects input 32."""
    network = SCNetwork(bit_length=33)
    network.add_layer(SCLayer(n_inputs=33, n_outputs=1, threshold=33, weights=[[0, 1]]))
    assert network.run([0.0] * 32 + [1.0]) == [True]
    assert network.run([1.0] + [0.0] * 32) == [False]


def test_simultaneous_selected_inputs_contribute_each_logical_one() -> None:
    """Co-occurring ones contribute separately to the output threshold."""
    network = SCNetwork(bit_length=33)
    network.add_layer(SCLayer(n_inputs=2, n_outputs=1, threshold=34, weights=[[3]]))
    assert network.run([1.0, 1.0]) == [True]
    assert network.run([1.0, 0.0]) == [False]


def test_direct_layer_forward_accepts_one_packed_input_vector() -> None:
    """The direct layer API still applies a mask to a single packed vector."""
    layer = SCLayer(n_inputs=33, n_outputs=2, threshold=1, weights=[[1, 0], [0, 1]])
    assert layer.forward([0, 1], 32) == [False, True]
