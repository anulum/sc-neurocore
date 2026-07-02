# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC Network Runner (ported from tinysc_riscv/network.rs)

"""Fixed-capacity feed-forward SC network runner.

Mirrors the bare-metal Rust implementation, providing a stack-like
execution model: encode inputs → layer-by-layer SC inference → decode outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Any, Literal

from .bitstream import popcount_slice, MASK32
from .lfsr import Lfsr16


MAX_NEURONS_PER_LAYER = 64
SCMode = Literal["unipolar"]


def _require_integer(value: object, name: str) -> int:
    """Return ``value`` as ``int`` after rejecting bool and non-integral aliases."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _require_positive_integer(
    value: object,
    name: str,
    *,
    max_value: int | None = None,
) -> int:
    """Return a positive integer bounded by ``max_value`` when provided."""
    integer = _require_integer(value, name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    if max_value is not None and integer > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return integer


def _require_nonnegative_integer(value: object, name: str) -> int:
    """Return a non-negative integer suitable for popcount thresholds."""
    integer = _require_integer(value, name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _require_u32_word(value: object, name: str) -> int:
    """Return an unsigned 32-bit word after rejecting lossy numeric aliases."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be unsigned 32-bit values")
    word = int(value)
    if word < 0 or word > MASK32:
        raise ValueError(f"{name} must be unsigned 32-bit values")
    return word


def _require_probability(value: object) -> float:
    """Return a finite unipolar probability in the inclusive ``[0, 1]`` range."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("input probabilities must be finite real values in [0, 1]")
    probability = float(value)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("input probabilities must be finite real values in [0, 1]")
    return probability


@dataclass
class SCLayer:
    """Single dense SC layer: weights × inputs via AND + popcount threshold."""

    n_inputs: int
    n_outputs: int
    threshold: int = 512
    weights: list[list[int]] = field(default_factory=list)
    sc_mode: SCMode = "unipolar"

    def __post_init__(self) -> None:
        """Validate and normalise layer configuration after dataclass construction."""
        self._validate_configuration()
        if not self.weights:
            self.weights = [
                [0x5555_5555] * ((self.n_inputs + 31) // 32) for _ in range(self.n_outputs)
            ]
        self._validate_weights()

    def _validate_configuration(self) -> None:
        if self.sc_mode != "unipolar":
            raise ValueError("SCLayer currently supports only sc_mode='unipolar'")
        self.n_inputs = _require_positive_integer(
            self.n_inputs,
            "n_inputs",
            max_value=MAX_NEURONS_PER_LAYER,
        )
        self.n_outputs = _require_positive_integer(
            self.n_outputs,
            "n_outputs",
            max_value=MAX_NEURONS_PER_LAYER,
        )
        self.threshold = _require_nonnegative_integer(self.threshold, "threshold")

    def _validate_weights(self) -> None:
        if len(self.weights) != self.n_outputs:
            raise ValueError("weights must contain one row per output")
        words_per_input = self.words_per_input
        validated_weights: list[list[int]] = []
        for row in self.weights:
            if len(row) != words_per_input:
                raise ValueError("each weight row must match words_per_input")
            validated_weights.append([_require_u32_word(word, "weight words") for word in row])
        self.weights = validated_weights

    @property
    def words_per_input(self) -> int:
        """Return packed weight words required to cover all layer inputs."""
        return (self.n_inputs + 31) // 32

    def forward(self, input_words: list[int], bit_length: int) -> list[bool]:
        """Run SC inference: AND each weight row with input, threshold popcount."""
        bit_length = _require_positive_integer(bit_length, "bit_length")
        if len(input_words) < self.words_per_input:
            raise ValueError("input_words length must be at least words_per_input")
        validated_inputs = [_require_u32_word(word, "input words") for word in input_words]
        spikes = []
        for row in self.weights:
            acc = 0
            for w, inp in zip(row, validated_inputs):
                acc += popcount_slice([w & inp])
            spikes.append(acc >= self.threshold)
        return spikes


@dataclass
class SCNetwork:
    """Multi-layer feed-forward SC network runner.

    Usage::

        net = SCNetwork(bit_length=1024)
        net.add_layer(SCLayer(n_inputs=32, n_outputs=16))
        net.add_layer(SCLayer(n_inputs=16, n_outputs=8))
        output = net.run([0.5] * 32)
    """

    bit_length: int = 1024
    layers: list[SCLayer] = field(default_factory=list)
    lfsr_seed: int = 0xACE1
    sc_mode: SCMode = "unipolar"

    def __post_init__(self) -> None:
        """Validate network-level stochastic-computing execution parameters."""
        if self.sc_mode != "unipolar":
            raise ValueError("SCNetwork currently supports only sc_mode='unipolar'")
        self.bit_length = _require_positive_integer(self.bit_length, "bit_length")
        self.lfsr_seed = _require_nonnegative_integer(self.lfsr_seed, "lfsr_seed")

    def add_layer(self, layer: SCLayer) -> None:
        """Append a layer after validating stochastic-mode compatibility."""
        if layer.sc_mode != self.sc_mode:
            raise ValueError("layer sc_mode must match network sc_mode")
        self.layers.append(layer)

    def encode_inputs(self, probabilities: list[float]) -> list[list[int]]:
        """Encode float probabilities into per-input packed bitstreams."""
        validated = [_require_probability(p) for p in probabilities]
        lfsr = Lfsr16(self.lfsr_seed)
        return [lfsr.encode_float(p, self.bit_length) for p in validated]

    def _spikes_to_bitstreams(self, spikes: list[bool], lfsr: Lfsr16) -> list[list[int]]:
        """Re-encode spike booleans as bitstreams for the next layer."""
        return [lfsr.encode_float(1.0 if s else 0.0, self.bit_length) for s in spikes]

    def _flatten_bitstreams(self, streams: list[list[int]]) -> list[int]:
        """Interleave per-input bitstreams into a flat word array.

        For a layer expecting N inputs, the flat array has N×words_per_input
        entries: [input0_word0, input1_word0, ..., inputN_word0, input0_word1, ...]
        However for simplicity we concatenate: [input0_words..., input1_words..., ...].
        The layer forward reads the first words_per_input words as the combined input.
        To combine inputs, we OR them together (SC saturating addition).
        """
        if not streams:
            return []
        wpi = len(streams[0])
        if wpi == 0:
            raise ValueError("encoded bitstreams must not be empty")
        combined = [0] * wpi
        for stream in streams:
            if len(stream) != wpi:
                raise ValueError("encoded bitstreams must have the same word width")
            for j, word in enumerate(stream):
                combined[j] = (
                    combined[j] | _require_u32_word(word, "encoded bitstream words")
                ) & MASK32
        return combined

    def run(self, input_probabilities: list[float]) -> list[bool]:
        """Full inference: encode → cascaded layer inference → spike output.

        Each layer's spike output is re-encoded as bitstreams and fed
        to the next layer. This is the correct SC cascade semantics.
        """
        if not self.layers:
            return []
        if len(input_probabilities) != self.layers[0].n_inputs:
            raise ValueError("input_probabilities length must match first layer n_inputs")

        lfsr = Lfsr16(self.lfsr_seed)
        input_streams = self.encode_inputs(input_probabilities)
        current_words = self._flatten_bitstreams(input_streams)

        current_spikes: list[bool] = []
        for layer in self.layers:
            current_spikes = layer.forward(current_words, self.bit_length)
            current_words = self._flatten_bitstreams(
                self._spikes_to_bitstreams(current_spikes, lfsr)
            )

        return current_spikes

    def export_weights(self) -> list[tuple[int, int, int, list[list[int]]]]:
        """Export all layer weights in serialization-ready format."""
        return [
            (layer.n_inputs, layer.n_outputs, layer.threshold, layer.weights)
            for layer in self.layers
        ]

    @classmethod
    def from_weights(
        cls,
        layers_data: list[tuple[Any, list[list[int]]]],
        bit_length: int = 1024,
        lfsr_seed: int = 0xACE1,
    ) -> SCNetwork:
        """Construct network from deserialized weight data."""
        net = cls(bit_length=bit_length, lfsr_seed=lfsr_seed)
        for lh, rows in layers_data:
            net.add_layer(
                SCLayer(
                    n_inputs=lh.n_inputs,
                    n_outputs=lh.n_outputs,
                    threshold=lh.threshold,
                    weights=rows,
                )
            )
        return net

    @property
    def layer_count(self) -> int:
        """Return the number of layers currently registered in the network."""
        return len(self.layers)

    @property
    def total_neurons(self) -> int:
        """Return the total output-neuron count across all registered layers."""
        return sum(layer.n_outputs for layer in self.layers)
