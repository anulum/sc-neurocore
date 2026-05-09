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
from typing import Any, Literal

from .bitstream import popcount_slice, MASK32
from .lfsr import Lfsr16


MAX_NEURONS_PER_LAYER = 64
SCMode = Literal["unipolar"]


@dataclass
class SCLayer:
    """Single dense SC layer: weights × inputs via AND + popcount threshold."""

    n_inputs: int
    n_outputs: int
    threshold: int = 512
    weights: list[list[int]] = field(default_factory=list)
    sc_mode: SCMode = "unipolar"

    def __post_init__(self) -> None:
        self._validate_configuration()
        if not self.weights:
            self.weights = [
                [0x5555_5555] * ((self.n_inputs + 31) // 32) for _ in range(self.n_outputs)
            ]
        self._validate_weights()

    def _validate_configuration(self) -> None:
        if self.sc_mode != "unipolar":
            raise ValueError("SCLayer currently supports only sc_mode='unipolar'")
        if self.n_inputs <= 0:
            raise ValueError("n_inputs must be positive")
        if self.n_outputs <= 0:
            raise ValueError("n_outputs must be positive")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if self.n_inputs > MAX_NEURONS_PER_LAYER:
            raise ValueError(f"n_inputs must be <= {MAX_NEURONS_PER_LAYER}")
        if self.n_outputs > MAX_NEURONS_PER_LAYER:
            raise ValueError(f"n_outputs must be <= {MAX_NEURONS_PER_LAYER}")

    def _validate_weights(self) -> None:
        if len(self.weights) != self.n_outputs:
            raise ValueError("weights must contain one row per output")
        words_per_input = self.words_per_input
        for row in self.weights:
            if len(row) != words_per_input:
                raise ValueError("each weight row must match words_per_input")
            if any((word < 0 or word > MASK32) for word in row):
                raise ValueError("weight words must be unsigned 32-bit values")

    @property
    def words_per_input(self) -> int:
        return (self.n_inputs + 31) // 32

    def forward(self, input_words: list[int], bit_length: int) -> list[bool]:
        """Run SC inference: AND each weight row with input, threshold popcount."""
        if bit_length <= 0:
            raise ValueError("bit_length must be positive")
        if len(input_words) < self.words_per_input:
            raise ValueError("input_words length must be at least words_per_input")
        if any((word < 0 or word > MASK32) for word in input_words):
            raise ValueError("input words must be unsigned 32-bit values")
        spikes = []
        for row in self.weights:
            acc = 0
            for w, inp in zip(row, input_words):
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
        if self.sc_mode != "unipolar":
            raise ValueError("SCNetwork currently supports only sc_mode='unipolar'")
        if self.bit_length <= 0:
            raise ValueError("bit_length must be positive")

    def add_layer(self, layer: SCLayer) -> None:
        if layer.sc_mode != self.sc_mode:
            raise ValueError("layer sc_mode must match network sc_mode")
        self.layers.append(layer)

    def encode_inputs(self, probabilities: list[float]) -> list[list[int]]:
        """Encode float probabilities into per-input packed bitstreams."""
        if any((p < 0.0 or p > 1.0) for p in probabilities):
            raise ValueError("input probabilities must be in [0, 1]")
        lfsr = Lfsr16(self.lfsr_seed)
        return [lfsr.encode_float(p, self.bit_length) for p in probabilities]

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
        combined = [0] * wpi
        for stream in streams:
            for j in range(wpi):
                combined[j] = (combined[j] | stream[j]) & MASK32
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
        return len(self.layers)

    @property
    def total_neurons(self) -> int:
        return sum(layer.n_outputs for layer in self.layers)
