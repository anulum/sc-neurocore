# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synapse precision specification

"""Per-synapse precision and error-bound specification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SynapsePrecision:
    """Precision assignment and conservative error bound for one synapse."""

    layer_index: int
    layer_name: str
    output_index: int
    input_index: int
    bit_width: int
    bitstream_length: int
    sensitivity: float
    quantization_error_bound: float
    stochastic_error_bound: float
    total_error_bound: float

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a JSON-serialisable precision-plan row."""
        return {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "output_index": self.output_index,
            "input_index": self.input_index,
            "bit_width": self.bit_width,
            "bitstream_length": self.bitstream_length,
            "sensitivity": self.sensitivity,
            "quantization_error_bound": self.quantization_error_bound,
            "stochastic_error_bound": self.stochastic_error_bound,
            "total_error_bound": self.total_error_bound,
        }
