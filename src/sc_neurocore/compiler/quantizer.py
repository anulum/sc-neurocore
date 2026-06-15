# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weight quantization facade

"""Quantize trained float weights to fixed-point / block-floating precision."""

from __future__ import annotations

from .block_floating import (
    BlockExponentLayout,
    BlockFloatingMode,
)
from .block_floating_quantization import (
    CompiledBlockFloatingDense,
    compile_dense_block_floating,
    dequantize_block_floating,
    quantize_block_floating,
)
from .fixed_point_quantization import (
    dequantize,
    dequantize_weights,
    q_weights_to_sc_probabilities,
    quantization_error,
    quantize_weights,
)
from .mixed_dense_quantization import (
    CompiledMixedDense,
    compile_dense_mixed_precision,
)
from .q_format import (
    Q8_8,
    Q16_16,
    QFormat,
    QFormatMixed,
    RoundingMode,
)
from .quantization_reports import (
    PrecisionEnvelopeReport,
    PrecisionTrapReport,
)


def parse_precision_format(fmt: str) -> QFormat | BlockFloatingMode:
    """Parse fixed-point and block-floating precision labels."""
    if not isinstance(fmt, str):
        raise TypeError(f"Expected precision format string, got {type(fmt)!r}")

    text = fmt.strip()
    upper = text.upper()
    if upper.startswith("BFP"):
        return BlockFloatingMode.from_aliases(text)
    if upper.startswith("Q"):
        return QFormat.from_string(text)
    raise ValueError(f"Unsupported precision format: {fmt!r}")


__all__ = [
    "BlockExponentLayout",
    "BlockFloatingMode",
    "CompiledBlockFloatingDense",
    "CompiledMixedDense",
    "PrecisionEnvelopeReport",
    "PrecisionTrapReport",
    "Q8_8",
    "Q16_16",
    "QFormat",
    "QFormatMixed",
    "RoundingMode",
    "compile_dense_block_floating",
    "compile_dense_mixed_precision",
    "dequantize",
    "dequantize_block_floating",
    "dequantize_weights",
    "parse_precision_format",
    "q_weights_to_sc_probabilities",
    "quantization_error",
    "quantize_block_floating",
    "quantize_weights",
]
