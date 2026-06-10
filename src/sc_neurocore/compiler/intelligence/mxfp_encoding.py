# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MXFP encoding

"""Microsoft Microscaling (MX) and IEEE FP8 encoding utilities.

Supports OCP Microscaling Formats Specification v1.0 (2024).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MXFPConfig:
    """Microsoft Microscaling (MX) floating-point format.

    Based on OCP Microscaling Formats Specification v1.0 (2024).

    Attributes
    ----------
    element_bits : int
        Bits per element (4, 6, or 8).
    exp_bits : int
        Exponent bits per element.
    mantissa_bits : int
        Mantissa bits per element (including implicit 1).
    block_size : int
        Elements per shared-exponent block.
    shared_exp_bits : int
        Shared exponent width (typically 8).
    """

    element_bits: int
    exp_bits: int
    mantissa_bits: int
    block_size: int = 32
    shared_exp_bits: int = 8

    @property
    def label(self) -> str:
        """Human-readable format label."""
        return f"MXFP{self.element_bits}"

    @property
    def bits_per_block(self) -> int:
        """Total bits for one block including shared exponent."""
        return self.shared_exp_bits + self.block_size * self.element_bits


MXFP4 = MXFPConfig(element_bits=4, exp_bits=2, mantissa_bits=1, block_size=32)
MXFP6 = MXFPConfig(element_bits=6, exp_bits=3, mantissa_bits=2, block_size=32)
MXFP8_E4M3 = MXFPConfig(element_bits=8, exp_bits=4, mantissa_bits=3, block_size=32)
MXFP8_E5M2 = MXFPConfig(element_bits=8, exp_bits=5, mantissa_bits=2, block_size=32)

# IEEE FP8 variants (block_size=1)
FP8_E4M3 = MXFPConfig(element_bits=8, exp_bits=4, mantissa_bits=3, block_size=1, shared_exp_bits=0)
FP8_E5M2 = MXFPConfig(element_bits=8, exp_bits=5, mantissa_bits=2, block_size=1, shared_exp_bits=0)


def mxfp_encode_block(
    values: list[float],
    config: MXFPConfig,
) -> tuple[int, list[int]]:
    """Encode a block of floats to MXFP format.

    Parameters
    ----------
    values : list[float]
        Block of float values (len must equal config.block_size).
    config : MXFPConfig
        MXFP format configuration.

    Returns
    -------
    tuple[int, list[int]]
        (shared_exponent, list_of_encoded_elements).
    """
    if len(values) != config.block_size:
        raise ValueError(f"Block size mismatch: got {len(values)}, expected {config.block_size}")

    # Find shared exponent (max abs value)
    abs_max = max(abs(v) for v in values) if values else 0.0
    if abs_max == 0:
        return (0, [0] * config.block_size)

    # Shared exponent = floor(log2(abs_max)) + bias
    exp_bias = (1 << (config.shared_exp_bits - 1)) - 1 if config.shared_exp_bits else 0
    shared_exp = int(math.floor(math.log2(abs_max))) + exp_bias if abs_max > 0 else 0
    shared_exp = (
        max(0, min((1 << config.shared_exp_bits) - 1, shared_exp)) if config.shared_exp_bits else 0
    )

    # Scale factor
    scale = 2.0 ** (shared_exp - exp_bias) if config.shared_exp_bits else 1.0
    max_mant = (1 << config.mantissa_bits) - 1

    encoded = []
    for v in values:
        sign = 1 if v < 0 else 0
        scaled = abs(v) / scale if scale > 0 else 0.0
        mant = min(max_mant, int(round(scaled * max_mant)))
        # Pack: sign | element mantissa
        elem = (sign << (config.element_bits - 1)) | mant
        encoded.append(elem & ((1 << config.element_bits) - 1))

    return (shared_exp, encoded)


def mxfp_decode_block(
    shared_exp: int,
    elements: list[int],
    config: MXFPConfig,
) -> list[float]:
    """Decode a block of MXFP elements to floats.

    Parameters
    ----------
    shared_exp : int
        Shared exponent.
    elements : list[int]
        Encoded element integers.
    config : MXFPConfig
        MXFP format configuration.

    Returns
    -------
    list[float]
        Decoded float values.
    """
    exp_bias = (1 << (config.shared_exp_bits - 1)) - 1 if config.shared_exp_bits else 0
    scale = 2.0 ** (shared_exp - exp_bias) if config.shared_exp_bits else 1.0
    max_mant = (1 << config.mantissa_bits) - 1

    decoded = []
    for elem in elements:
        sign = (elem >> (config.element_bits - 1)) & 1
        mant = elem & ((1 << (config.element_bits - 1)) - 1)
        value = (mant / max_mant) * scale if max_mant > 0 else 0.0
        decoded.append(-value if sign else value)

    return decoded
