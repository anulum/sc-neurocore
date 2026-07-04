# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for fixed-point and block-floating precision configs

"""Contracts for the fixed-point and block-floating precision configurations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.compiler.precision_config import (
    BlockFloatingPrecisionConfig,
    PrecisionConfig,
)


def _bfp() -> BlockFloatingPrecisionConfig:
    return BlockFloatingPrecisionConfig(mantissa_bits=16, exponent_bits=3, block_size=32)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"mantissa_bits": 1.0, "exponent_bits": 3, "block_size": 32}, TypeError),
        ({"mantissa_bits": True, "exponent_bits": 3, "block_size": 32}, TypeError),
        ({"mantissa_bits": 16, "exponent_bits": 1.0, "block_size": 32}, TypeError),
        ({"mantissa_bits": 16, "exponent_bits": True, "block_size": 32}, TypeError),
        ({"mantissa_bits": 16, "exponent_bits": 3, "block_size": 1.0}, TypeError),
        ({"mantissa_bits": 16, "exponent_bits": 3, "block_size": True}, TypeError),
        ({"mantissa_bits": 16, "exponent_bits": 3, "block_size": 32, "signed": 1}, TypeError),
        ({"mantissa_bits": 1, "exponent_bits": 3, "block_size": 32}, ValueError),
        ({"mantissa_bits": 16, "exponent_bits": 0, "block_size": 32}, ValueError),
        ({"mantissa_bits": 16, "exponent_bits": 3, "block_size": 0}, ValueError),
    ],
)
def test_block_floating_config_rejects_invalid_fields(
    kwargs: dict[str, Any],
    error: type[Exception],
) -> None:
    """Block-floating configs reject invalid type and range contract fields."""
    with pytest.raises(error):
        BlockFloatingPrecisionConfig(**kwargs)


def test_block_floating_config_properties_and_range() -> None:
    """The block-floating config exposes the complete mantissa/exponent range contract."""
    bfp = _bfp()

    assert bfp.data_width == 16
    assert bfp.fraction == 15
    assert bfp.emit_fraction == 15
    assert bfp.kind == "block_floating"
    assert bfp.int_bits == 15
    assert bfp.exponent_bias == 3
    assert bfp.exponent_code_min == 0
    assert bfp.exponent_code_max == 7
    assert bfp.mantissa_abs_max == 32767
    assert bfp.max_exponent == 4
    assert bfp.min_exponent == -3
    assert bfp.max_value == pytest.approx(524272.0)
    assert bfp.min_value == pytest.approx(-524272.0)
    assert bfp.resolution == pytest.approx(0.125)
    assert bfp.q_label == "BFP16E3X32"
    assert bfp.is_block_floating is True
    assert bfp.can_represent(bfp.max_value)
    assert bfp.can_represent(bfp.min_value)
    assert not bfp.can_represent(bfp.max_value + bfp.resolution)


def test_block_floating_config_delegates_layout_and_exponent_validation() -> None:
    """Block-floating configs delegate exponent layout and validation deterministically."""
    bfp = _bfp()
    count = bfp.block_exponent_count(100)
    layout = bfp.block_exponent_layout(100)

    assert count == 4
    assert layout.parameter_count == 100
    assert layout.block_size == 32
    assert layout.exponent_count == 4
    assert layout.last_block_size == 4
    validated = bfp.validate_exponents(np.zeros(count, dtype=np.int64), parameter_count=100)
    assert validated.shape == (count,)


def test_block_floating_config_manifest_contracts() -> None:
    """Block-floating manifests expose both abstract and concrete layout contracts."""
    bfp = _bfp()

    manifest = bfp.manifest()
    assert manifest["kind"] == "block_floating"
    assert manifest["label"] == "BFP16E3X32"
    assert manifest["data_width"] == 16
    assert manifest["fraction"] == 15
    assert manifest["mantissa_bits"] == 16
    assert manifest["exponent_bits"] == 3
    assert manifest["block_size"] == 32
    assert manifest["signed"] is True
    assert manifest["emitted_fraction"] == 15
    assert manifest["emitted_datapath_width"] == 16
    assert manifest["emitted_datapath_fraction"] == 15
    assert manifest["exponent_stream_width"] == 3
    assert manifest["exponent_bias"] == 3
    assert manifest["exponent_code_range"] == [0, 7]
    assert manifest["exponent_range"] == [-3, 4]
    assert manifest["mantissa_abs_max"] == 32767
    assert manifest["minimum_quantum"] == pytest.approx(0.125)
    assert manifest["max_abs_value"] == pytest.approx(524272.0)
    assert manifest["block_exponent_alignment"] == "contiguous_flattened_block"
    assert manifest["block_exponent_count"] == "ceil(parameter_count / block_size)"
    assert manifest["block_exponent_count_policy"] == "ceil(parameter_count / block_size)"
    assert manifest["exponent_vector_width"] == "exponent_bits * ceil(parameter_count / block_size)"
    assert manifest["datapath_contract"] == "fixed_mantissa_with_explicit_shared_exponent_metadata"

    concrete = bfp.manifest_for_parameter_count(100)
    assert concrete["parameter_count"] == 100
    assert concrete["block_exponent_count"] == 4
    assert concrete["exponent_vector_width"] == 12
    assert concrete["block_exponent_layout"] == {
        "alignment": "contiguous_flattened_block",
        "flattened_order": "row_major",
        "parameter_count": 100,
        "block_size": 32,
        "exponent_count": 4,
        "last_block_size": 4,
        "exponent_index_formula": "parameter_index // block_size",
    }


def test_block_floating_config_encode_is_unsupported() -> None:
    """Block-floating encoding without per-block exponents is unsupported."""
    with pytest.raises(NotImplementedError):
        _bfp().encode(1.0)


def test_precision_config_signed_and_unsigned_ranges() -> None:
    """Fixed-point configs expose complete signed and unsigned dynamic ranges."""
    signed = PrecisionConfig(8, 7)
    unsigned = PrecisionConfig(8, 7, signed=False)

    assert signed.int_bits == 0
    assert signed.max_value == pytest.approx(127 / 128)
    assert signed.min_value == pytest.approx(-1.0)
    assert signed.resolution == pytest.approx(1 / 128)
    assert signed.q_label == "Q1.7"
    assert signed.emit_fraction == 7
    assert signed.kind == "fixed"
    assert signed.is_block_floating is False
    assert signed.can_represent(-1.0)
    assert signed.can_represent(127 / 128)
    assert not signed.can_represent(1.0)

    assert unsigned.int_bits == 1
    assert unsigned.max_value == pytest.approx(255 / 128)
    assert unsigned.min_value == pytest.approx(0.0)
    assert unsigned.resolution == pytest.approx(1 / 128)
    assert unsigned.q_label == "UQ1.7"
    assert unsigned.emit_fraction == 7
    assert unsigned.kind == "fixed"
    assert unsigned.is_block_floating is False
    assert unsigned.can_represent(255 / 128)
    assert not unsigned.can_represent(-1 / 128)


def test_precision_config_manifest_and_saturating_encode() -> None:
    """Fixed-point manifests and scalar encoders preserve the public compiler contract."""
    signed = PrecisionConfig(8, 7)
    unsigned = PrecisionConfig(8, 7, signed=False)

    assert signed.manifest() == {
        "kind": "fixed",
        "data_width": 8,
        "fraction": 7,
        "signed": True,
        "label": "Q1.7",
        "emitted_datapath_width": 8,
        "emitted_datapath_fraction": 7,
        "exponent_stream_width": 0,
        "exponent_vector_width": 0,
        "datapath_contract": "fixed_point_twos_complement",
    }
    assert signed.encode(0.5) == 64
    assert signed.encode(2.0) == 127
    assert signed.encode(-2.0) == -128

    assert unsigned.manifest() == {
        "kind": "fixed",
        "data_width": 8,
        "fraction": 7,
        "signed": False,
        "label": "UQ1.7",
        "emitted_datapath_width": 8,
        "emitted_datapath_fraction": 7,
        "exponent_stream_width": 0,
        "exponent_vector_width": 0,
        "datapath_contract": "fixed_point_twos_complement",
    }
    assert unsigned.encode(0.5) == 64
    assert unsigned.encode(2.0) == 255
    assert unsigned.encode(-1.0) == 0
