# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for mixed precision compiler contracts

from __future__ import annotations

from typing import Any, cast

import pytest

from sc_neurocore.compiler.mixed_precision import (
    BlockFloatingPrecisionConfig,
    from_preset,
)


def test_block_floating_precision_config_matches_quantizer_exponent_contract() -> None:
    """BFP16E3X32 must use every exponent code with the quantizer bias."""

    config = BlockFloatingPrecisionConfig(16, 3, 32)

    assert config.exponent_bias == 3
    assert config.min_exponent == -3
    assert config.max_exponent == 4
    assert config.exponent_code_min == 0
    assert config.exponent_code_max == 7
    assert config.mantissa_abs_max == 32_767
    assert config.resolution == pytest.approx(0.125)
    assert config.max_value == pytest.approx(524_272.0)
    assert config.can_represent(-524_272.0)
    assert config.can_represent(524_272.0)
    assert not config.can_represent(524_272.125)


def test_block_floating_manifest_carries_alignment_metadata() -> None:
    """Manifest must contain enough metadata for downstream block alignment."""

    manifest = BlockFloatingPrecisionConfig(16, 3, 32).manifest()

    assert manifest["kind"] == "block_floating"
    assert manifest["exponent_bias"] == 3
    assert manifest["exponent_code_range"] == [0, 7]
    assert manifest["exponent_range"] == [-3, 4]
    assert manifest["mantissa_abs_max"] == 32_767
    assert manifest["minimum_quantum"] == pytest.approx(0.125)
    assert manifest["max_abs_value"] == pytest.approx(524_272.0)
    assert manifest["block_exponent_alignment"] == "contiguous_flattened_block"
    assert manifest["block_exponent_count"] == "ceil(parameter_count / block_size)"


def test_from_preset_block_floating_preserves_exact_metadata() -> None:
    """Named BFP presets must preserve exact exponent and block semantics."""

    spec = from_preset({"v": "bfp16e3x32"})
    config = spec.get("v")

    assert isinstance(config, BlockFloatingPrecisionConfig)
    assert config.manifest()["exponent_range"] == [-3, 4]
    assert config.manifest()["exponent_code_range"] == [0, 7]


def test_block_floating_precision_config_rejects_invalid_contracts() -> None:
    """Invalid BFP precision contracts must fail before manifest emission."""

    with pytest.raises(ValueError, match="mantissa_bits"):
        BlockFloatingPrecisionConfig(1, 3, 32)
    with pytest.raises(ValueError, match="exponent_bits"):
        BlockFloatingPrecisionConfig(16, 0, 32)
    with pytest.raises(ValueError, match="block_size"):
        BlockFloatingPrecisionConfig(16, 3, 0)
    with pytest.raises(TypeError, match="signed"):
        BlockFloatingPrecisionConfig(16, 3, 32, signed=cast(Any, 1))
