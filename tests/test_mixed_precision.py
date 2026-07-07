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
    BlockFloatingScalarEncodingError,
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
    assert manifest["label"] == "BFP16E3X32"
    assert manifest["data_width"] == 16
    assert manifest["fraction"] == 15
    assert manifest["emitted_datapath_width"] == 16
    assert manifest["emitted_datapath_fraction"] == 15
    assert manifest["exponent_stream_width"] == 3
    assert manifest["exponent_bias"] == 3
    assert manifest["exponent_code_range"] == [0, 7]
    assert manifest["exponent_range"] == [-3, 4]
    assert manifest["mantissa_abs_max"] == 32_767
    assert manifest["minimum_quantum"] == pytest.approx(0.125)
    assert manifest["max_abs_value"] == pytest.approx(524_272.0)
    assert manifest["block_exponent_alignment"] == "contiguous_flattened_block"
    assert manifest["block_exponent_count"] == "ceil(parameter_count / block_size)"
    assert manifest["block_exponent_count_policy"] == "ceil(parameter_count / block_size)"
    assert manifest["exponent_vector_width"] == "exponent_bits * ceil(parameter_count / block_size)"
    assert manifest["datapath_contract"] == (
        "fixed_mantissa_with_explicit_shared_exponent_metadata"
    )


def test_block_floating_manifest_carries_concrete_layout_when_count_known() -> None:
    """Concrete parameter counts must produce exact exponent-vector metadata."""

    config = BlockFloatingPrecisionConfig(16, 3, 32)
    manifest = config.manifest_for_parameter_count(65)

    assert manifest["parameter_count"] == 65
    assert manifest["block_exponent_count"] == 3
    assert manifest["exponent_vector_width"] == 9
    assert manifest["block_exponent_layout"] == {
        "alignment": "contiguous_flattened_block",
        "flattened_order": "row_major",
        "parameter_count": 65,
        "block_size": 32,
        "exponent_count": 3,
        "last_block_size": 1,
        "exponent_index_formula": "parameter_index // block_size",
    }


def test_mixed_precision_spec_manifest_rejects_unknown_parameter_counts() -> None:
    """Parameter-count manifests must not silently attach to the wrong variable."""

    spec = from_preset({"v": "bfp16e3x32"})

    variables = cast(
        dict[str, dict[str, object]],
        spec.manifest(parameter_counts={"v": 65})["variables"],
    )
    assert variables["v"]["block_exponent_count"] == 3
    with pytest.raises(KeyError, match="unknown"):
        spec.manifest(parameter_counts={"unknown": 65})


def test_mixed_precision_spec_manifest_is_emitter_facing_contract() -> None:
    """Mixed fixed/BFP manifests must preserve deterministic emitter assignments."""

    spec = from_preset({"v": "q1616", "w": "bfp16e3x32"})
    manifest = spec.manifest(parameter_counts={"w": 65})
    variables = cast(dict[str, dict[str, object]], manifest["variables"])

    assert manifest["variable_order"] == ["v", "w"]
    assert variables["v"]["variable"] == "v"
    assert variables["v"]["assignment_index"] == 0
    assert variables["v"]["kind"] == "fixed"
    assert variables["v"]["label"] == "Q16.16"
    assert variables["v"]["emitted_datapath_width"] == 32
    assert variables["v"]["emitted_datapath_fraction"] == 16
    assert variables["v"]["exponent_stream_width"] == 0
    assert variables["v"]["exponent_vector_width"] == 0
    assert variables["v"]["datapath_contract"] == "fixed_point_twos_complement"

    assert variables["w"]["variable"] == "w"
    assert variables["w"]["assignment_index"] == 1
    assert variables["w"]["kind"] == "block_floating"
    assert variables["w"]["label"] == "BFP16E3X32"
    assert variables["w"]["emitted_datapath_width"] == 16
    assert variables["w"]["emitted_datapath_fraction"] == 15
    assert variables["w"]["block_exponent_count"] == 3
    assert variables["w"]["exponent_stream_width"] == 3
    assert variables["w"]["exponent_vector_width"] == 9
    assert variables["w"]["datapath_contract"] == (
        "fixed_mantissa_with_explicit_shared_exponent_metadata"
    )
    assert variables["w"]["emitter_contract_version"] == "mixed_precision_emitter.v1"


def test_mixed_precision_spec_manifest_rejects_malformed_bfp_parameter_count() -> None:
    """BFP parameter-count metadata must fail closed before emitter handoff."""

    spec = from_preset({"v": "bfp16e3x32"})

    with pytest.raises(TypeError, match="parameter_count"):
        spec.manifest(parameter_counts={"v": cast(Any, 1.5)})


def test_mixed_precision_spec_rejects_bfp_for_scalar_only_consumers() -> None:
    """Scalar-only consumers reject BFP selections before scalar encode calls."""
    spec = from_preset({"fixed": "q88", "weights": "bfp16e3x32"})

    with pytest.raises(BlockFloatingScalarEncodingError, match="weights"):
        spec.require_scalar_encoding(consumer="scalar parameter emitter")

    with pytest.raises(BlockFloatingScalarEncodingError, match="scalar_only"):
        from_preset({"weights": "bfp16e3x32"}, scalar_only=True)


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
