# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive runtime manifest

"""Deterministic metadata generation for adaptive-precision contracts."""

from __future__ import annotations

from typing import Any

from .quantizer import BlockFloatingMode, QFormat


def _precision_label(parsed: QFormat | BlockFloatingMode, *, source: str) -> str:
    """Return a deterministic textual label for telemetry."""
    if isinstance(parsed, QFormat):
        return f"Q{parsed.integer_bits}.{parsed.fraction_bits}"

    label = parsed.label
    if source.upper().endswith(f"X{parsed.block_size}"):
        return label
    return f"{label}X{parsed.block_size}"


def _precision_manifest(
    parsed: QFormat | BlockFloatingMode,
    source: str,
    resolved_width: int,
    emitted_fraction: int,
    *,
    kind: str,
    parameter_count: int | None = None,
) -> dict[str, Any]:
    """Build deterministic metadata for precision contracts."""
    if isinstance(parsed, QFormat):
        if parameter_count is not None:
            raise ValueError("parameter_count metadata is only valid for block-floating precision")
        return {
            "kind": kind,
            "source": source,
            "label": f"Q{parsed.integer_bits}.{parsed.fraction_bits}",
            "data_width": resolved_width,
            "fraction": parsed.fraction_bits,
            "signed": True,
            "emitted_fraction": emitted_fraction,
            "emitted_datapath_width": resolved_width,
            "emitted_datapath_fraction": emitted_fraction,
            "exponent_stream_width": 0,
            "exponent_vector_width": 0,
            "datapath_contract": "fixed_point_twos_complement",
            "emitter_contract_version": "adaptive_precision_emitter.v1",
        }

    metadata: dict[str, Any] = dict(parsed.metadata)
    metadata.update(
        {
            "kind": kind,
            "source": source,
            "label": _precision_label(parsed, source=source),
            "data_width": resolved_width,
            "fraction": emitted_fraction,
            "signed": True,
            "emitted_fraction": emitted_fraction,
            "emitted_datapath_width": resolved_width,
            "emitted_datapath_fraction": emitted_fraction,
            "emitted_datapath_contract": (
                "mantissa_width_fixed_datapath_with_detached_shared_exponent_stream"
            ),
            "exponent_stream_width": parsed.exponent_bits,
            "exponent_bias": parsed.exponent_bias,
            "exponent_code_range": [0, (1 << parsed.exponent_bits) - 1],
            "mantissa_abs_max": parsed.mantissa_range,
            "minimum_quantum": 2.0**parsed.min_exponent,
            "max_abs_value": float(parsed.mantissa_range) * (2.0**parsed.max_exponent),
            "block_exponent_alignment": "contiguous_flattened_block",
            "block_exponent_count": "ceil(parameter_count / block_size)",
            "block_exponent_count_policy": "ceil(parameter_count / block_size)",
            "exponent_vector_width": "exponent_bits * ceil(parameter_count / block_size)",
            "datapath_contract": "fixed_mantissa_with_explicit_shared_exponent_metadata",
            "bfp_emission_status": "metadata_only_until_target_bfp_datapath_selection",
            "emitter_contract_version": "adaptive_precision_emitter.v1",
        }
    )
    if parameter_count is not None:
        layout = parsed.block_exponent_layout(parameter_count)
        metadata.update(
            {
                "parameter_count": parameter_count,
                "block_exponent_count": layout.exponent_count,
                "block_exponent_layout": layout.manifest(),
                "exponent_vector_width": parsed.exponent_bits * layout.exponent_count,
            }
        )
    return metadata
