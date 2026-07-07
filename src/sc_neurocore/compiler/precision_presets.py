# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Precision presets

"""Named precision presets for standard neuromorphic targets."""

from __future__ import annotations

from .block_floating import BlockFloatingMode
from .mixed_precision_spec import MixedPrecisionSpec
from .precision_config import (
    BlockFloatingPrecisionConfig,
    PrecisionConfig,
    PrecisionSpecLike,
)
from .quantizer import parse_precision_format

PRECISION_PRESETS: dict[str, PrecisionConfig | BlockFloatingPrecisionConfig] = {
    "q17": PrecisionConfig(8, 7),
    "q44": PrecisionConfig(8, 4),
    "q88": PrecisionConfig(16, 8),
    "q412": PrecisionConfig(16, 12),
    "q115": PrecisionConfig(16, 15),
    "q99": PrecisionConfig(18, 9),
    "q1212": PrecisionConfig(24, 12),
    "q1413": PrecisionConfig(27, 13),
    "q2012": PrecisionConfig(32, 12),
    "q1616": PrecisionConfig(32, 16),
    "q824": PrecisionConfig(32, 24),
    "q1818": PrecisionConfig(36, 18),
    "bfp16e3x32": BlockFloatingPrecisionConfig(16, 3, 32),
}


def _parse_precision_spec(
    spec: PrecisionSpecLike,
) -> PrecisionConfig | BlockFloatingPrecisionConfig:
    """Parse legacy and explicit precision specs."""
    if isinstance(spec, PrecisionConfig):
        return spec
    if isinstance(spec, BlockFloatingPrecisionConfig):
        return spec

    # parse_precision_format returns QFormat | BlockFloatingMode, so the two cases
    # below are exhaustive once the explicit-config short-circuits above are handled.
    parsed = parse_precision_format(spec)
    if isinstance(parsed, BlockFloatingMode):
        return BlockFloatingPrecisionConfig(
            mantissa_bits=parsed.mantissa_bits,
            exponent_bits=parsed.exponent_bits,
            block_size=parsed.block_size,
        )
    return PrecisionConfig(
        data_width=parsed.total_bits,
        fraction=parsed.fraction_bits,
        signed=True,
    )


def from_preset(
    var_presets: dict[str, PrecisionSpecLike],
    *,
    scalar_only: bool = False,
) -> MixedPrecisionSpec:
    """Create a MixedPrecisionSpec from named presets.

    Set ``scalar_only`` when the downstream consumer cannot carry detached
    block-exponent metadata. Block-floating selections then fail during preset
    resolution instead of later scalar encoding.
    """
    configs: dict[str, PrecisionConfig | BlockFloatingPrecisionConfig] = {}
    for var, preset_name in var_presets.items():
        if isinstance(preset_name, (PrecisionConfig, BlockFloatingPrecisionConfig)):
            configs[var] = preset_name
            continue

        if not isinstance(preset_name, str):
            raise TypeError(f"Unsupported preset for {var!r}: {preset_name!r}")

        try:
            configs[var] = _parse_precision_spec(preset_name)
            continue
        except (ValueError, TypeError):
            pass

        key = preset_name.lower().replace(".", "").replace("-", "").replace("_", "")
        if key not in PRECISION_PRESETS:
            available = ", ".join(sorted(PRECISION_PRESETS.keys()))
            raise KeyError(f"Unknown preset '{preset_name}'. Available: {available}")
        preset_cfg = PRECISION_PRESETS[key]
        configs[var] = preset_cfg
    spec = MixedPrecisionSpec(configs)
    if scalar_only:
        spec.require_scalar_encoding(consumer="from_preset(scalar_only=True)")
    return spec
