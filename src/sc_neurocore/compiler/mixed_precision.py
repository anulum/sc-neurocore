# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mixed-precision per-variable engine

"""Mixed-precision compilation: different bit widths per state variable.

Provides two APIs:

1. **Dict API** — explicit per-variable precision specification
2. **Constraint solver** — automatic precision selection from bounds + target

Usage (Dict API)::

    from sc_neurocore.compiler.mixed_precision import MixedPrecisionSpec

    spec = MixedPrecisionSpec({
        "v": PrecisionConfig(data_width=16, fraction=8),   # Q8.8
        "u": PrecisionConfig(data_width=8,  fraction=4),   # Q4.4
    })
    # 50% resource savings for the recovery variable

Usage (Constraint Solver)::

    from sc_neurocore.compiler.mixed_precision import solve_precision

    spec = solve_precision(
        bounds={"v": (-128, 127), "u": (-10, 10)},
        min_resolution={"v": 0.01, "u": 0.1},
        max_total_bits=32,
    )
    # → v gets Q8.8 (16-bit), u gets Q4.4 (8-bit) = 24 bits total
"""

from __future__ import annotations

import math
from dataclasses import dataclass


from sc_neurocore.compiler.quantizer import (
    BlockFloatingMode,
    QFormat,
    parse_precision_format,
)

PrecisionSpecLike = str | PrecisionConfig | BlockFloatingPrecisionConfig


@dataclass(frozen=True)
class BlockFloatingPrecisionConfig:
    """Block-floating specification for a single variable.

    The emitter stores only mantissa width in the fixed datapath and carries
    exponent metadata through a detached manifest.
    """

    mantissa_bits: int
    exponent_bits: int
    block_size: int
    signed: bool = True

    @property
    def data_width(self) -> int:
        """Storage width for the mantissa payload."""
        return self.mantissa_bits

    @property
    def fraction(self) -> int:
        """Conservative default fractional estimate.

        Used for any fixed-point compatibility fallbacks.
        """
        return max(1, self.mantissa_bits - 1)

    @property
    def emit_fraction(self) -> int:
        """Deterministic fraction used by compile paths that still emit Q-format RTL."""
        return self.fraction

    @property
    def kind(self) -> str:
        return "block_floating"

    @property
    def int_bits(self) -> int:
        return self.mantissa_bits - 1

    @property
    def max_value(self) -> float:
        return float((1 << (self.exponent_bits - 1)) * ((1 << self.mantissa_bits) - 1))

    @property
    def min_value(self) -> float:
        return -self.max_value

    @property
    def resolution(self) -> float:
        return 2.0 ** (-(self.mantissa_bits - 1))

    @property
    def q_label(self) -> str:
        return f"BFP{self.mantissa_bits}E{self.exponent_bits}X{self.block_size}"

    @property
    def min_exponent(self) -> int:
        return -(1 << (self.exponent_bits - 1))

    @property
    def max_exponent(self) -> int:
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def is_block_floating(self) -> bool:
        return True

    def can_represent(self, value: float) -> bool:
        return self.min_value <= value <= self.max_value

    def encode(self, value: float) -> int:
        del value
        raise NotImplementedError("Block-floating encoding requires per-block exponent metadata.")

    def manifest(self) -> dict[str, float | int | str]:
        return {
            "kind": self.kind,
            "mantissa_bits": self.mantissa_bits,
            "exponent_bits": self.exponent_bits,
            "block_size": self.block_size,
            "signed": self.signed,
            "emitted_fraction": self.emit_fraction,
            "exponent_range": [self.min_exponent, self.max_exponent],
        }


@dataclass(frozen=True)
class PrecisionConfig:
    """Fixed-point configuration for a single variable.

    Attributes
    ----------
    data_width : int
        Total bit width.
    fraction : int
        Number of fractional bits.
    signed : bool
        True for signed two's complement.
    """

    data_width: int
    fraction: int
    signed: bool = True

    @property
    def int_bits(self) -> int:
        """Number of integer bits (excluding sign)."""
        return self.data_width - self.fraction - (1 if self.signed else 0)

    @property
    def max_value(self) -> float:
        """Maximum representable value."""
        if self.signed:
            return ((1 << (self.data_width - 1)) - 1) / (1 << self.fraction)
        return ((1 << self.data_width) - 1) / (1 << self.fraction)

    @property
    def min_value(self) -> float:
        """Minimum representable value."""
        if self.signed:
            return -(1 << (self.data_width - 1)) / (1 << self.fraction)
        return 0.0

    @property
    def resolution(self) -> float:
        """Smallest representable step."""
        return 1.0 / (1 << self.fraction)

    @property
    def q_label(self) -> str:
        """Human-readable Q-format label."""
        prefix = "Q" if self.signed else "UQ"
        return f"{prefix}{self.int_bits}.{self.fraction}"

    @property
    def emit_fraction(self) -> int:
        """Exact fraction for fixed-point emission."""
        return self.fraction

    @property
    def kind(self) -> str:
        return "fixed"

    @property
    def is_block_floating(self) -> bool:
        return False

    def manifest(self) -> dict[str, float | int | str]:
        return {
            "kind": self.kind,
            "data_width": self.data_width,
            "fraction": self.fraction,
            "signed": self.signed,
            "label": self.q_label,
        }

    def can_represent(self, value: float) -> bool:
        """Check if a value fits in this format without overflow."""
        return self.min_value <= value <= self.max_value

    def encode(self, value: float) -> int:
        """Encode a float to Q-format integer."""
        raw = round(value * (1 << self.fraction))
        if self.signed:
            lo = -(1 << (self.data_width - 1))
            hi = (1 << (self.data_width - 1)) - 1
        else:
            lo = 0
            hi = (1 << self.data_width) - 1
        return max(lo, min(hi, raw))


@dataclass
class MixedPrecisionSpec:
    """Specification for mixed-precision compilation.

    Maps each state variable to its own PrecisionConfig, enabling
    heterogeneous datapaths in a single Verilog module.

    Parameters
    ----------
    var_configs : dict[str, PrecisionConfig]
        Per-variable precision configuration.
    """

    var_configs: dict[str, PrecisionConfig | BlockFloatingPrecisionConfig]

    @property
    def total_bits(self) -> int:
        """Total bit count across all variables."""
        return sum(c.data_width for c in self.var_configs.values())

    @property
    def variables(self) -> list[str]:
        """List of variable names."""
        return list(self.var_configs.keys())

    def get(self, var: str) -> PrecisionConfig | BlockFloatingPrecisionConfig:
        """Get the precision config for a variable.

        Parameters
        ----------
        var : str
            Variable name.

        Returns
        -------
        PrecisionConfig
            The precision configuration.

        Raises
        ------
        KeyError
            If the variable is not in the spec.
        """
        if var not in self.var_configs:
            raise KeyError(
                f"Variable '{var}' not in mixed-precision spec. "
                f"Available: {', '.join(self.var_configs.keys())}"
            )
        return self.var_configs[var]

    def summary(self) -> str:
        """Return a human-readable summary of the precision allocation.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [f"Mixed-Precision Allocation ({self.total_bits} bits total):"]
        for var, cfg in self.var_configs.items():
            range_text = _precision_range(cfg)
            lines.append(
                f"  {var:12s} → {cfg.q_label:8s} ({cfg.data_width}-bit)"
                f"  range=[{range_text}]  res={cfg.resolution:.6f}"
                f"  kind={cfg.kind}"
            )
        return "\n".join(lines)


def _precision_range(cfg: PrecisionConfig | BlockFloatingPrecisionConfig) -> str:
    """Range descriptor shared between fixed and block formats."""
    return f"[{cfg.min_value:.1f}, {cfg.max_value:.1f}]"


def _parse_precision_spec(spec: PrecisionSpecLike) -> PrecisionConfig | BlockFloatingPrecisionConfig:
    """Parse legacy and explicit precision specs for mixed-precision workflows."""
    if isinstance(spec, PrecisionConfig):
        return spec
    if isinstance(spec, BlockFloatingPrecisionConfig):
        return spec

    parsed = parse_precision_format(spec)
    if isinstance(parsed, BlockFloatingMode):
        return BlockFloatingPrecisionConfig(
            mantissa_bits=parsed.mantissa_bits,
            exponent_bits=parsed.exponent_bits,
            block_size=parsed.block_size,
        )
    if isinstance(parsed, QFormat):
        return PrecisionConfig(
            data_width=parsed.total_bits,
            fraction=parsed.fraction_bits,
            signed=True,
        )

    raise TypeError(f"Unsupported precision spec: {spec!r}")


# ═══════════════════════════════════════════════════════════════════════
# Standard precision configs
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Constraint Solver
# ═══════════════════════════════════════════════════════════════════════


def _min_bits_for_range(lo: float, hi: float, signed: bool = True) -> int:
    """Compute minimum integer bits to cover a value range.

    Parameters
    ----------
    lo, hi : float
        The value range.
    signed : bool
        Whether the format is signed.

    Returns
    -------
    int
        Minimum integer bits needed.
    """
    abs_max = max(abs(lo), abs(hi))
    if abs_max == 0:
        return 1
    return math.ceil(math.log2(abs_max + 1)) + (1 if signed else 0)


def _min_frac_for_resolution(resolution: float) -> int:
    """Compute minimum fractional bits for a target resolution.

    Parameters
    ----------
    resolution : float
        Desired minimum step size.

    Returns
    -------
    int
        Minimum fractional bits needed.
    """
    if resolution <= 0:
        return 16  # Default to high precision
    return math.ceil(math.log2(1.0 / resolution))


def solve_precision(
    bounds: dict[str, tuple[float, float]],
    *,
    min_resolution: dict[str, float] | None = None,
    max_total_bits: int | None = None,
    signed: bool = True,
    align_to: int = 1,
) -> MixedPrecisionSpec:
    """Automatically solve for optimal per-variable precision.

    Uses a constraint-based approach:
    1. For each variable, compute minimum integer bits from bounds
    2. Compute minimum fractional bits from resolution requirements
    3. If max_total_bits is set, iteratively reduce least-critical fractions

    Parameters
    ----------
    bounds : dict
        Mapping from variable name to (min, max) value bounds.
    min_resolution : dict, optional
        Mapping from variable name to minimum required resolution.
        Defaults to 0.01 for all variables.
    max_total_bits : int, optional
        If set, the solver will reduce fractional bits to fit.
    signed : bool
        Whether to use signed formats.
    align_to : int
        Align each variable's data_width to this multiple (1=no alignment,
        8=byte-align, 16=halfword-align).

    Returns
    -------
    MixedPrecisionSpec
        Optimal per-variable precision configuration.
    """
    if min_resolution is None:
        min_resolution = {v: 0.01 for v in bounds}

    configs: dict[str, PrecisionConfig] = {}

    for var, (lo, hi) in bounds.items():
        int_bits = _min_bits_for_range(lo, hi, signed)
        res = min_resolution.get(var, 0.01)
        frac_bits = _min_frac_for_resolution(res)

        # Sign bit
        sign_bits = 1 if signed else 0
        total = sign_bits + int_bits + frac_bits

        # Align
        if align_to > 1:
            total = math.ceil(total / align_to) * align_to

        configs[var] = PrecisionConfig(
            data_width=total,
            fraction=frac_bits,
            signed=signed,
        )

    spec = MixedPrecisionSpec(configs)

    # If total exceeds budget, iteratively reduce least-sensitive fractions
    if max_total_bits is not None:
        while spec.total_bits > max_total_bits:
            # Find the variable with the most fractional bits — reduce it
            worst = max(
                spec.var_configs.keys(),
                key=lambda v: spec.var_configs[v].fraction,
            )
            old = spec.var_configs[worst]
            if old.fraction <= 1:
                break  # Can't reduce further
            new_frac = old.fraction - 1
            new_dw = old.data_width - 1
            if align_to > 1:
                new_dw = math.ceil(new_dw / align_to) * align_to
            spec.var_configs[worst] = PrecisionConfig(
                data_width=new_dw,
                fraction=new_frac,
                signed=signed,
            )

    return spec


def from_preset(
    var_presets: dict[str, PrecisionSpecLike],
) -> MixedPrecisionSpec:
    """Create a MixedPrecisionSpec from named presets.

    Parameters
    ----------
    var_presets : dict[str, str | PrecisionConfig | BlockFloatingPrecisionConfig]
        Mapping from variable name to preset name or precision object.

    Returns
    -------
    MixedPrecisionSpec
        The resulting spec.

    Raises
    ------
    KeyError
        If a preset name is not found.
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
    return MixedPrecisionSpec(configs)
