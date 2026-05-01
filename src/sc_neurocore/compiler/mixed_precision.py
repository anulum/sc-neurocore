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

    var_configs: dict[str, PrecisionConfig]

    @property
    def total_bits(self) -> int:
        """Total bit count across all variables."""
        return sum(c.data_width for c in self.var_configs.values())

    @property
    def variables(self) -> list[str]:
        """List of variable names."""
        return list(self.var_configs.keys())

    def get(self, var: str) -> PrecisionConfig:
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
            lines.append(
                f"  {var:12s} → {cfg.q_label:8s} ({cfg.data_width}-bit)"
                f"  range=[{cfg.min_value:.1f}, {cfg.max_value:.1f}]"
                f"  res={cfg.resolution:.6f}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Standard precision configs
# ═══════════════════════════════════════════════════════════════════════

PRECISION_PRESETS: dict[str, PrecisionConfig] = {
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
    var_presets: dict[str, str],
) -> MixedPrecisionSpec:
    """Create a MixedPrecisionSpec from named presets.

    Parameters
    ----------
    var_presets : dict[str, str]
        Mapping from variable name to preset name (e.g. ``{"v": "q88", "u": "q44"}``).

    Returns
    -------
    MixedPrecisionSpec
        The resulting spec.

    Raises
    ------
    KeyError
        If a preset name is not found.
    """
    configs: dict[str, PrecisionConfig] = {}
    for var, preset_name in var_presets.items():
        key = preset_name.lower().replace(".", "").replace("-", "").replace("_", "")
        if key not in PRECISION_PRESETS:
            available = ", ".join(sorted(PRECISION_PRESETS.keys()))
            raise KeyError(f"Unknown preset '{preset_name}'. Available: {available}")
        configs[var] = PRECISION_PRESETS[key]
    return MixedPrecisionSpec(configs)
