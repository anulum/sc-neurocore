# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mixed-precision specification

"""Heterogeneous precision specification for multi-ODE systems."""

from __future__ import annotations

from dataclasses import dataclass

from .precision_config import BlockFloatingPrecisionConfig, PrecisionConfig


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
        """Get the precision config for a variable."""
        if var not in self.var_configs:
            raise KeyError(
                f"Variable '{var}' not in mixed-precision spec. "
                f"Available: {', '.join(self.var_configs.keys())}"
            )
        return self.var_configs[var]

    def require_scalar_encoding(
        self,
        *,
        consumer: str = "scalar precision consumer",
    ) -> None:
        """Reject variables whose precision needs detached block exponents."""
        for var, cfg in self.var_configs.items():
            cfg.require_scalar_encoding(variable=var, consumer=consumer)

    def summary(self) -> str:
        """Return a human-readable summary of the precision allocation."""
        lines = [f"Mixed-Precision Allocation ({self.total_bits} bits total):"]
        for var, cfg in self.var_configs.items():
            range_text = f"[{cfg.min_value:.1f}, {cfg.max_value:.1f}]"
            lines.append(
                f"  {var:12s} → {cfg.q_label:8s} ({cfg.data_width}-bit)"
                f"  range=[{range_text}]  res={cfg.resolution:.6f}"
                f"  kind={cfg.kind}"
            )
        return "\n".join(lines)

    def manifest(
        self,
        *,
        parameter_counts: dict[str, int] | None = None,
    ) -> dict[str, object]:
        """Return deterministic per-variable precision metadata."""
        if parameter_counts is not None:
            unknown = sorted(set(parameter_counts) - set(self.var_configs))
            if unknown:
                raise KeyError(f"Unknown variable(s) in parameter_counts: {', '.join(unknown)}")

        variables: dict[str, object] = {}
        for index, (var, cfg) in enumerate(self.var_configs.items()):
            variable_manifest: dict[str, object]
            if isinstance(cfg, BlockFloatingPrecisionConfig):
                parameter_count = None if parameter_counts is None else parameter_counts.get(var)
                variable_manifest = cfg.manifest_for_parameter_count(parameter_count)
            else:
                variable_manifest = dict(cfg.manifest())
            variable_manifest.update(
                {
                    "variable": var,
                    "assignment_index": index,
                    "emitter_contract_version": "mixed_precision_emitter.v1",
                }
            )
            variables[var] = variable_manifest

        return {
            "kind": "mixed_precision_spec",
            "total_bits": self.total_bits,
            "variable_count": len(self.var_configs),
            "variable_order": list(self.var_configs),
            "variables": variables,
        }
