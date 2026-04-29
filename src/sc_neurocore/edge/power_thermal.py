# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA power and thermal deployment model

"""Deterministic FPGA power and thermal model export.

The model is a pre-silicon estimate unless an explicit measured power value is
provided. It does not claim board measurements, signoff power, or tapeout
readiness.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sc_neurocore.energy.estimator import EnergyReport, estimate
from sc_neurocore.energy.fpga_models import TARGETS

SCHEMA_VERSION = "sc-neurocore.fpga.power-thermal.v1"

_DEFAULT_THETA_JA_C_PER_W = {
    "ice40": 38.0,
    "ecp5": 24.0,
    "artix7": 18.0,
    "zynq": 16.0,
}


@dataclass(frozen=True)
class PowerThermalConfig:
    """Inputs for an FPGA power and thermal deployment estimate."""

    target: str = "ice40"
    layer_sizes: tuple[tuple[int, int], ...] = ((1, 1),)
    bitstream_length: int = 256
    clock_mhz: float = 100.0
    event_driven: bool = False
    include_infra: bool = True
    ambient_c: float = 25.0
    theta_ja_c_per_w: float | None = None
    measured_power_mw: float | None = None
    source_reports: dict[str, str] = field(default_factory=dict)
    artefact_name: str = "power_thermal_model.json"

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"unknown FPGA target '{self.target}'")
        if not self.layer_sizes:
            raise ValueError("layer_sizes must not be empty")
        for n_inputs, n_neurons in self.layer_sizes:
            if n_inputs <= 0 or n_neurons <= 0:
                raise ValueError("layer_sizes entries must be positive")
        if self.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if self.clock_mhz <= 0.0 or not math.isfinite(self.clock_mhz):
            raise ValueError("clock_mhz must be finite and positive")
        if not math.isfinite(self.ambient_c):
            raise ValueError("ambient_c must be finite")
        if self.theta_ja_c_per_w is not None and (
            self.theta_ja_c_per_w <= 0.0 or not math.isfinite(self.theta_ja_c_per_w)
        ):
            raise ValueError("theta_ja_c_per_w must be finite and positive")
        if self.measured_power_mw is not None and (
            self.measured_power_mw < 0.0 or not math.isfinite(self.measured_power_mw)
        ):
            raise ValueError("measured_power_mw must be finite and non-negative")


def build_power_thermal_model(config: PowerThermalConfig) -> dict[str, Any]:
    """Build a deterministic JSON-compatible FPGA power/thermal model."""

    report = estimate(
        list(config.layer_sizes),
        target=config.target,
        bitstream_length=config.bitstream_length,
        event_driven=config.event_driven,
        clock_mhz=config.clock_mhz,
        include_infra=config.include_infra,
    )
    theta = config.theta_ja_c_per_w or _DEFAULT_THETA_JA_C_PER_W[config.target]
    power_mw = (
        config.measured_power_mw
        if config.measured_power_mw is not None
        else report.total_dynamic_power_mw
    )
    source_mode = (
        "report_derived" if config.measured_power_mw is not None else "pre_silicon_estimate"
    )
    junction_c = config.ambient_c + (power_mw / 1000.0) * theta

    return {
        "schema_version": SCHEMA_VERSION,
        "source_mode": source_mode,
        "target": _target_payload(config.target),
        "workload": {
            "layer_sizes": [list(layer) for layer in config.layer_sizes],
            "bitstream_length": config.bitstream_length,
            "clock_mhz": config.clock_mhz,
            "event_driven": config.event_driven,
            "include_infra": config.include_infra,
        },
        "resource_estimate": _resource_payload(report),
        "power": {
            "basis": (
                "explicit measured/report power"
                if config.measured_power_mw is not None
                else "dynamic pre-silicon estimate from sc_neurocore.energy.estimator"
            ),
            "total_power_mw": power_mw,
            "dynamic_power_estimate_mw": report.total_dynamic_power_mw,
            "energy_per_inference_nj": report.energy_per_inference_nj,
        },
        "thermal": {
            "ambient_c": config.ambient_c,
            "theta_ja_c_per_w": theta,
            "theta_source": "caller"
            if config.theta_ja_c_per_w is not None
            else "board_profile_default",
            "estimated_junction_c": junction_c,
            "margin_to_85c": 85.0 - junction_c,
        },
        "source_reports": dict(sorted(config.source_reports.items())),
        "limitations": [
            "pre-silicon estimates are not a substitute for board power measurement",
            "default thermal resistance is a board-profile assumption unless caller supplied theta_ja_c_per_w",
            "static power is not included unless measured_power_mw is supplied",
        ],
    }


def write_power_thermal_model(output_dir: str | Path, config: PowerThermalConfig) -> Path:
    """Write `power_thermal_model.json` beside generated FPGA artefacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / config.artefact_name
    destination.write_text(
        json.dumps(build_power_thermal_model(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _target_payload(target: str) -> dict[str, Any]:
    spec = TARGETS[target]
    return {
        "name": spec.name,
        "family": spec.family,
        "total_luts": spec.total_luts,
        "total_bram_kb": spec.total_bram_kb,
        "total_dsp": spec.total_dsp,
        "voltage": spec.voltage,
        "max_freq_mhz": spec.max_freq_mhz,
    }


def _resource_payload(report: EnergyReport) -> dict[str, Any]:
    return {
        "total_luts": report.total_luts,
        "total_ffs": report.total_ffs,
        "total_bram_kb": report.total_bram_kb,
        "infra_luts": report.infra_luts,
        "total_latency_cycles": report.total_latency_cycles,
        "fits_on_target": report.fits_on_target,
        "utilisation_pct": report.utilization_pct,
        "layers": [
            {
                "name": layer.name,
                "n_inputs": layer.n_inputs,
                "n_neurons": layer.n_neurons,
                "n_synapses": layer.n_synapses,
                "bitstream_length": layer.bitstream_length,
                "luts": layer.luts,
                "ffs": layer.ffs,
                "bram_bits": layer.bram_bits,
                "dynamic_power_mw": layer.dynamic_power_mw,
                "latency_cycles": layer.latency_cycles,
            }
            for layer in report.layers
        ],
    }
