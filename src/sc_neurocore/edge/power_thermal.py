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
import re
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


@dataclass(frozen=True)
class VivadoPowerReport:
    """Power and thermal values parsed from a routed Vivado power report."""

    path: str
    tool_version: str
    report_date: str
    design: str
    device: str
    total_on_chip_power_w: float
    dynamic_power_w: float
    static_power_w: float
    effective_tja_c_per_w: float
    max_ambient_c: float
    junction_temperature_c: float
    confidence_level: str


@dataclass(frozen=True)
class VivadoUtilizationReport:
    """Resource counts parsed from a Vivado utilisation report."""

    path: str
    tool_version: str
    report_date: str
    design: str
    device: str
    slice_luts: int
    slice_luts_available: int
    slice_luts_util_pct: float
    slice_registers: int
    slice_registers_available: int
    slice_registers_util_pct: float
    block_ram_tiles: int
    block_ram_tiles_available: int
    block_ram_tiles_util_pct: float
    dsps: int
    dsps_available: int
    dsps_util_pct: float


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


def build_power_thermal_model_from_vivado_reports(
    report_dir: str | Path,
    config: PowerThermalConfig,
    *,
    power_report: str = "system_wrapper_power_routed.rpt",
    utilization_report: str = "system_wrapper_utilization_placed.rpt",
) -> dict[str, Any]:
    """Build a power/thermal model seeded from Vivado implementation reports.

    The returned model keeps the estimator resource breakdown for workload
    context, but replaces headline power and thermal values with the routed
    report values from Vivado.
    """

    report_root = Path(report_dir)
    power = parse_vivado_power_report(report_root / power_report)
    utilisation = parse_vivado_utilization_report(report_root / utilization_report)
    report_config = PowerThermalConfig(
        target=config.target,
        layer_sizes=config.layer_sizes,
        bitstream_length=config.bitstream_length,
        clock_mhz=config.clock_mhz,
        event_driven=config.event_driven,
        include_infra=config.include_infra,
        ambient_c=power.junction_temperature_c
        - power.total_on_chip_power_w * power.effective_tja_c_per_w,
        theta_ja_c_per_w=power.effective_tja_c_per_w,
        measured_power_mw=power.total_on_chip_power_w * 1000.0,
        source_reports={
            **config.source_reports,
            "power": str((report_root / power_report).as_posix()),
            "utilization": str((report_root / utilization_report).as_posix()),
        },
        artefact_name=config.artefact_name,
    )
    model = build_power_thermal_model(report_config)
    model["source_mode"] = "vivado_report_derived"
    model["power"].update(
        {
            "basis": "Vivado routed report_power",
            "reported_dynamic_power_mw": power.dynamic_power_w * 1000.0,
            "reported_static_power_mw": power.static_power_w * 1000.0,
            "confidence_level": power.confidence_level,
        }
    )
    model["thermal"].update(
        {
            "reported_junction_c": power.junction_temperature_c,
            "reported_max_ambient_c": power.max_ambient_c,
        }
    )
    model["implementation"] = {
        "tool_version": power.tool_version,
        "report_date": power.report_date,
        "design": power.design,
        "device": power.device,
        "utilisation": {
            "slice_luts": utilisation.slice_luts,
            "slice_luts_available": utilisation.slice_luts_available,
            "slice_luts_util_pct": utilisation.slice_luts_util_pct,
            "slice_registers": utilisation.slice_registers,
            "slice_registers_available": utilisation.slice_registers_available,
            "slice_registers_util_pct": utilisation.slice_registers_util_pct,
            "block_ram_tiles": utilisation.block_ram_tiles,
            "block_ram_tiles_available": utilisation.block_ram_tiles_available,
            "block_ram_tiles_util_pct": utilisation.block_ram_tiles_util_pct,
            "dsps": utilisation.dsps,
            "dsps_available": utilisation.dsps_available,
            "dsps_util_pct": utilisation.dsps_util_pct,
        },
    }
    model["limitations"] = [
        "Vivado power is report-derived and still requires physical board measurement",
        "thermal values use Vivado's effective TJA and routed report assumptions",
        "activity accuracy depends on the switching data available to report_power",
    ]
    return model


def parse_vivado_power_report(path: str | Path) -> VivadoPowerReport:
    """Parse headline power and thermal fields from a Vivado report_power file."""

    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8", errors="replace")
    return VivadoPowerReport(
        path=str(report_path.as_posix()),
        tool_version=_extract_header(text, "Tool Version"),
        report_date=_extract_header(text, "Date"),
        design=_extract_header(text, "Design"),
        device=_extract_header(text, "Device"),
        total_on_chip_power_w=_extract_summary_float(text, "Total On-Chip Power (W)"),
        dynamic_power_w=_extract_summary_float(text, "Dynamic (W)"),
        static_power_w=_extract_summary_float(text, "Device Static (W)"),
        effective_tja_c_per_w=_extract_summary_float(text, "Effective TJA (C/W)"),
        max_ambient_c=_extract_summary_float(text, "Max Ambient (C)"),
        junction_temperature_c=_extract_summary_float(text, "Junction Temperature (C)"),
        confidence_level=_extract_summary_text(text, "Confidence Level"),
    )


def parse_vivado_utilization_report(path: str | Path) -> VivadoUtilizationReport:
    """Parse FPGA resource counts from a Vivado report_utilization file."""

    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8", errors="replace")
    slice_luts = _extract_utilisation_row(text, "Slice LUTs")
    slice_registers = _extract_utilisation_row(text, "Slice Registers")
    block_ram_tiles = _extract_utilisation_row(text, "Block RAM Tile")
    dsps = _extract_utilisation_row(text, "DSPs")
    return VivadoUtilizationReport(
        path=str(report_path.as_posix()),
        tool_version=_extract_header(text, "Tool Version"),
        report_date=_extract_header(text, "Date"),
        design=_extract_header(text, "Design"),
        device=_extract_header(text, "Device"),
        slice_luts=slice_luts[0],
        slice_luts_available=slice_luts[1],
        slice_luts_util_pct=slice_luts[2],
        slice_registers=slice_registers[0],
        slice_registers_available=slice_registers[1],
        slice_registers_util_pct=slice_registers[2],
        block_ram_tiles=block_ram_tiles[0],
        block_ram_tiles_available=block_ram_tiles[1],
        block_ram_tiles_util_pct=block_ram_tiles[2],
        dsps=dsps[0],
        dsps_available=dsps[1],
        dsps_util_pct=dsps[2],
    )


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


def write_power_thermal_model_from_vivado_reports(
    report_dir: str | Path,
    output_dir: str | Path,
    config: PowerThermalConfig,
    *,
    power_report: str = "system_wrapper_power_routed.rpt",
    utilization_report: str = "system_wrapper_utilization_placed.rpt",
) -> Path:
    """Write report-derived FPGA power/thermal JSON beside deployable artefacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / config.artefact_name
    model = build_power_thermal_model_from_vivado_reports(
        report_dir,
        config,
        power_report=power_report,
        utilization_report=utilization_report,
    )
    destination.write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n",
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


def _extract_header(text: str, label: str) -> str:
    pattern = re.compile(rf"^\|\s*{re.escape(label)}\s*:\s*(.*?)\s*$", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"missing Vivado header field: {label}")
    return match.group(1).rstrip("|").strip()


def _extract_summary_float(text: str, label: str) -> float:
    raw = _extract_summary_text(text, label)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Vivado field {label!r} is not numeric: {raw!r}") from exc


def _extract_summary_text(text: str, label: str) -> str:
    pattern = re.compile(rf"^\|\s*{re.escape(label)}\s*\|\s*([^|]+?)\s*\|", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"missing Vivado summary field: {label}")
    return match.group(1).strip()


def _extract_utilisation_row(text: str, label: str) -> tuple[int, int, float]:
    pattern = re.compile(
        rf"^\|\s*{re.escape(label)}\s*\|\s*([0-9]+)\s*\|\s*[0-9]+\s*\|"
        rf"\s*[0-9]*\s*\|\s*([0-9]+)\s*\|\s*([0-9.]+)\s*\|",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"missing Vivado utilisation row: {label}")
    return int(match.group(1)), int(match.group(2)), float(match.group(3))
