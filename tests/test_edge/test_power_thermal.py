# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA power and thermal model tests

"""Tests for FPGA power and thermal deployment artefacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sc_neurocore.edge.power_thermal import (
    SCHEMA_VERSION,
    PowerThermalConfig,
    build_power_thermal_model,
    build_power_thermal_model_from_vivado_reports,
    parse_vivado_power_report,
    parse_vivado_utilization_report,
    write_power_thermal_model,
    write_power_thermal_model_from_vivado_reports,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PYNQ_IMPL_DIR = REPO_ROOT / "sc_shd_pynq" / "sc_shd_pynq.runs" / "impl_1"


def test_power_thermal_model_is_deterministic_pre_silicon_estimate() -> None:
    config = PowerThermalConfig(
        target="ice40",
        layer_sizes=((4, 8), (8, 2)),
        bitstream_length=128,
        clock_mhz=50.0,
        event_driven=True,
    )

    first = build_power_thermal_model(config)
    second = build_power_thermal_model(config)

    assert first == second
    assert first["schema_version"] == SCHEMA_VERSION
    assert first["source_mode"] == "pre_silicon_estimate"
    assert first["target"]["family"] == "ice40"
    assert first["workload"]["layer_sizes"] == [[4, 8], [8, 2]]
    assert first["power"]["total_power_mw"] == first["power"]["dynamic_power_estimate_mw"]
    assert first["power"]["total_power_mw"] > 0.0
    assert first["thermal"]["estimated_junction_c"] > first["thermal"]["ambient_c"]
    assert first["thermal"]["theta_source"] == "board_profile_default"
    assert "not a substitute for board power measurement" in first["limitations"][0]


def test_power_thermal_model_uses_explicit_report_power() -> None:
    model = build_power_thermal_model(
        PowerThermalConfig(
            target="artix7",
            layer_sizes=((16, 4),),
            measured_power_mw=250.0,
            theta_ja_c_per_w=10.0,
            source_reports={"power": "reports/power.rpt"},
        )
    )

    assert model["source_mode"] == "report_derived"
    assert model["power"]["total_power_mw"] == 250.0
    assert model["thermal"]["theta_source"] == "caller"
    assert model["thermal"]["estimated_junction_c"] == 27.5
    assert model["source_reports"] == {"power": "reports/power.rpt"}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"target": "unknown"}, "unknown FPGA target"),
        ({"layer_sizes": ()}, "layer_sizes must not be empty"),
        ({"layer_sizes": ((0, 1),)}, "layer_sizes entries must be positive"),
        ({"bitstream_length": 0}, "bitstream_length must be positive"),
        ({"clock_mhz": 0.0}, "clock_mhz must be finite and positive"),
        ({"clock_mhz": math.inf}, "clock_mhz must be finite and positive"),
        ({"ambient_c": math.nan}, "ambient_c must be finite"),
        ({"theta_ja_c_per_w": -1.0}, "theta_ja_c_per_w must be finite and positive"),
        ({"measured_power_mw": -1.0}, "measured_power_mw must be finite and non-negative"),
    ],
)
def test_power_thermal_config_rejects_invalid_inputs(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PowerThermalConfig(**kwargs)


def test_write_power_thermal_model_writes_sorted_json(tmp_path: Path) -> None:
    output = tmp_path / "deploy"
    path = write_power_thermal_model(
        output,
        PowerThermalConfig(target="ecp5", layer_sizes=((2, 2),), bitstream_length=64),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path == output / "power_thermal_model.json"
    assert payload["target"]["family"] == "ecp5"
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_write_power_thermal_model_honours_custom_artefact_name(tmp_path: Path) -> None:
    output = tmp_path / "deploy"
    path = write_power_thermal_model(
        output,
        PowerThermalConfig(
            target="ecp5",
            layer_sizes=((2, 2),),
            bitstream_length=64,
            artefact_name="custom_power.json",
        ),
    )
    assert path == output / "custom_power.json"


def test_parse_vivado_power_report_extracts_routed_pynq_values() -> None:
    report = parse_vivado_power_report(PYNQ_IMPL_DIR / "system_wrapper_power_routed.rpt")

    assert report.tool_version.startswith("Vivado v.2025.2")
    assert report.design == "system_wrapper"
    assert report.device == "xc7z020clg400-1"
    assert report.total_on_chip_power_w == 1.674
    assert report.dynamic_power_w == 1.538
    assert report.static_power_w == 0.136
    assert report.effective_tja_c_per_w == 11.5
    assert report.junction_temperature_c == 44.3
    assert report.confidence_level == "Medium"


def test_parse_vivado_utilization_report_extracts_routed_pynq_resources() -> None:
    report = parse_vivado_utilization_report(
        PYNQ_IMPL_DIR / "system_wrapper_utilization_placed.rpt"
    )

    assert report.tool_version.startswith("Vivado v.2025.2")
    assert report.design == "system_wrapper"
    assert report.device == "xc7z020clg400-1"
    assert report.slice_luts == 1452
    assert report.slice_luts_available == 53200
    assert report.slice_luts_util_pct == 2.73
    assert report.slice_registers == 1453
    assert report.block_ram_tiles == 0
    assert report.dsps == 0
    assert report.dsps_available == 220


def test_build_power_thermal_model_from_vivado_reports_uses_report_values() -> None:
    model = build_power_thermal_model_from_vivado_reports(
        PYNQ_IMPL_DIR,
        PowerThermalConfig(
            target="zynq",
            layer_sizes=((700, 128), (128, 128), (128, 20)),
            bitstream_length=256,
            clock_mhz=100.0,
        ),
    )

    assert model["source_mode"] == "vivado_report_derived"
    assert model["power"]["total_power_mw"] == 1674.0
    assert model["power"]["reported_dynamic_power_mw"] == 1538.0
    assert model["power"]["reported_static_power_mw"] == 136.0
    assert model["thermal"]["theta_ja_c_per_w"] == 11.5
    assert model["thermal"]["reported_junction_c"] == 44.3
    assert model["implementation"]["device"] == "xc7z020clg400-1"
    assert model["implementation"]["utilisation"]["slice_luts"] == 1452
    assert model["implementation"]["utilisation"]["block_ram_tiles"] == 0
    assert model["implementation"]["utilisation"]["dsps"] == 0
    assert "Vivado power is report-derived" in model["limitations"][0]


def test_write_power_thermal_model_from_vivado_reports_writes_json(tmp_path: Path) -> None:
    destination = write_power_thermal_model_from_vivado_reports(
        PYNQ_IMPL_DIR,
        tmp_path,
        PowerThermalConfig(target="zynq", layer_sizes=((8, 4),), bitstream_length=64),
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert destination == tmp_path / "power_thermal_model.json"
    assert payload["source_mode"] == "vivado_report_derived"
    assert payload["source_reports"]["power"].endswith("system_wrapper_power_routed.rpt")


def test_vivado_field_extractors_fail_closed_on_malformed_reports() -> None:
    from sc_neurocore.edge.power_thermal import (
        _extract_header,
        _extract_summary_float,
        _extract_summary_text,
        _extract_utilisation_row,
    )

    with pytest.raises(ValueError, match="missing Vivado header field"):
        _extract_header("no matching line", "Total On-Chip Power (W)")
    with pytest.raises(ValueError, match="missing Vivado summary field"):
        _extract_summary_text("no matching line", "Confidence Level")
    with pytest.raises(ValueError, match="missing Vivado utilisation row"):
        _extract_utilisation_row("no matching line", "CLB LUTs")
    with pytest.raises(ValueError, match="is not numeric"):
        _extract_summary_float("| Dynamic (W) | not_a_number |", "Dynamic (W)")
