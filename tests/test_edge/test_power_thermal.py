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
from pathlib import Path

import pytest

from sc_neurocore.edge.power_thermal import (
    SCHEMA_VERSION,
    PowerThermalConfig,
    build_power_thermal_model,
    write_power_thermal_model,
)


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
