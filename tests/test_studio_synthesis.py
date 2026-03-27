# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Synthesis Dashboard (Block 3)

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.synthesis import (
    _DEVICE_CAPACITY,
    _TARGETS,
    _parse_yosys_json,
    check_tools,
    estimate_resources,
    multi_target_synthesis,
    run_synthesis,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


# --- Tool Detection ---


class TestToolDetection:
    def test_check_tools_returns_dict(self):
        result = check_tools()
        assert "yosys" in result
        assert "nextpnr_ice40" in result
        for tool_info in result.values():
            assert "available" in tool_info
            assert "version" in tool_info

    def test_check_tools_has_all_expected(self):
        result = check_tools()
        expected = {"yosys", "nextpnr_ice40", "nextpnr_ecp5", "firtool"}
        assert set(result.keys()) == expected

    def test_tools_status_endpoint(self, client):
        r = client.get("/api/synth/tools-status")
        assert r.status_code == 200
        data = r.json()
        assert "yosys" in data
        assert isinstance(data["yosys"]["available"], bool)


# --- Synthesis Endpoint ---


class TestSynthesisEndpoint:
    def test_synth_requires_verilog(self, client):
        r = client.post("/api/synth/run", json={"target": "ice40"})
        assert r.status_code == 422

    def test_synth_empty_verilog_rejected(self, client):
        r = client.post("/api/synth/run", json={"verilog": "", "target": "ice40"})
        assert r.status_code == 422

    def test_synth_with_stub_verilog(self, client):
        verilog = "module test(); endmodule"
        r = client.post("/api/synth/run", json={"verilog": verilog, "target": "ice40"})
        assert r.status_code == 200
        data = r.json()
        assert "success" in data
        assert "target" in data
        assert data["target"] == "ice40"

    def test_synth_invalid_target(self, client):
        verilog = "module test(); endmodule"
        r = client.post("/api/synth/run", json={"verilog": verilog, "target": "invalid"})
        assert r.status_code == 422

    def test_synth_all_valid_targets(self, client):
        verilog = "module test(); endmodule"
        for target in _TARGETS:
            r = client.post("/api/synth/run", json={"verilog": verilog, "target": target})
            assert r.status_code == 200
            data = r.json()
            assert data["target"] == target


# --- PnR Endpoint ---


class TestPnREndpoint:
    def test_pnr_requires_json_path(self, client):
        r = client.post("/api/synth/pnr", json={"target": "ice40"})
        assert r.status_code == 422

    def test_pnr_empty_path_rejected(self, client):
        r = client.post("/api/synth/pnr", json={"json_path": "", "target": "ice40"})
        assert r.status_code == 422

    def test_pnr_nonexistent_path(self, client):
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": "/tmp/does_not_exist.json", "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False


# --- Resource Estimation ---


class TestEstimation:
    def test_estimate_returns_structure(self):
        result = estimate_resources(10, "ice40")
        assert result["target"] == "ice40"
        assert result["estimated"] is True
        assert "resources" in result
        assert "capacity" in result
        assert "utilisation" in result

    def test_estimate_all_targets(self):
        for target in _TARGETS:
            result = estimate_resources(5, target)
            assert result["target"] == target
            for k in ["luts", "ffs", "brams", "dsps"]:
                assert k in result["utilisation"]

    def test_estimate_scales_with_ops(self):
        small = estimate_resources(5, "ice40")
        large = estimate_resources(50, "ice40")
        assert large["resources"]["luts"] > small["resources"]["luts"]
        assert large["resources"]["ffs"] > small["resources"]["ffs"]

    def test_estimate_endpoint(self, client):
        r = client.post(
            "/api/synth/estimate",
            json={"ir_op_count": 10, "target": "ecp5"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["target"] == "ecp5"
        assert data["estimated"] is True

    def test_estimate_endpoint_rejects_zero(self, client):
        r = client.post("/api/synth/estimate", json={"ir_op_count": 0, "target": "ice40"})
        assert r.status_code == 422


# --- Multi-Target Synthesis ---


class TestMultiTarget:
    def test_multi_target_returns_all(self):
        verilog = "module test(); endmodule"
        result = multi_target_synthesis(verilog)
        assert "targets" in result
        assert "supported" in result
        assert set(result["supported"]) == set(_TARGETS.keys())
        for target in _TARGETS:
            assert target in result["targets"]

    def test_multi_target_endpoint(self, client):
        verilog = "module test(); endmodule"
        r = client.post("/api/synth/multi-target", json={"verilog": verilog})
        assert r.status_code == 200
        data = r.json()
        assert "targets" in data
        assert "supported" in data

    def test_multi_target_requires_verilog(self, client):
        r = client.post("/api/synth/multi-target", json={})
        assert r.status_code == 422


# --- Yosys JSON Parser ---


class TestYosysJsonParser:
    def test_parse_empty_design(self, tmp_path):
        data = {"modules": {}}
        json_path = str(tmp_path / "empty.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        result = _parse_yosys_json(json_path)
        assert result["luts"] == 0
        assert result["ffs"] == 0
        assert result["cells"] == 0

    def test_parse_with_luts_and_ffs(self, tmp_path):
        data = {
            "modules": {
                "top": {
                    "cells": {
                        "c0": {"type": "SB_LUT4"},
                        "c1": {"type": "SB_LUT4"},
                        "c2": {"type": "SB_DFF"},
                        "c3": {"type": "DSP48"},
                    },
                    "netnames": {"n0": {}, "n1": {}, "n2": {}},
                }
            }
        }
        json_path = str(tmp_path / "design.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        result = _parse_yosys_json(json_path)
        assert result["luts"] == 2
        assert result["ffs"] == 1
        assert result["dsps"] == 1
        assert result["cells"] == 4
        assert result["wires"] == 3

    def test_parse_bram_detection(self, tmp_path):
        data = {
            "modules": {
                "mem": {
                    "cells": {
                        "r0": {"type": "SB_RAM256x16"},
                        "r1": {"type": "BRAM_TDP36"},
                    },
                    "netnames": {},
                }
            }
        }
        json_path = str(tmp_path / "mem.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        result = _parse_yosys_json(json_path)
        assert result["brams"] == 2

    def test_parse_multi_module(self, tmp_path):
        data = {
            "modules": {
                "a": {"cells": {"c0": {"type": "LUT4"}}, "netnames": {"n": {}}},
                "b": {"cells": {"c0": {"type": "DFF"}}, "netnames": {"n": {}}},
            }
        }
        json_path = str(tmp_path / "multi.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        result = _parse_yosys_json(json_path)
        assert result["luts"] == 1
        assert result["ffs"] == 1
        assert result["cells"] == 2
        assert result["wires"] == 2


# --- Device Capacity ---


class TestDeviceCapacity:
    def test_all_targets_have_capacity(self):
        for target in _TARGETS:
            assert target in _DEVICE_CAPACITY
            cap = _DEVICE_CAPACITY[target]
            assert cap["luts"] > 0
            assert cap["ffs"] > 0

    def test_capacity_values_sane(self):
        for target, cap in _DEVICE_CAPACITY.items():
            assert cap["luts"] <= 100_000
            assert cap["ffs"] <= 100_000
            assert cap["brams"] <= 500
            assert cap["dsps"] <= 500


# --- run_synthesis unit ---


class TestRunSynthesis:
    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            run_synthesis("module t(); endmodule", "nonexistent")

    def test_returns_target_field(self):
        result = run_synthesis("module t(); endmodule", "ice40")
        assert result["target"] == "ice40"
