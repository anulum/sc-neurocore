# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
    return TestClient(create_app(), base_url="http://127.0.0.1")


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

    def test_synth_non_string_verilog_rejected(self, client):
        r = client.post("/api/synth/run", json={"verilog": {"module": "x"}, "target": "ice40"})
        assert r.status_code == 422

    def test_synth_oversized_verilog_rejected(self, client):
        huge = "module x;\n" + ("wire a;\n" * 400_000) + "endmodule\n"
        r = client.post("/api/synth/run", json={"verilog": huge, "target": "ice40"})
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

    def test_pnr_rejects_non_json_input(self, client, tmp_path):
        path = tmp_path / "design.txt"
        path.write_text("not a netlist", encoding="utf-8")
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": str(path), "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert ".json netlist" in data["error"]

    def test_pnr_rejects_directory_input(self, client, tmp_path):
        directory = tmp_path / "dir_as_input.json"
        directory.mkdir()
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": str(directory), "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "not a regular file" in data["error"]

    def test_pnr_rejects_non_json_payload(self, client, tmp_path):
        path = tmp_path / "invalid.json"
        path.write_text("this is not json", encoding="utf-8")
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": str(path), "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "valid UTF-8 JSON" in data["error"]

    def test_pnr_rejects_non_object_json_payload(self, client, tmp_path):
        path = tmp_path / "array_payload.json"
        path.write_text("[]", encoding="utf-8")
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": str(path), "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "must be an object" in data["error"]

    def test_pnr_rejects_symlink_input(self, client, tmp_path):
        target = tmp_path / "netlist.json"
        target.write_text("{}", encoding="utf-8")
        symlink = tmp_path / "netlist_link.json"
        symlink.symlink_to(target)
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": str(symlink), "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "must not be a symlink" in data["error"]


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

    def test_estimate_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            estimate_resources(5, "unknown")

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

    def test_estimate_endpoint_rejects_non_integer(self, client):
        r = client.post("/api/synth/estimate", json={"ir_op_count": "10", "target": "ice40"})
        assert r.status_code == 422
        assert "integer" in r.text

    def test_estimate_endpoint_rejects_unknown_target(self, client):
        r = client.post("/api/synth/estimate", json={"ir_op_count": 10, "target": "unknown"})
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

    def test_multi_target_rejects_non_string_verilog(self, client):
        r = client.post(
            "/api/synth/multi-target", json={"verilog": {"rtl": "module x();endmodule"}}
        )
        assert r.status_code == 422

    def test_multi_target_rejects_oversized_verilog(self, client):
        huge = "module x;\n" + ("wire a;\n" * 400_000) + "endmodule\n"
        r = client.post("/api/synth/multi-target", json={"verilog": huge})
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

    def test_parse_rejects_non_object_modules(self, tmp_path):
        data = {"modules": []}
        json_path = str(tmp_path / "bad_modules.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="'modules' must be an object"):
            _parse_yosys_json(json_path)

    def test_parse_rejects_non_object_cells(self, tmp_path):
        data = {"modules": {"top": {"cells": [], "netnames": {}}}}
        json_path = str(tmp_path / "bad_cells.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="cells' must be an object"):
            _parse_yosys_json(json_path)


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

    def test_non_string_verilog_raises(self):
        with pytest.raises(ValueError, match="verilog_source must be a string"):
            run_synthesis(123, "ice40")  # type: ignore[arg-type]

    def test_empty_verilog_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            run_synthesis("   ", "ice40")

    def test_returns_target_field(self):
        result = run_synthesis("module t(); endmodule", "ice40")
        assert result["target"] == "ice40"
