# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the per-model silicon capability matrix

"""Every operation the matrix enables runs; every one it disables says why.

The enabled co-simulation combinations of a map model and an adaptive Euler
model are executed through Icarus Verilog and a C compiler and must be
bit-exact; an RK model compiles and has co-simulation disabled by name.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.model_capabilities import (
    FORMAL_INVENTORY,
    MODEL_CAPABILITIES_SCHEMA_VERSION,
    model_capabilities,
)
from sc_neurocore.studio.model_compile_configuration import resolve_model_compile_configuration
from sc_neurocore.studio.model_cosim import run_model_cosim
from sc_neurocore.studio.synthesis import check_tools

_COSIM_TOOLS = all(shutil.which(tool) for tool in ("iverilog", "vvp", "gcc"))
needs_cosim_tools = pytest.mark.skipif(
    not _COSIM_TOOLS, reason="iverilog, vvp and gcc are required"
)


def _operations(name: str, **kwargs: object) -> dict:
    capabilities = model_capabilities(name, **kwargs)
    assert capabilities is not None
    assert capabilities["schema_version"] == MODEL_CAPABILITIES_SCHEMA_VERSION
    assert capabilities["model"] == name
    return capabilities["operations"]


class TestMatrix:
    @needs_cosim_tools
    def test_a_map_model_offers_the_chain_this_host_can_run(self) -> None:
        tools = check_tools()
        operations = _operations("AdaptiveThresholdIFNeuron", tool_status=tools)

        assert operations["compile"] == {
            "enabled": True,
            "reason": None,
            "integrators": ["map"],
            "q_formats": ["Q8.8", "Q16.16"],
        }
        cosim = operations["cosimulate"]
        assert cosim["enabled"] is True
        assert cosim["combinations"] == [
            {"integrator": "map", "q_format": "Q8.8", "mirrored": True},
            {"integrator": "map", "q_format": "Q16.16", "mirrored": True},
        ]
        assert operations["synthesise"]["enabled"] is bool(tools["yosys"]["available"])
        pnr = operations["place_and_route"]
        if operations["synthesise"]["enabled"]:
            assert pnr["disabled_targets"]["gowin"] == (
                "this target has no place-and-route tool in the Studio flow"
            )
            assert "xilinx" in pnr["disabled_targets"]
            for target in ("ice40", "ecp5"):
                installed = bool(tools[f"nextpnr_{target}"]["available"])
                assert (target in pnr["targets"]) is installed

    def test_an_rk_model_compiles_and_names_why_it_cannot_be_co_simulated(self) -> None:
        operations = _operations("SCScaledResetAdaptiveIFNeuron")

        assert operations["compile"]["enabled"] is True
        assert operations["compile"]["integrators"] == ["rk4"]
        assert operations["cosimulate"] == {
            "enabled": False,
            "reason": "no bit-true C kernel mirrors this model's RTL for any offered integrator",
            "unmirrored_integrators": ["rk4"],
        }
        assert operations["synthesise"]["reason"] == (
            "synthesis runs only on RTL whose co-simulation was bit-exact"
        )
        assert operations["place_and_route"]["reason"] == "place and route follows synthesis"

    def test_a_model_without_a_compilable_schema_disables_everything(self) -> None:
        operations = _operations("ATypeKNeuron")
        assert operations["compile"]["reason"] == (
            "the model has no canonical schema with an executable profile for the compiler"
        )
        assert operations["cosimulate"] == {
            "enabled": False,
            "reason": "the model does not compile",
        }

    def test_a_model_no_format_holds_is_not_compiled(self) -> None:
        compile_op = _operations("SCClippedRationalRecoveryMapNeuron")["compile"]
        assert compile_op["enabled"] is False
        assert compile_op["reason"].startswith("no Studio fixed-point format holds the model")
        assert compile_op["integrators"] == ["map"]

    def test_without_simulators_co_simulation_names_the_missing_tools(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty PATH is an installation without Icarus Verilog or a C compiler."""
        monkeypatch.setenv("PATH", str(tmp_path))
        operations = _operations("AdaptiveThresholdIFNeuron", tool_status={})
        assert operations["cosimulate"]["reason"] == (
            "co-simulation needs iverilog, vvp, gcc, which is not installed"
        )
        assert operations["synthesise"]["enabled"] is False

    @needs_cosim_tools
    def test_synthesis_and_place_and_route_follow_the_installed_tools(self) -> None:
        without_yosys = _operations("AdaptiveThresholdIFNeuron", tool_status={})
        assert without_yosys["synthesise"]["reason"] == "Yosys is not installed"

        yosys_only = _operations(
            "AdaptiveThresholdIFNeuron", tool_status={"yosys": {"available": True}}
        )
        assert yosys_only["synthesise"]["targets"] == ["ice40", "ecp5", "gowin", "xilinx"]
        assert yosys_only["place_and_route"]["enabled"] is False
        assert yosys_only["place_and_route"]["reason"] == (
            "no target's place-and-route tool is installed"
        )
        assert yosys_only["place_and_route"]["disabled_targets"]["ice40"] == (
            "nextpnr-ice40 is not installed"
        )

    def test_the_formal_job_is_named_though_the_studio_cannot_run_it(self, tmp_path: Path) -> None:
        formal = _operations("AdaptiveThresholdIFNeuron")["formal"]
        jobs = json.loads(FORMAL_INVENTORY.read_text(encoding="utf-8"))["jobs"]
        (job,) = [entry for entry in jobs if entry["class"] == "AdaptiveThresholdIFNeuron"]
        assert formal["enabled"] is False
        assert formal["reason"].startswith("the Studio has no route that runs a formal job")
        assert formal["catalogue_job"]["module"] == job["module"]
        assert formal["catalogue_job"]["not_established"] == job["not_established"]

        assert _operations("ATypeKNeuron")["formal"]["catalogue_job"] is None
        absent = _operations("AdaptiveThresholdIFNeuron", formal_inventory=tmp_path / "none.json")
        assert absent["formal"]["catalogue_job"] is None

    def test_a_name_the_catalogue_does_not_hold_has_no_matrix(self) -> None:
        assert model_capabilities("NoSuchNeuron") is None


@needs_cosim_tools
@pytest.mark.parametrize("model", ["AdaptiveThresholdIFNeuron", "AdExNeuron"])
def test_every_enabled_co_simulation_combination_is_bit_exact(model: str) -> None:
    """The matrix enables only what executes: run each combination for real."""
    combinations = _operations(model)["cosimulate"]["combinations"]
    ran = 0
    for combination in combinations:
        if not combination["mirrored"]:
            continue
        configuration = resolve_model_compile_configuration(
            {
                "model_name": model,
                "integrator": combination["integrator"],
                "q_format": combination["q_format"],
            }
        )
        execution = run_model_cosim(configuration, current=1.0, n_steps=24)
        assert execution.report["bit_exact"] is True, combination
        ran += 1
    assert ran >= 1


class TestRoute:
    @pytest.fixture(scope="class")
    def client(self) -> TestClient:
        return TestClient(create_app(), base_url="http://127.0.0.1")

    def test_the_route_serves_the_matrix(self, client: TestClient) -> None:
        response = client.get("/api/models/AdaptiveThresholdIFNeuron/capabilities")
        assert response.status_code == 200
        assert response.json()["schema_version"] == MODEL_CAPABILITIES_SCHEMA_VERSION

    def test_the_route_refuses_an_unknown_model(self, client: TestClient) -> None:
        response = client.get("/api/models/NoSuchNeuron/capabilities")
        assert response.status_code == 404
