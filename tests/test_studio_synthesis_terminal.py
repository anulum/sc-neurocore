# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio selected-RTL synthesis and PnR terminal

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest

from sc_neurocore.studio.compile_traceability import build_model_compile_traceability
from sc_neurocore.studio.model_compile_configuration import (
    resolve_model_compile_configuration,
)
from sc_neurocore.studio.model_cosim import run_model_cosim
from sc_neurocore.studio.synthesis import run_synthesis_terminal

HAS_TERMINAL_TOOLS = all(
    shutil.which(tool) is not None for tool in ("gcc", "iverilog", "vvp", "yosys", "nextpnr-ecp5")
)


def _source_chain(verilog: str) -> tuple[dict[str, object], dict[str, object]]:
    compile_traceability = build_model_compile_traceability(
        model_name="ExampleNeuron",
        schema_name="example",
        schema_sha256="a" * 64,
        params={"gain": 2.0},
        dt=0.25,
        integrator="map",
        q_format="Q8.8",
        module_name="example_neuron",
        verilog=verilog,
    ).to_public_dict()
    cosim_parity: dict[str, object] = {
        "bit_exact": True,
        "configuration": {
            "dt": 0.25,
            "integrator": "map",
            "model_name": "ExampleNeuron",
            "q_format": "Q8.8",
            "schema_name": "example",
            "schema_sha256": "a" * 64,
        },
        "module_name": "example_neuron",
        "reference": {"trace_sha256": "b" * 64},
        "rtl": {
            "source_sha256": compile_traceability["output"]["rtl_sha256"],  # type: ignore[index]
            "trace_sha256": "b" * 64,
        },
        "schema_version": "studio.cosim-parity.v1",
        "status": "completed",
    }
    return compile_traceability, cosim_parity


def test_terminal_binds_netlist_and_routed_artifact_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sc_neurocore.studio.synthesis as synthesis

    verilog = "module example_neuron(input clk, output q); assign q = clk; endmodule"
    compile_traceability, cosim_parity = _source_chain(verilog)

    def fake_synthesis(
        _verilog: str,
        target: str,
        *,
        root: Path,
        process_limits: object,
        target_provenance: Mapping[str, object],
    ) -> tuple[dict[str, object], Path]:
        del process_limits
        netlist = root / "design.json"
        netlist.write_text('{"modules": {}}', encoding="utf-8")
        return (
            {
                "capacity": {},
                "log_excerpt": "yosys complete",
                "resources": {},
                "success": True,
                "target": target,
                "target_provenance": dict(target_provenance),
                "utilisation": {},
            },
            netlist,
        )

    def fake_pnr(json_path: str, target: str, **_kwargs: object) -> dict[str, object]:
        synthesis._pnr_output_path(Path(json_path), target).write_text(
            "routed design",
            encoding="utf-8",
        )
        return {
            "critical_path": "clk to q",
            "log_excerpt": "nextpnr complete",
            "max_freq_mhz": 42.5,
            "success": True,
        }

    monkeypatch.setattr(synthesis, "_run_synthesis_in_directory", fake_synthesis)
    monkeypatch.setattr(synthesis, "run_pnr", fake_pnr)
    monkeypatch.setattr(
        synthesis,
        "check_tools",
        lambda: {
            "nextpnr_ecp5": {"available": True, "version": "nextpnr test"},
            "yosys": {"available": True, "version": "yosys test"},
        },
    )

    execution = run_synthesis_terminal(
        verilog,
        "ecp5",
        compile_traceability=compile_traceability,
        cosim_parity=cosim_parity,
    )

    assert execution.report["schema_version"] == "studio.silicon-terminal.v1"
    assert execution.report["success"] is True
    assert execution.report["status"] == "completed"
    assert (
        execution.report["source_chain"]["rtl_sha256"]
        == (  # type: ignore[index]
            compile_traceability["output"]["rtl_sha256"]  # type: ignore[index]
        )
    )
    assert execution.report["artifacts"] == {  # type: ignore[index]
        "netlist_sha256": "ab6b54e2978d53a10c1db5983b4712ee25ed937b2da8b7e3ba43f7a184da2fdb",
        "routed_design_sha256": "466edbb65d32d109e3950417bd7ce667f390c483b528db79ff6427f6fbf0e009",
    }
    assert execution.netlist_json == b'{"modules": {}}'
    assert execution.routed_design == b"routed design"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("compile_rtl", "compile output digest"),
        ("compile_trace", "traceability digest"),
        ("cosim_rtl", "co-simulated source digest"),
        ("cosim_parity", "not bit-exact"),
        ("configuration", "field 'q_format'"),
        ("trace_digest", "evidence digest"),
    ],
)
def test_terminal_rejects_broken_selected_rtl_chain(mutation: str, message: str) -> None:
    verilog = "module example_neuron; endmodule"
    compile_traceability, cosim_parity = _source_chain(verilog)
    if mutation == "compile_rtl":
        compile_traceability["output"]["rtl_sha256"] = "0" * 64  # type: ignore[index]
    elif mutation == "compile_trace":
        compile_traceability["traceability_sha256"] = "0" * 64
    elif mutation == "cosim_rtl":
        cosim_parity["rtl"]["source_sha256"] = "0" * 64  # type: ignore[index]
    elif mutation == "cosim_parity":
        cosim_parity["bit_exact"] = False
    elif mutation == "configuration":
        cosim_parity["configuration"]["q_format"] = "Q16.16"  # type: ignore[index]
    else:
        cosim_parity["reference"]["trace_sha256"] = "z" * 64  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        run_synthesis_terminal(
            verilog,
            "ecp5",
            compile_traceability=compile_traceability,
            cosim_parity=cosim_parity,
        )


def test_terminal_rejects_target_without_pnr() -> None:
    verilog = "module example_neuron; endmodule"
    compile_traceability, cosim_parity = _source_chain(verilog)

    with pytest.raises(ValueError, match="no place-and-route terminal"):
        run_synthesis_terminal(
            verilog,
            "gowin",
            compile_traceability=compile_traceability,
            cosim_parity=cosim_parity,
        )


@pytest.mark.skipif(not HAS_TERMINAL_TOOLS, reason="selected-model EDA tools are required")
def test_real_ecp5_terminal_routes_selected_model_rtl() -> None:
    configuration = resolve_model_compile_configuration(
        {
            "integrator": "map",
            "model_name": "AdaptiveThresholdIFNeuron",
            "q_format": "Q8.8",
        }
    )
    verilog = configuration.to_verilog()
    compile_traceability = build_model_compile_traceability(
        model_name=configuration.model_name,
        schema_name=configuration.schema_name,
        schema_sha256=configuration.schema_sha256,
        params=configuration.params,
        dt=configuration.dt,
        integrator=configuration.integrator,
        q_format=configuration.q_format.q_label,
        module_name=configuration.module_name,
        verilog=verilog,
    ).to_public_dict()
    cosim_parity = run_model_cosim(configuration, current=10.0, n_steps=4).report

    execution = run_synthesis_terminal(
        verilog,
        "ecp5",
        compile_traceability=compile_traceability,
        cosim_parity=cosim_parity,
    )

    assert execution.report["success"] is True
    assert execution.report["place_and_route"]["max_freq_mhz"] > 0  # type: ignore[index]
    assert execution.netlist_json
    assert execution.routed_design
    assert b"design.v" in execution.netlist_json
    assert b"/tmp/" not in execution.netlist_json
