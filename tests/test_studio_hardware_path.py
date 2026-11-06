# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the one-command hardware path of a catalogue model

"""Run real catalogue models through the hardware path and reach every refusal.

Completed paths use Icarus Verilog, a C compiler, Yosys and, where installed,
nextpnr. A host without them is produced the way the capability tests do it:
an empty or restricted ``PATH``. A co-simulation that is not bit-exact cannot
be produced from a real model (every combination the matrix enables agrees),
so the stop logic is fed a real report whose traces were made to differ.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.compiler.guarantee_summary import GUARANTEE_CLASSES
from sc_neurocore.studio.hardware_path import (
    DEFAULT_CURRENT,
    DEFAULT_N_STEPS,
    DEFAULT_TARGET,
    PLACE_AND_ROUTE_TARGETS,
    HardwarePathRefusal,
    _check_combination,
    formal_guarantee,
    gate_refusal,
    model_guarantee_summary,
    readiness_labels,
    render_model_guarantee_text,
    require_bit_exact,
    run_hardware_path,
    supported_models,
)
from sc_neurocore.studio.model_capabilities import FORMAL_INVENTORY, model_capabilities
from sc_neurocore.studio.model_catalogue import get_model_detail
from sc_neurocore.studio.model_compile_configuration import (
    ResolvedModelCompileConfiguration,
    resolve_model_compile_configuration,
)
from sc_neurocore.studio.model_cosim import _first_mismatch, run_model_cosim
from sc_neurocore.studio.readiness_seal import SEAL_PATH, sealed_detail

COSIM_TOOLS = ("gcc", "iverilog", "vvp")
HAS_COSIM = all(shutil.which(tool) for tool in COSIM_TOOLS)
HAS_SYNTHESIS = HAS_COSIM and shutil.which("yosys") is not None
needs_synthesis = pytest.mark.skipif(
    not HAS_SYNTHESIS, reason="gcc, iverilog, vvp and yosys are required"
)
needs_cosim = pytest.mark.skipif(not HAS_COSIM, reason="gcc, iverilog and vvp are required")
needs_ecp5 = pytest.mark.skipif(
    not (HAS_SYNTHESIS and shutil.which("nextpnr-ecp5")), reason="nextpnr-ecp5 is required"
)
needs_ice40 = pytest.mark.skipif(
    not (HAS_SYNTHESIS and shutil.which("nextpnr-ice40")), reason="nextpnr-ice40 is required"
)


def _inventory_job(model: str) -> dict[str, Any]:
    document = json.loads(FORMAL_INVENTORY.read_text(encoding="utf-8"))
    jobs = [*document["jobs"], *document["retained_jobs"]]
    (job,) = [entry for entry in jobs if entry["class"] == model]
    return dict(job)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_artifacts_bound(manifest: Mapping[str, Any], out_dir: Path) -> None:
    """Every listed artefact exists with its digest, and every written file is listed."""
    listed = manifest["artifacts"]
    for relative, entry in listed.items():
        path = out_dir / relative
        assert _sha256(path) == entry["sha256"], relative
        assert path.stat().st_size == entry["bytes"]
    written = {
        str(path.relative_to(out_dir))
        for path in out_dir.rglob("*")
        if path.is_file() and path.name not in {"hardware_path.json", "hardware_path.txt"}
    }
    assert written == set(listed)


def _restricted_path(tmp_path: Path, tools: tuple[str, ...]) -> str:
    """Return a PATH that holds only the named installed tools."""
    directory = tmp_path / "bin"
    directory.mkdir()
    for tool in tools:
        found = shutil.which(tool)
        assert found is not None, tool
        os.symlink(found, directory / tool)
    return str(directory)


def _refusal(model: str, out_dir: Path, **kwargs: Any) -> HardwarePathRefusal:
    with pytest.raises(HardwarePathRefusal) as raised:
        run_hardware_path(model, out_dir=out_dir, **kwargs)
    return raised.value


def _manifest_of(refusal: HardwarePathRefusal) -> dict[str, Any]:
    assert refusal.manifest_path is not None
    manifest: dict[str, Any] = json.loads(refusal.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "refused"
    assert manifest["refusal"] == refusal.to_public_dict()
    text = (refusal.manifest_path.parent / "hardware_path.txt").read_text(encoding="utf-8")
    assert f"REFUSED ({refusal.code})" in text
    return manifest


class TestCompletedPaths:
    """Real models through every stage."""

    @needs_ecp5
    def test_lapicque_runs_through_synthesis_and_place_and_route(self, tmp_path: Path) -> None:
        """Lapicque at Q8.8 with map is co-simulated bit-exactly and routed for ECP5.

        Q8.8 keeps the netlist within the terminal's 16 MiB place-and-route input
        limit with every Yosys build the Studio meets (0.33 and 0.70 were run); at
        Q16.16, Yosys 0.33 exceeds it and the path refuses place and route.
        """
        out = tmp_path / "lapicque"
        result = run_hardware_path(
            "LapicqueNeuron", out_dir=out, q_format="Q8.8", integrator="map", n_steps=24
        )
        manifest = result.manifest
        assert manifest["status"] == "completed"
        assert manifest["refusal"] is None
        assert [stage["stage"] for stage in manifest["stages"]] == [
            "capabilities",
            "configuration",
            "cosimulation",
            "guarantees",
            "synthesis",
            "place_and_route",
        ]
        assert json.loads(result.manifest_path.read_text(encoding="utf-8")) == manifest
        _assert_artifacts_bound(manifest, out)

        configuration = manifest["stages"][1]
        rtl = out / configuration["rtl"]
        assert configuration["rtl"] == "rtl/sc_lapicque_neuron.v"
        trace = json.loads((out / "rtl/compile_traceability.json").read_text(encoding="utf-8"))
        report = json.loads((out / "cosim/report.json").read_text(encoding="utf-8"))
        assert _sha256(rtl) == trace["output"]["rtl_sha256"] == report["rtl"]["source_sha256"]
        assert report["bit_exact"] is True
        assert report["configuration"]["q_format"] == "Q8.8"
        assert report["stimulus"] == {
            "current": DEFAULT_CURRENT,
            "current_q": 2560,
            "n_steps": 24,
        }

        summary = json.loads((out / "guarantees.json").read_text(encoding="utf-8"))
        cosim, formal = summary["guarantees"]
        assert cosim["class"] == "CO-SIMULATED ONLY"
        assert cosim["not_covered"] == report["boundary"]["not_covered"]
        assert cosim["statement"] == report["boundary"]["statement"]
        job = _inventory_job("LapicqueNeuron")
        assert formal["class"] == "PROVEN AT OTHER PARAMETERS"
        assert formal["proved_parameters"] == {
            "module": job["module"],
            "profile": job["profile"],
            "q_format": job["q_format"],
        }
        assert formal["run_parameters"]["q_format"] == "Q8.8"
        assert formal["differs"] == ["module", "q_format"]
        assert formal["ran_here"] is False
        assert summary["classes"] == {
            name: int(name in {"CO-SIMULATED ONLY", "PROVEN AT OTHER PARAMETERS"})
            for name in GUARANTEE_CLASSES
        }
        sealed = sealed_detail("LapicqueNeuron", SEAL_PATH)
        assert summary["readiness"]["source"] == "sealed"
        assert summary["readiness"]["science_label"] == sealed["science_label"]
        assert summary["readiness"]["silicon_label"] == sealed["silicon_label"]

        synthesis, pnr = manifest["stages"][4], manifest["stages"][5]
        terminal = json.loads((out / "synthesis/report.json").read_text(encoding="utf-8"))
        assert synthesis["report_sha256"] == _sha256(out / "synthesis/report.json")
        assert synthesis["netlist_sha256"] == terminal["artifacts"]["netlist_sha256"]
        assert synthesis["routed_design_sha256"] == _sha256(out / "synthesis/routed.config")
        assert terminal["source_chain"]["rtl_sha256"] == _sha256(rtl)
        assert pnr["status"] == "completed" and pnr["max_freq_mhz"] > 0
        assert set(manifest["tools"]["cosimulation"]) == set(COSIM_TOOLS)
        assert manifest["tools"]["eda"]["yosys"]

        boundary = manifest["boundary"]
        assert any("not the float model" in item for item in boundary["does_not_establish"])
        assert any("no device was programmed" in item for item in boundary["does_not_establish"])
        assert any("tools' estimates" in item for item in boundary["establishes"])
        assert "COMPLETED" in result.summary_path.read_text(encoding="utf-8")

    @needs_synthesis
    def test_a_target_without_place_and_route_is_synthesised_only(self, tmp_path: Path) -> None:
        """AdaptiveThresholdIF on Gowin is synthesised and place and route is recorded not run."""
        out = tmp_path / "adaptive"
        result = run_hardware_path(
            "AdaptiveThresholdIFNeuron",
            out_dir=out,
            q_format="Q16.16",
            integrator="map",
            n_steps=16,
            target="gowin",
        )
        stages = {stage["stage"]: stage for stage in result.manifest["stages"]}
        assert stages["capabilities"]["place_and_route"] == {
            "runs": False,
            "reason": "this target has no place-and-route tool in the Studio flow",
        }
        assert stages["place_and_route"] == {
            "stage": "place_and_route",
            "status": "not run",
            "target": "gowin",
            "reason": "this target has no place-and-route tool in the Studio flow",
        }
        assert stages["synthesis"]["resources"]["luts"] > 0
        assert not (out / "synthesis/netlist.json").exists()
        _assert_artifacts_bound(result.manifest, out)
        assert any(
            "place and route or timing for gowin: not run" in item
            for item in result.manifest["boundary"]["does_not_establish"]
        )
        text = result.summary_path.read_text(encoding="utf-8")
        assert "place_and_route: not run: this target has no place-and-route tool" in text

    @needs_synthesis
    def test_a_model_whose_rtl_is_its_formal_jobs_rtl_is_formal(self, tmp_path: Path) -> None:
        """Izhikevich's RTL under its job's module name is the job's RTL: FORMAL.

        The target is synthesis-only, so the classification does not depend on the
        host's place-and-route build.
        """
        job = _inventory_job("Izhikevich2007Neuron")
        out = tmp_path / "izhikevich"
        result = run_hardware_path(
            "Izhikevich2007Neuron",
            out_dir=out,
            q_format=job["q_format"],
            n_steps=16,
            module_name=job["module"],
            target="gowin",
        )
        summary = json.loads((out / "guarantees.json").read_text(encoding="utf-8"))
        formal = summary["guarantees"][1]
        assert formal["class"] == "FORMAL"
        assert "differs" not in formal
        assert formal["job_rtl_sha256"] == _sha256(FORMAL_INVENTORY.parent / f"{job['module']}.v")
        assert summary["classes"]["FORMAL"] == 1
        assert any(
            "byte-identical to the RTL of catalogue formal job" in item
            for item in result.manifest["boundary"]["establishes"]
        )
        assert result.manifest["stages"][-2]["stage"] == "synthesis"
        assert result.manifest["stages"][-1]["status"] == "not run"
        assert "formal: FORMAL" in (out / "guarantees.txt").read_text(encoding="utf-8")

    @needs_ice40
    def test_a_design_the_device_cannot_place_is_refused_after_synthesis(
        self, tmp_path: Path
    ) -> None:
        """Izhikevich's ports exceed the iCE40 UP5K package: nextpnr fails and the path refuses."""
        out = tmp_path / "ice40"
        refusal = _refusal("Izhikevich2007Neuron", out, q_format="Q8.8", n_steps=8, target="ice40")
        assert (refusal.stage, refusal.code) == ("place_and_route", "place-and-route-failed")
        assert "for ice40" in refusal.message
        manifest = _manifest_of(refusal)
        assert [stage["stage"] for stage in manifest["stages"]][-1] == "synthesis"
        terminal = json.loads((out / "synthesis/report.json").read_text(encoding="utf-8"))
        assert terminal["synthesis"]["success"] is True
        assert terminal["place_and_route"]["success"] is False
        assert "synthesis/routed.asc" not in manifest["artifacts"]

    @needs_synthesis
    def test_defaults_are_the_models_and_the_studios(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without format or integrator the model's defaults run; P&R absent is recorded."""
        tools = ("gcc", "iverilog", "vvp", "yosys", "as", "ld")
        monkeypatch.setenv("PATH", _restricted_path(tmp_path, tools))
        out = tmp_path / "defaults"
        out.mkdir()
        result = run_hardware_path("ResonateAndFireNeuron", out_dir=out, n_steps=8)
        capabilities = result.manifest["stages"][0]
        detail = get_model_detail("ResonateAndFireNeuron")
        assert detail is not None
        choices = detail["compile_configuration"]
        assert capabilities["q_format"] == choices["default_q_format"]
        assert capabilities["integrator"] == choices["default_integrator"]
        assert result.manifest["request"]["target"] == DEFAULT_TARGET
        assert capabilities["place_and_route"] == {
            "runs": False,
            "reason": "nextpnr-ecp5 is not installed",
        }
        assert result.manifest["stages"][-1]["status"] == "not run"


def test_the_defaults_are_the_studio_request_defaults() -> None:
    """Current, steps and target equal the Studio HTTP request defaults."""
    pytest.importorskip("fastapi", reason="the Studio request schemas need fastapi")
    from sc_neurocore.studio.api.schemas import ModelCosimRequest, SynthesisTerminalRequest

    assert ModelCosimRequest.model_fields["current"].default == DEFAULT_CURRENT
    assert ModelCosimRequest.model_fields["n_steps"].default == DEFAULT_N_STEPS
    assert SynthesisTerminalRequest.model_fields["target"].default == DEFAULT_TARGET
    assert PLACE_AND_ROUTE_TARGETS == ("ice40", "ecp5")


class TestRefusals:
    """Every stage refuses by name and records the refusal."""

    def test_output_that_holds_files_or_is_a_file_is_refused_before_writing(
        self, tmp_path: Path
    ) -> None:
        """A non-empty directory and a file are refused and left untouched."""
        (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
        refusal = _refusal("LapicqueNeuron", tmp_path)
        assert (refusal.stage, refusal.code) == ("output", "output-not-empty")
        assert refusal.manifest_path is None
        assert sorted(path.name for path in tmp_path.iterdir()) == ["keep.txt"]

        refusal = _refusal("LapicqueNeuron", tmp_path / "keep.txt")
        assert (refusal.stage, refusal.code) == ("output", "output-not-directory")

        empty = tmp_path / "empty"
        empty.mkdir()
        (tmp_path / "linked").symlink_to(empty, target_is_directory=True)
        refusal = _refusal("LapicqueNeuron", tmp_path / "linked")
        assert (refusal.stage, refusal.code) == ("output", "output-is-symlink")
        assert list(empty.iterdir()) == []

    def test_an_unknown_model_is_refused(self, tmp_path: Path) -> None:
        """A name the catalogue does not hold is refused at the capability gate."""
        refusal = _refusal("NoSuchNeuron", tmp_path / "out")
        assert (refusal.stage, refusal.code) == ("capabilities", "unknown-model")
        assert "'NoSuchNeuron' is not a catalogue model" in refusal.message
        manifest = _manifest_of(refusal)
        assert manifest["stages"] == [] and manifest["artifacts"] == {}
        assert any(
            "refused there (unknown-model)" in item
            for item in manifest["boundary"]["does_not_establish"]
        )

    def test_a_model_without_an_executable_schema_is_refused(self, tmp_path: Path) -> None:
        """ATypeK has no compilable schema: compile is refused with the matrix's reason."""
        refusal = _refusal("ATypeKNeuron", tmp_path / "out")
        assert (refusal.stage, refusal.code) == ("capabilities", "compile-refused")
        assert "no canonical schema with an executable profile" in refusal.message
        _manifest_of(refusal)

    def test_a_model_no_kernel_mirrors_is_refused(self, tmp_path: Path) -> None:
        """An RK4-only model compiles but cannot be co-simulated, so the path refuses."""
        refusal = _refusal("SCScaledResetAdaptiveIFNeuron", tmp_path / "out")
        assert (refusal.stage, refusal.code) == ("capabilities", "cosimulate-refused")
        assert "no bit-true C kernel mirrors this model's RTL" in refusal.message

    @needs_synthesis
    def test_an_undeclared_integrator_is_refused(self, tmp_path: Path) -> None:
        """An integrator outside the model's family is refused, naming the declared ones."""
        refusal = _refusal("LapicqueNeuron", tmp_path / "out", integrator="rk4")
        assert (refusal.stage, refusal.code) == ("capabilities", "integrator-not-declared")
        assert "'rk4' is not declared for LapicqueNeuron; declared: map" in refusal.message

    @needs_synthesis
    def test_an_unknown_target_is_refused(self, tmp_path: Path) -> None:
        """A synthesis target the Studio does not register is refused by name."""
        refusal = _refusal("LapicqueNeuron", tmp_path / "out", target="artix")
        assert (refusal.stage, refusal.code) == ("capabilities", "unknown-target")
        assert "'artix'" in refusal.message

    @needs_synthesis
    def test_a_format_that_cannot_hold_the_model_is_refused(self, tmp_path: Path) -> None:
        """Alpha does not fit Q8.8 (the contract says why); Q4.4 is no Studio format."""
        refusal = _refusal("AlphaNeuron", tmp_path / "a", q_format="Q8.8")
        assert (refusal.stage, refusal.code) == ("capabilities", "q-format-not-offered")
        assert "offered: Q16.16; Q8.8 cannot hold this neuron" in refusal.message

        refusal = _refusal("AlphaNeuron", tmp_path / "b", q_format="Q4.4")
        assert refusal.code == "q-format-not-offered"
        assert refusal.message.endswith("Studio compiles at Q8.8, Q16.16")

    @needs_synthesis
    def test_a_configuration_the_resolver_refuses_is_refused(self, tmp_path: Path) -> None:
        """An unknown parameter override is refused by the configuration stage."""
        refusal = _refusal(
            "LapicqueNeuron", tmp_path / "out", q_format="Q16.16", params={"no_such": 1.0}
        )
        assert (refusal.stage, refusal.code) == ("configuration", "configuration-refused")
        assert "Unknown schema parameter override(s): no_such" in refusal.message
        assert [stage["stage"] for stage in _manifest_of(refusal)["stages"]] == ["capabilities"]

    @needs_synthesis
    def test_a_current_outside_the_format_is_refused_by_the_cosimulation(
        self, tmp_path: Path
    ) -> None:
        """A current whose input word would wrap is refused after the RTL is written."""
        refusal = _refusal("LapicqueNeuron", tmp_path / "out", q_format="Q8.8", current=1000.0)
        assert (refusal.stage, refusal.code) == ("cosimulation", "cosim-refused")
        assert "outside Q8.8" in refusal.message
        manifest = _manifest_of(refusal)
        assert "rtl/sc_lapicque_neuron.v" in manifest["artifacts"]
        assert "cosim/report.json" not in manifest["artifacts"]

    def test_without_simulators_the_gate_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty PATH is a host without Icarus Verilog or a C compiler."""
        monkeypatch.setenv("PATH", str(tmp_path))
        refusal = _refusal("LapicqueNeuron", tmp_path / "out")
        assert (refusal.stage, refusal.code) == ("capabilities", "cosimulate-refused")
        assert "co-simulation needs iverilog, vvp, gcc, which is not installed" in refusal.message
        assert _manifest_of(refusal)["tools"]["eda"]["yosys"] is None

    @needs_cosim
    def test_without_yosys_the_gate_refuses_synthesis(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the simulators but no Yosys, synthesis is unavailable and the path refuses."""
        monkeypatch.setenv("PATH", _restricted_path(tmp_path, COSIM_TOOLS))
        refusal = _refusal("LapicqueNeuron", tmp_path / "out")
        assert (refusal.stage, refusal.code) == ("capabilities", "synthesise-refused")
        assert refusal.message.endswith("synthesise refused: Yosys is not installed")

    @needs_synthesis
    def test_a_c_compiler_that_cannot_assemble_fails_the_cosimulation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A C compiler without the assembler on PATH fails to build the reference kernel."""
        tools = ("gcc", "iverilog", "vvp", "yosys")
        monkeypatch.setenv("PATH", _restricted_path(tmp_path, tools))
        refusal = _refusal("LapicqueNeuron", tmp_path / "out", n_steps=4)
        assert (refusal.stage, refusal.code) == ("cosimulation", "cosim-failed")
        assert "Native command 'gcc' exited 1" in refusal.message
        assert "cannot execute" in refusal.message


def test_combinations_the_matrix_does_not_mirror_are_refused() -> None:
    """An undeclared, unmirrored or unmatched integrator and format pair is refused by name.

    No catalogue model offers a second integrator or an unmirrored pair today,
    so a real matrix is given one.
    """
    capabilities = model_capabilities("LapicqueNeuron", tool_status={"yosys": {"available": True}})
    assert capabilities is not None
    operations = copy.deepcopy(capabilities["operations"])
    detail = get_model_detail("LapicqueNeuron")
    assert detail is not None
    choices = detail["compile_configuration"]

    operations["compile"]["integrators"].append("euler")
    with pytest.raises(HardwarePathRefusal) as raised:
        _check_combination("LapicqueNeuron", operations, choices, "euler", "Q8.8")
    assert raised.value.code == "integrator-not-cosimulated"
    assert "co-simulated integrators: map" in raised.value.message

    operations["cosimulate"]["combinations"][0]["mirrored"] = False
    with pytest.raises(HardwarePathRefusal) as raised:
        _check_combination("LapicqueNeuron", operations, choices, "map", "Q8.8")
    assert raised.value.code == "combination-not-mirrored"
    _check_combination("LapicqueNeuron", operations, choices, "map", "Q16.16")


def test_the_gate_names_the_first_disabled_required_operation() -> None:
    """Compile, co-simulation and synthesis are required in that order."""
    enabled = {"enabled": True, "reason": None}
    assert gate_refusal({"compile": enabled, "cosimulate": enabled, "synthesise": enabled}) is None
    assert gate_refusal(
        {
            "compile": enabled,
            "cosimulate": {"enabled": False, "reason": "why"},
            "synthesise": {"enabled": False, "reason": "later"},
        }
    ) == ("cosimulate", "why")


@pytest.fixture(scope="module")
def adaptive_cosim() -> tuple[ResolvedModelCompileConfiguration, str, dict[str, Any], Any]:
    """Run one real AdaptiveThresholdIF co-simulation for the stop and summary cases."""
    if not HAS_COSIM:
        pytest.skip("gcc, iverilog and vvp are required")
    configuration = resolve_model_compile_configuration(
        {"model_name": "AdaptiveThresholdIFNeuron", "integrator": "map", "q_format": "Q8.8"}
    )
    execution = run_model_cosim(configuration, current=10.0, n_steps=16)
    return configuration, configuration.to_verilog(), dict(execution.report), execution


class TestCosimulationStop:
    """RTL that disagrees with its reference is never passed on."""

    def test_a_bit_exact_report_passes(self, adaptive_cosim: Any) -> None:
        """The real report is completed and bit-exact."""
        _configuration, _verilog, report, _execution = adaptive_cosim
        assert require_bit_exact(report, model_name="AdaptiveThresholdIFNeuron") is report

    def test_a_differing_requested_run_stops_the_path(self, adaptive_cosim: Any) -> None:
        """A real trace with one state word changed stops the path at its first cycle."""
        _configuration, _verilog, report, execution = adaptive_cosim
        rtl_trace = [list(row) for row in execution.rtl_trace]
        rtl_trace[5][1] += 1
        mismatch = _first_mismatch(rtl_trace, execution.reference_trace, report["signals"])
        assert mismatch is not None
        planted = {**report, "bit_exact": False, "first_mismatch": mismatch}
        with pytest.raises(HardwarePathRefusal) as raised:
            require_bit_exact(planted, model_name="AdaptiveThresholdIFNeuron")
        refusal = raised.value
        assert (refusal.stage, refusal.code) == ("cosimulation", "cosim-not-bit-exact")
        assert "requested run at cycle 6 on v_out" in refusal.message
        assert "the RTL is not synthesised" in refusal.message

    def test_a_differing_stress_schedule_stops_the_path(self, adaptive_cosim: Any) -> None:
        """A difference only in the stress schedule also stops the path."""
        _configuration, _verilog, report, execution = adaptive_cosim
        stress_trace = [list(row) for row in execution.stress_rtl_trace]
        stress_trace[0][0] ^= 1
        mismatch = _first_mismatch(
            stress_trace, execution.stress_reference_trace, report["signals"]
        )
        planted = {
            **report,
            "bit_exact": False,
            "stress": {**report["stress"], "bit_exact": False, "first_mismatch": mismatch},
        }
        with pytest.raises(HardwarePathRefusal, match="stress schedule at cycle 1 on spike_out"):
            require_bit_exact(planted, model_name="AdaptiveThresholdIFNeuron")

    def test_an_incomplete_report_stops_the_path(self, adaptive_cosim: Any) -> None:
        """A report without a completed status is not accepted even if it says bit-exact."""
        _configuration, _verilog, report, _execution = adaptive_cosim
        planted = {**report, "status": "failed", "stress": {}}
        with pytest.raises(HardwarePathRefusal, match="status 'failed'"):
            require_bit_exact(planted, model_name="AdaptiveThresholdIFNeuron")


class TestGuarantees:
    """The formal class follows the job's RTL bytes; readiness follows the seal."""

    def test_formal_only_for_byte_identical_job_rtl(self, tmp_path: Path) -> None:
        """The job's own RTL is FORMAL; one changed byte or another module is not."""
        job = _inventory_job("RulkovMapNeuron")
        configuration = resolve_model_compile_configuration(
            {
                "model_name": "RulkovMapNeuron",
                "q_format": job["q_format"],
                "module_name": job["module"],
            }
        )
        verilog = configuration.to_verilog()
        run = {"module": job["module"], "schema": job["profile"], "q_format": job["q_format"]}
        formal = formal_guarantee("RulkovMapNeuron", verilog, run)
        assert formal["class"] == "FORMAL"
        assert formal["compared_rtl_sha256"] == formal["job_rtl_sha256"]

        copy_dir = tmp_path / "catalogue"
        copy_dir.mkdir()
        inventory = copy_dir / FORMAL_INVENTORY.name
        shutil.copyfile(FORMAL_INVENTORY, inventory)
        job_rtl = (FORMAL_INVENTORY.parent / f"{job['module']}.v").read_bytes()
        (copy_dir / f"{job['module']}.v").write_bytes(job_rtl.replace(b"module", b"module ", 1))
        changed = formal_guarantee("RulkovMapNeuron", verilog, run, inventory=inventory)
        assert changed["class"] == "PROVEN AT OTHER PARAMETERS"
        assert changed["differs"] == []
        assert changed["reason"] == "the job's RTL file differs from the emitted RTL"

        (copy_dir / f"{job['module']}.v").unlink()
        missing = formal_guarantee("RulkovMapNeuron", verilog, run, inventory=inventory)
        assert missing["class"] == "NONE"
        assert missing["job_rtl"] == f"{job['module']}.v" and "job" not in missing
        assert "is not available, so nothing it proves can be attached" in missing["reason"]

        renamed = resolve_model_compile_configuration(
            {"model_name": "RulkovMapNeuron", "q_format": job["q_format"]}
        )
        other = formal_guarantee(
            "RulkovMapNeuron",
            renamed.to_verilog(),
            {**run, "module": renamed.module_name},
        )
        assert other["class"] == "PROVEN AT OTHER PARAMETERS"
        assert other["differs"] == ["module"]

    def test_no_job_and_no_inventory_are_none(self, tmp_path: Path) -> None:
        """A model without a job, or an installation without the inventory, is NONE."""
        document = json.loads(FORMAL_INVENTORY.read_text(encoding="utf-8"))
        classes = {entry["class"] for entry in [*document["jobs"], *document["retained_jobs"]]}
        assert "GLIFNeuron" not in classes
        none = formal_guarantee("GLIFNeuron", "module m; endmodule", {})
        assert none["class"] == "NONE"
        assert none["reason"] == "the catalogue has no formal job for GLIFNeuron"

        absent = formal_guarantee(
            "LapicqueNeuron", "module m; endmodule", {}, inventory=tmp_path / "inventory.json"
        )
        assert absent["class"] == "NONE"
        assert absent["inventory_available"] is False

    def test_readiness_comes_from_the_seal_or_says_it_is_unsealed(self, tmp_path: Path) -> None:
        """The sealed labels are served; a missing seal is reported, never given tiers."""
        sealed = readiness_labels("AdaptiveThresholdIFNeuron")
        assert sealed["source"] == "sealed"
        assert (
            sealed["silicon_label"] == sealed_detail("AdaptiveThresholdIFNeuron")["silicon_label"]
        )
        unsealed = readiness_labels("AdaptiveThresholdIFNeuron", tmp_path / "seal.json")
        assert unsealed["source"] == "unsealed"
        assert unsealed["science_label"] == "S0"
        assert "carries no readiness seal" in unsealed["unsealed_reason"]

    def test_summary_without_inventory_or_seal_says_so(
        self, adaptive_cosim: Any, tmp_path: Path
    ) -> None:
        """An installation without the formal inventory or seal records both honestly."""
        configuration, verilog, report, _execution = adaptive_cosim
        summary = model_guarantee_summary(
            configuration,
            verilog,
            report,
            formal_inventory=tmp_path / "inventory.json",
            readiness_seal=tmp_path / "seal.json",
        )
        assert summary["classes"]["NONE"] == 1
        assert summary["classes"]["CO-SIMULATED ONLY"] == 1
        text = render_model_guarantee_text(summary)
        assert "formal: NONE (the formal inventory (inventory.json) is not available" in text
        assert "readiness (unsealed): S0 / none (AdaptiveThresholdIFNeuron:" in text


class TestSupportedModels:
    """The supported set comes from the capability matrix."""

    def test_model_and_host_verdicts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model verdicts do not depend on the host; host verdicts follow its tools."""
        names = ["LapicqueNeuron", "ATypeKNeuron", "SCScaledResetAdaptiveIFNeuron", "NoSuch"]
        monkeypatch.setenv("PATH", str(tmp_path))
        rows = {row["model"]: row for row in supported_models(names)}
        assert list(rows) == names
        lapicque = rows["LapicqueNeuron"]
        assert lapicque["model_supported"] is True and lapicque["model_refusal"] is None
        assert lapicque["combinations"] == [
            {"integrator": "map", "q_format": "Q8.8"},
            {"integrator": "map", "q_format": "Q16.16"},
        ]
        assert lapicque["supported"] is False
        assert lapicque["refusal"]["operation"] == "cosimulate"
        assert rows["ATypeKNeuron"]["model_refusal"]["operation"] == "compile"
        assert rows["SCScaledResetAdaptiveIFNeuron"]["model_refusal"] == {
            "operation": "cosimulate",
            "reason": "no bit-true C kernel mirrors this model's RTL for any offered integrator",
        }
        assert rows["NoSuch"]["refusal"] == {
            "operation": "catalogue",
            "reason": "the catalogue holds no descriptor",
        }

    @needs_synthesis
    def test_a_host_with_the_tools_supports_the_path(self) -> None:
        """With every required tool installed, a model-supported model is host-supported."""
        (row,) = supported_models(["AdaptiveThresholdIFNeuron"])
        assert row["supported"] is True and row["refusal"] is None


class TestSynthesisStageRefusals:
    """Synthesis that cannot complete stops the path with the report written."""

    def _recorder(self, tmp_path: Path) -> Any:
        """Return a recorder for one model's synthesis stage."""
        from sc_neurocore.studio.hardware_path import _Recorder

        return _Recorder(tmp_path, {"model": "AdaptiveThresholdIFNeuron"})

    def _traceability(
        self, configuration: ResolvedModelCompileConfiguration, verilog: str
    ) -> dict[str, Any]:
        """Return the compile traceability record of ``verilog`` for the configuration."""
        from sc_neurocore.studio.compile_traceability import build_model_compile_traceability

        return dict(
            build_model_compile_traceability(
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
        )

    def test_rtl_that_is_not_the_co_simulated_bytes_is_refused(
        self, adaptive_cosim: Any, tmp_path: Path
    ) -> None:
        """The digest-bound terminal refuses RTL whose bytes differ from the evidence."""
        from sc_neurocore.studio.hardware_path import _terminal_stage

        configuration, verilog, report, _execution = adaptive_cosim
        traceability = self._traceability(configuration, verilog)
        with pytest.raises(HardwarePathRefusal) as raised:
            _terminal_stage(
                self._recorder(tmp_path),
                {"establishes": [], "does_not_establish": []},
                verilog + "// changed after co-simulation\n",
                "ice40",
                traceability,
                report,
            )
        assert (raised.value.stage, raised.value.code) == ("synthesis", "terminal-refused")
        assert "does not match the compile output digest" in raised.value.message

    @needs_synthesis
    def test_rtl_that_yosys_rejects_is_refused_with_the_report(
        self, adaptive_cosim: Any, tmp_path: Path
    ) -> None:
        """Synthesis that fails stops the path; its report is written first."""
        from sc_neurocore.studio.hardware_path import _synthesis_only_stage
        from sc_neurocore.studio.synthesis import check_tools

        recorder = self._recorder(tmp_path)
        with pytest.raises(HardwarePathRefusal) as raised:
            _synthesis_only_stage(
                recorder,
                {"establishes": [], "does_not_establish": []},
                "module broken(input wire a; endmodule\n",
                "ice40",
                check_tools(),
                "not under test",
                adaptive_cosim[2],
            )
        assert (raised.value.stage, raised.value.code) == ("synthesis", "synthesis-failed")
        report = tmp_path / "synthesis" / "report.json"
        assert json.loads(report.read_text(encoding="utf-8"))["success"] is False
        assert recorder.artifacts["synthesis/report.json"]["sha256"] == _sha256(report)

    @needs_synthesis
    def test_terminal_synthesis_failure_is_refused_with_the_report(
        self, adaptive_cosim: Any, tmp_path: Path
    ) -> None:
        """A digest-consistent chain whose RTL Yosys rejects stops before place and route.

        The co-simulation report is the real one with only its RTL digest pointed at
        the rejected text, so the terminal's chain check passes and Yosys runs.
        """
        from sc_neurocore.studio.hardware_path import _terminal_stage

        configuration, _verilog, report, _execution = adaptive_cosim
        broken = f"module {configuration.module_name}(input wire clk; endmodule\n"
        digest = hashlib.sha256(broken.encode("utf-8")).hexdigest()
        chained = {**copy.deepcopy(report), "rtl": {**report["rtl"], "source_sha256": digest}}
        recorder = self._recorder(tmp_path)
        with pytest.raises(HardwarePathRefusal) as raised:
            _terminal_stage(
                recorder,
                {"establishes": [], "does_not_establish": []},
                broken,
                "ice40",
                self._traceability(configuration, broken),
                chained,
            )
        assert (raised.value.stage, raised.value.code) == ("synthesis", "synthesis-failed")
        assert "synthesis/report.json" in recorder.artifacts


class TestTheBitExactStopInThePath:
    """A co-simulation that does not agree never reaches synthesis, on any target."""

    @staticmethod
    def _planted(configuration: Any, *, current: float, n_steps: int) -> Any:
        """Run the real co-simulation, then perturb one RTL state word of its trace.

        The mismatch is computed by the co-simulation's own comparison, as the unit
        tests of the stop do; only the report it returns is replaced.
        """
        import dataclasses

        execution = run_model_cosim(configuration, current=current, n_steps=n_steps)
        rtl_trace = [list(row) for row in execution.rtl_trace]
        rtl_trace[3][1] += 1
        report: dict[str, Any] = dict(execution.report)
        mismatch = _first_mismatch(rtl_trace, execution.reference_trace, report["signals"])
        report.update(bit_exact=False, first_mismatch=mismatch)
        return dataclasses.replace(execution, report=report, rtl_trace=rtl_trace)

    @needs_synthesis
    def test_a_synthesis_only_target_is_refused_at_cosimulation(self, tmp_path: Path) -> None:
        """Through the whole path, non-bit-exact RTL leaves no synthesis artefact."""
        out = tmp_path / "planted"
        with pytest.raises(HardwarePathRefusal) as raised:
            run_hardware_path(
                "AdaptiveThresholdIFNeuron",
                out_dir=out,
                q_format="Q8.8",
                integrator="map",
                n_steps=16,
                target="gowin",
                cosimulate=self._planted,
            )
        refusal = raised.value
        assert (refusal.stage, refusal.code) == ("cosimulation", "cosim-not-bit-exact")
        manifest = _manifest_of(refusal)
        assert manifest["status"] == "refused"
        assert (
            json.loads((out / "cosim/report.json").read_text(encoding="utf-8"))["bit_exact"]
            is False
        )
        assert not (out / "synthesis").exists()
        assert not any(name.startswith("synthesis/") for name in manifest["artifacts"])
        assert not any("Yosys synthesised" in item for item in manifest["boundary"]["establishes"])
        assert [stage["stage"] for stage in manifest["stages"]] == ["capabilities", "configuration"]

    @needs_synthesis
    def test_the_synthesis_only_stage_repeats_the_check(
        self, adaptive_cosim: Any, tmp_path: Path
    ) -> None:
        """Called with a report that is not bit-exact, the stage refuses before Yosys runs."""
        from sc_neurocore.studio.hardware_path import _Recorder, _synthesis_only_stage
        from sc_neurocore.studio.synthesis import check_tools

        _configuration, verilog, report, execution = adaptive_cosim
        rtl_trace = [list(row) for row in execution.rtl_trace]
        rtl_trace[2][1] -= 1
        mismatch = _first_mismatch(rtl_trace, execution.reference_trace, report["signals"])
        planted = {**report, "bit_exact": False, "first_mismatch": mismatch}
        recorder = _Recorder(tmp_path, {"model": "AdaptiveThresholdIFNeuron"})
        with pytest.raises(HardwarePathRefusal) as raised:
            _synthesis_only_stage(
                recorder,
                {"establishes": [], "does_not_establish": []},
                verilog,
                "gowin",
                check_tools(),
                "not under test",
                planted,
            )
        assert (raised.value.stage, raised.value.code) == ("cosimulation", "cosim-not-bit-exact")
        assert recorder.artifacts == {} and not (tmp_path / "synthesis").exists()
