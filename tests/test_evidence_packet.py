# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evidence packet export and verification

"""Export real evidence packets of dense designs, verify them, and refuse every tamper.

Packets are written for real dense graphs with the installed engine, Icarus Verilog,
SymbiYosys and Yosys. A formal job whose solver is missing must be recorded as
unavailable with that reason, never as a result. Every member is re-hashed by the
verifier; a missing, extra, altered or symbolically linked member and a broken
manifest digest must each be refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir  # noqa: E402

from sc_neurocore.cli import main  # noqa: E402
from sc_neurocore.safety_cert.dense_cosim import (  # noqa: E402
    PRIMITIVES,
    SimulatorUnavailableError,
    UnsupportedDesignError,
    cosimulate,
    parse_dense_design,
)
from sc_neurocore.safety_cert.evidence_packet import (  # noqa: E402
    CONTAINED_MODULES,
    MANIFEST,
    SCHEMA,
    TOOLS,
    FormalTarget,
    boundary_statement,
    checkout_root,
    export_evidence_packet,
    formal_targets,
    place_and_route,
    run_formal_job,
    source_identity,
    synthesize,
    tool_versions,
    verify_evidence_packet,
)
from sc_neurocore.compiler.guarantee_summary import summarize_graph  # noqa: E402
from sc_neurocore.hdl.resources import baseline_primitive_text  # noqa: E402
from sc_neurocore.studio.synthesis import SYNTHESIS_TIMEOUT_SECONDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: Per-tool wall-clock limit these tests give Yosys and nextpnr. A reference value for
#: the hosts that run them (hosted CI runners, the workstation); the library has no
#: default, and converting test limits to host-derived values is a tracked backlog item.
TOOL_TIMEOUT_S = 900.0
_DCLS = (
    'sc.graph @dcls {\n  %0 = sc.input "spike" : bitstream<7>\n'
    '  %1 = sc.input "weights" : vec<fixed<16,8>,1>\n'
    '  %2 = sc.input "centre" : fixed<16,8>\n  %3 = sc.input "sigma" : fixed<16,8>\n'
    "  %4 = sc.dcls_layer %0, weights=%1, centre=%2, sigma=%3, taps=1, depth=7, dw=16, "
    'frac=8, ptr=3, offsets=[6] : fixed<16,8>\n  sc.output "result" %4\n}\n'
)


def dense_text(
    n_inputs: int, n_neurons: int, width: int, weights: list[float], weight_type: str | None
) -> str:
    """Return the IR text of a real dense graph built with the engine.

    Parameters
    ----------
    n_inputs, n_neurons, width : int
        Geometry and state width.
    weights : list of float
        Row-major weights.
    weight_type : str or None
        Declared weight vector type, or the builder's default.

    Returns
    -------
    str
        The graph's IR text.
    """
    builder = ScGraphBuilder(f"dense_{n_inputs}x{n_neurons}_w{width}")
    inputs = builder.input("inputs", f"vec<rate,{n_inputs}>")
    matrix = (
        builder.constant_f64_vec(weights)
        if weight_type is None
        else builder.constant_f64_vec(weights, weight_type)
    )
    leak = builder.constant_i64(16, f"fixed<{width},8>")
    gain = builder.constant_i64(256, f"fixed<{width},8>")
    result = builder.dense_forward(
        inputs,
        matrix,
        leak,
        gain,
        n_inputs=n_inputs,
        n_neurons=n_neurons,
        data_width=width,
        seed_base=0x2345,
        stream_length=9,
        y_min=0,
        y_max=300,
    )
    builder.output("spikes", result)
    graph = builder.build()
    assert graph.verify() is None
    return str(graph.to_text())


def require_simulator() -> None:
    """Skip when Icarus Verilog is missing, without which no packet can be exported."""
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog (iverilog, vvp) is not installed; CI installs it")


def read_json(path: Path) -> Any:
    """Return the JSON value of a file."""
    return json.loads(path.read_text(encoding="utf-8"))


def require_place_and_route() -> None:
    """Skip without nextpnr-ice40, unless the resource-check CI job demands it."""
    if shutil.which("nextpnr-ice40") is None:
        if os.environ.get("SC_NEUROCORE_REQUIRE_FPGA_TOOLS") == "1":
            raise RuntimeError("SC_NEUROCORE_REQUIRE_FPGA_TOOLS=1 but nextpnr-ice40 is missing")
        pytest.skip("nextpnr-ice40 is not installed; the CI test job installs it")


DESIGNS = {
    "2x2": (2, 2, 16, [0.5, 0.25, 0.75, 0.9], None),
    "3x2": (3, 2, 12, [0.0, 0.25, 1.0, 0.5, -0.1, 0.75], None),
    "3x2 wide": (3, 2, 40, [0.5, 0.25, 1.0, 0.5, 0.9, 0.75], "vec<fixed<12,10>,6>"),
}

#: Designs whose Yosys run may exceed the Studio's fixed 60-second synthesis limit on a
#: loaded host: the 40-bit state words make each neuron's multipliers large.
SLOW_SYNTHESIS = frozenset({"3x2 wide"})


@pytest.fixture(scope="module")
def packet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export one 3x2 packet shared by the tamper tests, which work on copies.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Module-scoped directories.

    Returns
    -------
    Path
        The packet directory.
    """
    require_simulator()
    out = tmp_path_factory.mktemp("shared") / "packet"
    export_evidence_packet(
        dense_text(*DESIGNS["3x2"]), out, cycles=64, pnr_timeout_s=TOOL_TIMEOUT_S
    )
    return out


@pytest.mark.parametrize("label", sorted(DESIGNS))
def test_a_real_dense_design_exports_a_verifying_packet(label: str, tmp_path: Path) -> None:
    """Every member is written from the real design and the packet verifies.

    Parameters
    ----------
    label : str
        Design under test.
    tmp_path : Path
        Output directory.
    """
    require_simulator()
    text = dense_text(*DESIGNS[label])
    graph = parse_ir(text)
    out = tmp_path / "packet"
    export = export_evidence_packet(text, out, pnr_timeout_s=TOOL_TIMEOUT_S)
    assert export.cosim_agrees and export.first_difference is None
    assert (out / "design.sc").read_text(encoding="utf-8") == text
    assert (out / "design.sv").read_text(encoding="utf-8") == graph.emit_sv()
    assert read_json(out / "design.guarantees.json") == summarize_graph(graph)
    assert "CO-SIMULATED ONLY" in (out / "design.guarantees.txt").read_text(encoding="utf-8")
    design = parse_dense_design(graph.to_text())
    assert read_json(out / "parameters.json") == design.to_parameters()
    reference = cosimulate(design, graph.emit_sv())
    assert read_json(out / "cosim/reference_trace.json") == [
        row.to_json() for row in reference.reference
    ]
    assert read_json(out / "cosim/rtl_trace.json") == read_json(out / "cosim/reference_trace.json")
    assert read_json(out / "cosim/stimulus.json") == [list(w) for w in reference.stimulus]
    verdict = read_json(out / "cosim/verdict.json")
    assert verdict["verdict"] == "AGREE" and verdict["cycles"] == 256
    _check_formal(out)
    _check_synthesis(out, may_time_out=label in SLOW_SYNTHESIS)
    _check_place_and_route(out, slow=label in SLOW_SYNTHESIS)
    tools = read_json(out / "tools.json")
    assert set(tools["tools"]) == set(TOOLS) and tools["sc_neurocore"]
    assert tools["tools"]["iverilog"]["available"] is True
    source = read_json(out / "source.json")
    assert source["checkout"] is True and re.fullmatch(r"[0-9a-f]{40}", source["commit"])
    evidence = read_json(out / "evidence.json")
    assert evidence["schema"] == SCHEMA and evidence["graph"] == graph.name
    boundary = evidence["boundary"]
    assert any("all 256 cycles" in sentence for sentence in boundary["establishes"])
    assert any("No unbounded proof" in s for s in boundary["does_not_establish"])
    assert any("not a signature" in s for s in boundary["does_not_establish"])
    text_summary = (out / "evidence.txt").read_text(encoding="utf-8")
    assert "This packet does not establish:" in text_summary
    manifest = read_json(out / MANIFEST)
    listed = {member["path"] for member in manifest["members"]}
    on_disk = {
        p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != MANIFEST
    }
    assert listed == on_disk and manifest["manifest_sha256"] == export.manifest_sha256
    for member in manifest["members"]:
        data = (out / member["path"]).read_bytes()
        assert member["sha256"] == hashlib.sha256(data).hexdigest()
        assert member["size"] == len(data)
    report = verify_evidence_packet(out)
    assert report.ok and report.problems == () and report.members == export.members


def _check_formal(out: Path) -> None:
    """Check that each contained module's job ran and passed, or names what is missing."""
    formal = read_json(out / "formal/formal.json")
    (boundary,) = formal["co_simulated_only"]
    assert boundary["module"] == "sc_dense_matrix_layer"
    assert boundary["test"] == "tests/test_ir_dense_rtl.py"
    jobs = {record["module"]: record for record in formal["jobs"]}
    assert list(jobs) == list(CONTAINED_MODULES["sc_dense_matrix_layer"])
    for module, record in jobs.items():
        assert record["job"] == f"hdl/formal/{module}.sby"
        assert record["mode"] == "prove" and record["depth"] > 0
        assert f"hdl/formal/{module}.sby" in record["sources_sha256"]
        assert record["proved_source_equals_simulated_primitive"] is True
        if record["status"] == "run":
            assert record["verdict"] == "PASS"
            log = (out / record["log"]).read_bytes()
            assert record["log_sha256"] == hashlib.sha256(log).hexdigest()
        else:
            assert record["status"] == "unavailable"
            missing = record["reason"].removeprefix("not installed: ").split(", ")
            assert all(shutil.which(name) is None for name in missing)


def _check_synthesis(out: Path, *, may_time_out: bool) -> None:
    """Check the Yosys record against the tool's presence.

    A design that may exceed the synthesis time limit must then be recorded as failed
    for exactly that reason, with the report written and the gap stated; any other
    outcome than completion or that timeout fails.
    """
    synthesis = read_json(out / "synthesis/synthesis.json")
    if shutil.which("yosys") is None:
        assert synthesis["status"] == "unavailable"
        return
    if may_time_out and synthesis["status"] == "failed":
        assert synthesis["reason"] == f"Synthesis timed out ({SYNTHESIS_TIMEOUT_SECONDS}s)"
        report = (out / synthesis["report"]).read_bytes()
        assert synthesis["report_sha256"] == hashlib.sha256(report).hexdigest()
        gap = f"No synthesis result: synthesis failed ({synthesis['reason']})."
        assert gap in read_json(out / "evidence.json")["boundary"]["does_not_establish"]
        return
    assert synthesis["status"] == "completed"
    report = (out / synthesis["report"]).read_bytes()
    assert synthesis["report_sha256"] == hashlib.sha256(report).hexdigest()
    assert json.loads(report)["resources"]["luts"] > 0


def _check_place_and_route(out: Path, *, slow: bool) -> None:
    """Check the place-and-route record, both reports and the boundary sentence.

    Without nextpnr-ice40 the record names the missing tool and no report is
    written. Otherwise both reports are bound by their SHA-256, the record repeats
    the report's verdicts, and a design inside the calibrated domain routes within
    the device, meets 12 MHz and stays within its estimate. A design whose Yosys
    run may exceed its limit may instead be undetermined at synthesis.
    """
    routed = read_json(out / "synthesis/synthesis.json")["place_and_route"]
    assert (routed["target"], routed["clock_mhz"]) == ("ice40", 12)
    boundary = read_json(out / "evidence.json")["boundary"]
    if shutil.which("nextpnr-ice40") is None or shutil.which("yosys") is None:
        assert routed["status"] == "unavailable" and "not installed" in routed["reason"]
        assert not (out / "synthesis/pnr_report.json").exists()
        assert f"No place and route: {routed['reason']}." in boundary["does_not_establish"]
        return
    assert routed["status"] == "run"
    report_bytes = (out / routed["report"]).read_bytes()
    comparison_bytes = (out / routed["estimate_vs_real"]).read_bytes()
    assert routed["report_sha256"] == hashlib.sha256(report_bytes).hexdigest()
    assert routed["estimate_vs_real_sha256"] == hashlib.sha256(comparison_bytes).hexdigest()
    report, comparison = json.loads(report_bytes), json.loads(comparison_bytes)
    assert routed["verdict"] == report["verdict"]
    assert routed["checks"] == {c["name"]: c["verdict"] for c in report["checks"]}
    assert comparison["estimate"] == report["estimate"]
    assert comparison["core_logic_cells"] == report.get("core_logic_cells")
    assert any(
        s.startswith("nextpnr-ice40 placed and routed the design out of context")
        and f"verdict {routed['verdict']}" in s
        for s in boundary["establishes"]
    )
    if slow and routed["checks"].get("synthesis") == "undetermined":
        return
    assert routed["checks"]["logic_fits_device"] == "pass"
    assert routed["checks"]["timing"] == "pass"
    if report["estimate"]["static_estimate"]:
        assert routed["checks"]["estimate"] == "pass"
        assert comparison["estimate_check"]["verdict"] == "pass"
    else:
        assert "estimate" not in routed["checks"] and comparison["estimate_check"] is None
        assert comparison["estimate_comparison"].startswith("not applicable: ")


def _copy(packet: Path, tmp_path: Path) -> Path:
    """Return a private copy of the shared packet."""
    copy = tmp_path / "copy"
    shutil.copytree(packet, copy, symlinks=True)
    return copy


def _tamper_missing(root: Path) -> None:
    """Delete one member."""
    (root / "cosim" / "rtl_trace.json").unlink()


def _tamper_extra(root: Path) -> None:
    """Add a file the manifest does not list."""
    (root / "cosim" / "notes.txt").write_text("added later\n", encoding="utf-8")


def _tamper_altered(root: Path) -> None:
    """Flip one bit of a member, keeping its size."""
    path = root / "design.sv"
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(bytes(data))


def _tamper_appended(root: Path) -> None:
    """Append one byte to a member."""
    with (root / "parameters.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")


def _tamper_symlink_member(root: Path) -> None:
    """Replace a member with a link to an identical file outside the packet."""
    target = root.parent / "outside.json"
    shutil.copyfile(root / "tools.json", target)
    (root / "tools.json").unlink()
    (root / "tools.json").symlink_to(target)


def _tamper_symlink_directory(root: Path) -> None:
    """Add a link to a member directory."""
    (root / "linked").symlink_to(root / "cosim", target_is_directory=True)


def _tamper_empty_directory(root: Path) -> None:
    """Add a directory that holds no member."""
    (root / "cosim" / "later").mkdir()


def _tamper_fifo(root: Path) -> None:
    """Add a named pipe, which the verifier must report without opening it."""
    os.mkfifo(root / "cosim" / "pipe")


def _tamper_manifest_digest(root: Path) -> None:
    """Change a listed description without updating the manifest digest."""
    manifest = read_json(root / MANIFEST)
    manifest["members"][0]["description"] = "a different description"
    (root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def _tamper_manifest_entry(root: Path) -> None:
    """List an entry whose path escapes the packet and lacks fields."""
    manifest = read_json(root / MANIFEST)
    manifest["members"].append({"path": "../escape", "category": "design"})
    (root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def _tamper_manifest_and_member(root: Path) -> None:
    """List another digest for a member."""
    manifest = read_json(root / MANIFEST)
    member = next(m for m in manifest["members"] if m["path"] == "design.sc")
    member["sha256"] = "0" * 64
    (root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def _tamper_manifest_size(root: Path) -> None:
    """List another size for a member, which the manifest digest does not cover."""
    manifest = read_json(root / MANIFEST)
    member = next(m for m in manifest["members"] if m["path"] == "evidence.txt")
    member["size"] += 1
    (root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


TAMPERS = {
    "missing member": (_tamper_missing, ["missing member: cosim/rtl_trace.json"]),
    "extra file": (_tamper_extra, ["extra file: cosim/notes.txt"]),
    "altered member": (_tamper_altered, ["altered member: design.sv"]),
    "altered size": (_tamper_appended, ["altered member: parameters.json"]),
    "symbolic link member": (_tamper_symlink_member, ["symbolic link: tools.json"]),
    "symbolic link directory": (_tamper_symlink_directory, ["symbolic link: linked"]),
    "empty directory": (_tamper_empty_directory, ["extra directory: cosim/later"]),
    "named pipe": (
        _tamper_fifo,
        ["extra entry that is not a regular file: cosim/pipe"],
    ),
    "manifest digest": (_tamper_manifest_digest, ["manifest.json digest does not match"]),
    "manifest size": (_tamper_manifest_size, ["altered member: evidence.txt"]),
    "invalid manifest entry": (_tamper_manifest_entry, ["invalid manifest entry"]),
    "manifest and member": (
        _tamper_manifest_and_member,
        ["manifest.json digest does not match", "altered member: design.sc"],
    ),
}


@pytest.mark.parametrize("tamper", sorted(TAMPERS))
def test_every_tamper_class_is_refused(packet: Path, tmp_path: Path, tamper: str) -> None:
    """The verifier names each kind of change and the CLI exits with status one.

    Parameters
    ----------
    packet : Path
        The shared packet.
    tmp_path : Path
        Directory of the tampered copy.
    tamper : str
        Kind of change.
    """
    copy = _copy(packet, tmp_path)
    assert verify_evidence_packet(copy).ok
    change, expected = TAMPERS[tamper]
    change(copy)
    report = verify_evidence_packet(copy)
    assert not report.ok
    for message in expected:
        assert any(problem.startswith(message) for problem in report.problems), report.problems
    assert main(["evidence-packet", "verify", str(copy)]) == 1


def _tamper_place_and_route_report(root: Path) -> None:
    """Lower the routed logic-cell count in the place-and-route report."""
    path = root / "synthesis" / "pnr_report.json"
    report = read_json(path)
    report["core_logic_cells"] -= 1
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tamper_estimate_comparison(root: Path) -> None:
    """Delete the estimate-against-routed record."""
    (root / "synthesis" / "estimate_vs_real.json").unlink()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_tamper_place_and_route_report, "altered member: synthesis/pnr_report.json"),
        (_tamper_estimate_comparison, "missing member: synthesis/estimate_vs_real.json"),
    ],
)
def test_the_place_and_route_members_are_bound_by_the_manifest(
    packet: Path, tmp_path: Path, change: Any, message: str
) -> None:
    """Both routed reports are listed, hashed and refused when changed or removed.

    Parameters
    ----------
    packet : Path
        The shared packet.
    tmp_path : Path
        Directory of the tampered copy.
    change : callable
        The change made to the copy.
    message : str
        Expected problem.
    """
    require_place_and_route()
    listed = {member["path"]: member for member in read_json(packet / MANIFEST)["members"]}
    for name in ("synthesis/pnr_report.json", "synthesis/estimate_vs_real.json"):
        assert listed[name]["category"] == "report"
        data = (packet / name).read_bytes()
        assert listed[name]["sha256"] == hashlib.sha256(data).hexdigest()
    copy = _copy(packet, tmp_path)
    change(copy)
    report = verify_evidence_packet(copy)
    assert not report.ok and message in report.problems
    assert main(["evidence-packet", "verify", str(copy)]) == 1


def test_place_and_route_that_cannot_run_writes_no_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsupported target or missing tools leave an unavailable record and no file.

    Parameters
    ----------
    tmp_path : Path
        Packet directories.
    monkeypatch : pytest.MonkeyPatch
        Empties ``PATH`` for the second case.
    """
    graph = parse_ir(dense_text(*DESIGNS["2x2"]))
    other = place_and_route(graph, tmp_path / "ecp5", "ecp5", timeout_seconds=TOOL_TIMEOUT_S)
    assert other == {
        "tool": "nextpnr-ice40",
        "target": "ecp5",
        "clock_mhz": 12,
        "status": "unavailable",
        "reason": "place and route is supported for ['ice40'], not 'ecp5'",
    }
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    missing = place_and_route(graph, tmp_path / "missing", timeout_seconds=TOOL_TIMEOUT_S)
    assert missing["status"] == "unavailable"
    assert missing["reason"] == "not installed: yosys, nextpnr-ice40"
    assert not (tmp_path / "ecp5").exists() and not (tmp_path / "missing").exists()


def test_routed_and_unrouted_place_and_route_are_stated_as_they_are() -> None:
    """A routed design is established with its verdict; an unrouted one is a gap."""
    cosim = {"verdict": "AGREE", "first_difference": None, "cycles": 9}
    routed = {
        "status": "run",
        "target": "ice40",
        "clock_mhz": 100,
        "verdict": "violation",
        "checks": {"io_fits_package": "pass", "timing": "violation"},
    }
    synthesis = {"status": "completed", "reason": None, "target": "ice40"}
    statement = boundary_statement(cosim, [], [], {**synthesis, "place_and_route": routed})
    assert any(
        "at 100 MHz with resource-feedback verdict violation "
        "(io_fits_package pass, timing violation)" in s
        for s in statement["establishes"]
    )
    assert not any(s.startswith("No place and route") for s in statement["does_not_establish"])
    unrouted = {"status": "unavailable", "reason": "not installed: nextpnr-ice40"}
    statement = boundary_statement(cosim, [], [], {**synthesis, "place_and_route": unrouted})
    assert "No place and route: not installed: nextpnr-ice40." in statement["does_not_establish"]
    assert not any("nextpnr-ice40 placed" in s for s in statement["establishes"])
    statement = boundary_statement(cosim, [], [], synthesis)
    assert "No place and route: none was run for this packet." in statement["does_not_establish"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "manifest.json is missing"),
        ("{not json", "manifest.json is not valid JSON"),
        ('{"schema": "other", "members": []}', "manifest.json does not follow schema"),
        ("[]", "manifest.json does not follow schema"),
    ],
)
def test_an_unusable_manifest_is_refused(
    packet: Path, tmp_path: Path, content: str | None, message: str
) -> None:
    """A missing, unreadable or foreign manifest refuses the packet outright.

    Parameters
    ----------
    packet : Path
        The shared packet.
    tmp_path : Path
        Directory of the copy.
    content : str or None
        Replacement manifest text, or ``None`` to delete it.
    message : str
        Expected start of the problem.
    """
    copy = _copy(packet, tmp_path)
    if content is None:
        (copy / MANIFEST).unlink()
    else:
        (copy / MANIFEST).write_text(content, encoding="utf-8")
    report = verify_evidence_packet(copy)
    assert not report.ok and report.problems[0].startswith(message)
    assert report.manifest_sha256 is None


def test_a_missing_directory_is_refused(tmp_path: Path) -> None:
    """A path that is not a directory is not a packet.

    Parameters
    ----------
    tmp_path : Path
        Base directory.
    """
    report = verify_evidence_packet(tmp_path / "absent")
    assert not report.ok and report.problems == (f"{tmp_path / 'absent'} is not a directory",)


def test_unsupported_and_invalid_graphs_write_nothing(tmp_path: Path) -> None:
    """Other operations are refused by name; invalid IR and a used directory are refused.

    Parameters
    ----------
    tmp_path : Path
        Output directories.
    """
    builder = ScGraphBuilder("lif_graph")
    current, leak, gain, noise = (builder.input(n, "i16") for n in ("i", "leak", "gain", "n"))
    builder.output("spike", builder.lif_step(current, leak, gain, noise))
    with pytest.raises(UnsupportedDesignError, match=r"sc\.lif_step"):
        export_evidence_packet(
            builder.build().to_text(), tmp_path / "lif", pnr_timeout_s=TOOL_TIMEOUT_S
        )
    assert not (tmp_path / "lif").exists()
    with pytest.raises(ValueError):
        export_evidence_packet(
            "not an IR graph", tmp_path / "invalid", pnr_timeout_s=TOOL_TIMEOUT_S
        )
    failing = _DCLS.replace("bitstream<7>", "bool").replace("offsets=[6]", "offsets=[0;1]")
    with pytest.raises(ValueError, match="fails IR verification"):
        export_evidence_packet(failing, tmp_path / "failing", pnr_timeout_s=TOOL_TIMEOUT_S)
    with pytest.raises(ValueError, match="clock_mhz must be positive"):
        export_evidence_packet(
            dense_text(*DESIGNS["2x2"]),
            tmp_path / "clock",
            clock_mhz=0,
            pnr_timeout_s=TOOL_TIMEOUT_S,
        )
    used = tmp_path / "used"
    used.mkdir()
    (used / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not an empty directory"):
        export_evidence_packet(dense_text(*DESIGNS["2x2"]), used, pnr_timeout_s=TOOL_TIMEOUT_S)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["used"]


def test_the_contained_modules_are_the_layer_instances() -> None:
    """The modules recorded inside the layer are exactly the ones its RTL instantiates."""
    source = baseline_primitive_text("sc_dense_matrix_layer.v")
    instantiated = re.findall(r"^\s*(sc_[a-z0-9_]+)\s*(?:#\(|u_\w+\s*\()", source, re.M)
    assert tuple(sorted(set(instantiated))) == CONTAINED_MODULES["sc_dense_matrix_layer"]
    assert {f"{name}.v" for name in instantiated} | {"sc_dense_matrix_layer.v"} == set(PRIMITIVES)
    for name in CONTAINED_MODULES["sc_dense_matrix_layer"]:
        assert (ROOT / "hdl" / "formal" / f"{name}.sby").is_file()


def test_modules_the_summary_names_are_run_once_each() -> None:
    """FORMAL and PROVEN AT OTHER PARAMETERS entries become one job per module."""
    builder = ScGraphBuilder("encoders")
    x, y = builder.input("x", "rate"), builder.input("y", "rate")
    product = builder.bitwise_and(builder.encode(x, 256, 0xACE1), builder.encode(y, 256, 0xBEEF))
    builder.output("count", builder.popcount(product))
    summary = summarize_graph(builder.build())
    targets, boundaries = formal_targets(summary)
    assert boundaries == []
    (target,) = targets
    assert target.module == "sc_bitstream_encoder"
    assert target.job == "hdl/formal/sc_bitstream_encoder.sby"
    assert target.origin.startswith("named by the guarantee summary")
    assert target.summary is not None and target.summary["class"] == "FORMAL"
    record = run_formal_job(None, target, Path("unused"))
    assert record["class"] == "FORMAL" and "differs" not in record
    assert record["proved_parameters"] == record["instance_parameters"]
    assert record["status"] == "unavailable"


def test_the_checkout_is_found_only_from_a_source_tree(tmp_path: Path) -> None:
    """Formal jobs and the commit are read only from a checkout holding them.

    Parameters
    ----------
    tmp_path : Path
        A tree laid out like an installed package.
    """
    import sc_neurocore.safety_cert.evidence_packet as module

    assert checkout_root(Path(module.__file__)) == ROOT
    installed = tmp_path / "lib" / "site-packages" / "sc_neurocore" / "compiler" / "x.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    assert checkout_root(installed) is None


def _require_formal(engine_solver: str) -> None:
    """Skip when SymbiYosys, Yosys or the named solver is missing."""
    missing = [n for n in ("sby", "yosys", engine_solver) if shutil.which(n) is None]
    if missing:
        pytest.skip(f"not installed: {', '.join(missing)}; CI installs the formal toolchain")


def _dcls_target() -> FormalTarget:
    """Return the summary-named target of a real DCLS layer graph."""
    (target,), _ = formal_targets(summarize_graph(parse_ir(_DCLS)))
    return target


def test_a_named_job_runs_in_a_scratch_copy_and_passes(tmp_path: Path) -> None:
    """The DCLS job runs outside the checkout, passes, and its log is hashed.

    Parameters
    ----------
    tmp_path : Path
        Packet directory receiving the log.
    """
    _require_formal("cvc5")
    before = sorted(p.name for p in (ROOT / "hdl" / "formal").iterdir())
    record = run_formal_job(ROOT, _dcls_target(), tmp_path)
    assert sorted(p.name for p in (ROOT / "hdl" / "formal").iterdir()) == before
    assert record["status"] == "run" and record["verdict"] == "PASS" and record["rc"] == 0
    assert (record["mode"], record["depth"], record["engine"]) == ("bmc", 16, "smtbmc cvc5")
    assert record["class"] == "PROVEN AT OTHER PARAMETERS" and "N_TAPS" in record["differs"]
    log = (tmp_path / "formal" / "sc_dcls_layer_core.log").read_bytes()
    assert record["log"] == "formal/sc_dcls_layer_core.log"
    assert record["log_sha256"] == hashlib.sha256(log).hexdigest()
    assert b"DONE (PASS, rc=0)" in log
    assert (
        record["sources_sha256"]["hdl/sc_dcls_layer_core.v"]
        == hashlib.sha256((ROOT / "hdl" / "sc_dcls_layer_core.v").read_bytes()).hexdigest()
    )
    assert "proved_source_equals_simulated_primitive" not in record
    statement = boundary_statement(
        {"verdict": "DIFFER", "first_difference": 3, "cycles": 9},
        [record],
        [],
        {"status": "completed", "reason": None, "target": "ice40"},
    )
    assert any("verdict PASS (bmc, depth 16" in s for s in statement["establishes"])
    assert any("DIFFER, first at cycle 3" in s for s in statement["establishes"])


def test_an_unrun_job_and_failed_synthesis_are_listed_as_not_established() -> None:
    """A formal job or synthesis without a result is a gap, never an established fact."""
    unrun = {
        "module": "sc_lif_neuron",
        "job": "hdl/formal/sc_lif_neuron.sby",
        "status": "unavailable",
        "reason": "not installed: yices-smt2",
    }
    statement = boundary_statement(
        {"verdict": "AGREE", "first_difference": None, "cycles": 9},
        [unrun],
        [],
        {"status": "failed", "reason": "no netlist", "target": "ice40"},
    )
    excluded = statement["does_not_establish"]
    assert excluded[:2] == [
        "No formal result for sc_lif_neuron: hdl/formal/sc_lif_neuron.sby was not run "
        "(not installed: yices-smt2).",
        "No synthesis result: synthesis failed (no netlist).",
    ]
    assert not any("sc_lif_neuron" in s or "Synthesis" in s for s in statement["establishes"])


@pytest.mark.parametrize("verdict", ["TIMEOUT", "FAIL", "UNKNOWN", "ERROR"])
def test_a_run_without_a_pass_is_listed_as_not_established(verdict: str) -> None:
    """Only a PASS verdict is established; any other verdict is a gap.

    Parameters
    ----------
    verdict : str
        A verdict other than ``PASS`` that a run can end with.
    """
    record = {
        "module": "sc_lif_neuron",
        "job": "hdl/formal/sc_lif_neuron.sby",
        "status": "run",
        "verdict": verdict,
        "mode": "prove",
        "depth": 25,
        "engine": "smtbmc",
    }
    statement = boundary_statement(
        {"verdict": "AGREE", "first_difference": None, "cycles": 9},
        [record],
        [],
        {"status": "completed", "reason": None, "target": "ice40"},
    )
    assert statement["does_not_establish"][0] == (
        "No formal result for sc_lif_neuron: hdl/formal/sc_lif_neuron.sby ran with verdict "
        f"{verdict}, which proves nothing."
    )
    assert not any("sc_lif_neuron" in s for s in statement["establishes"])


def _primitive_repository(root: Path, source: str) -> FormalTarget:
    """Write a repository whose ``sc_lif_neuron`` job reads the given module source."""
    (root / "hdl" / "formal").mkdir(parents=True)
    (root / "hdl" / "sc_lif_neuron.v").write_text(source, encoding="utf-8")
    (root / "hdl" / "formal" / "sc_lif_neuron.sby").write_text(
        "[options]\nmode bmc\ndepth 2\n\n[engines]\nsmtbmc z3\n\n[script]\n"
        "read -formal sc_lif_neuron.v\nprep -top sc_lif_neuron\n\n[files]\n"
        "../sc_lif_neuron.v\n",
        encoding="utf-8",
    )
    return FormalTarget("sc_lif_neuron", "hdl/formal/sc_lif_neuron.sby", "a test job")


def test_a_proof_of_another_source_than_the_simulated_one_is_a_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job that reads a source other than the simulated primitive proves nothing about it.

    Parameters
    ----------
    tmp_path : Path
        Repositories and packet directory.
    monkeypatch : pytest.MonkeyPatch
        Empties ``PATH`` so that no job runs; the source comparison precedes the run.
    """
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    cosim = {"verdict": "AGREE", "first_difference": None, "cycles": 9}
    synthesis = {"status": "completed", "reason": None, "target": "ice40"}
    gap = (
        "No formal result for the simulated sc_lif_neuron: hdl/formal/sc_lif_neuron.sby "
        "reads hdl/sc_lif_neuron.v, which differs from the packaged sc_lif_neuron.v "
        "primitive the co-simulation used."
    )
    other = _primitive_repository(tmp_path / "other", "module sc_lif_neuron; endmodule\n")
    record = run_formal_job(tmp_path / "other", other, tmp_path)
    assert record["proved_source_equals_simulated_primitive"] is False
    assert gap in boundary_statement(cosim, [record], [], synthesis)["does_not_establish"]
    same = _primitive_repository(tmp_path / "same", baseline_primitive_text("sc_lif_neuron.v"))
    record = run_formal_job(tmp_path / "same", same, tmp_path)
    assert record["proved_source_equals_simulated_primitive"] is True
    assert gap not in boundary_statement(cosim, [record], [], synthesis)["does_not_establish"]


def test_a_job_that_overruns_its_limit_is_recorded_as_a_timeout(tmp_path: Path) -> None:
    """A run stopped by its wall-clock limit is a TIMEOUT record, not a verdict.

    Parameters
    ----------
    tmp_path : Path
        Packet directory.
    """
    _require_formal("cvc5")
    record = run_formal_job(ROOT, _dcls_target(), tmp_path, timeout_s=0.05)
    assert record["status"] == "run" and record["verdict"] == "TIMEOUT"
    assert "timed out" in record["reason"] and "log" not in record


def _small_repository(root: Path, files: str) -> FormalTarget:
    """Write a repository holding one SymbiYosys job and return its target."""
    (root / "hdl" / "formal").mkdir(parents=True)
    (root / "hdl" / "small.v").write_text("module small; endmodule\n", encoding="utf-8")
    (root / "hdl" / "formal" / "small.sby").write_text(
        "[options]\nmode bmc\ndepth 2\n\n[engines]\nsmtbmc z3\n\n[script]\n"
        f"read -formal small.v\nprep -top small\n\n[files]\n{files}\n",
        encoding="utf-8",
    )
    return FormalTarget("small", "hdl/formal/small.sby", "a test job")


def test_jobs_that_cannot_run_are_recorded_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No checkout, absent sources or missing tools leave an unavailable record.

    Parameters
    ----------
    tmp_path : Path
        Repositories and packet directory.
    monkeypatch : pytest.MonkeyPatch
        Sets ``PATH`` for the last case only.
    """
    target = FormalTarget("small", "hdl/formal/small.sby", "a test job")
    record = run_formal_job(None, target, tmp_path)
    assert record["status"] == "unavailable" and "not present" in record["reason"]
    absent = _small_repository(tmp_path / "absent", "nowhere.v")
    record = run_formal_job(tmp_path / "absent", absent, tmp_path)
    assert record["status"] == "unavailable" and "sources are missing" in record["reason"]
    good = _small_repository(tmp_path / "good", "../small.v")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    record = run_formal_job(tmp_path / "good", good, tmp_path)
    assert record == {
        "module": "small",
        "job": "hdl/formal/small.sby",
        "origin": "a test job",
        "mode": "bmc",
        "depth": 2,
        "engine": "smtbmc z3",
        "sources_sha256": {
            "hdl/formal/small.sby": hashlib.sha256(
                (tmp_path / "good/hdl/formal/small.sby").read_bytes()
            ).hexdigest(),
            "hdl/small.v": hashlib.sha256(b"module small; endmodule\n").hexdigest(),
        },
        "assumptions": [],
        "status": "unavailable",
        "reason": "not installed: sby, yosys, z3",
    }


def test_a_scratch_copy_inside_the_checkout_is_refused(tmp_path: Path) -> None:
    """The job is never run where its run directory would land in the checkout.

    Parameters
    ----------
    tmp_path : Path
        Repository and packet directory.
    """
    _require_formal("z3")
    target = _small_repository(tmp_path / "repo", "../small.v")
    inside = tmp_path / "repo" / "scratch"
    inside.mkdir()
    with pytest.raises(ValueError, match="outside the checkout"):
        run_formal_job(tmp_path / "repo", target, tmp_path, scratch_parent=inside)
    assert list(inside.iterdir()) == []


@pytest.mark.parametrize(
    ("engine", "binary"),
    [("smtbmc", "yices-smt2"), ("smtbmc --nopresat z3", "z3"), ("abc pdr", None), ("", None)],
)
def test_the_solver_an_engine_line_needs(engine: str, binary: str | None) -> None:
    """``smtbmc`` defaults to Yices; other engines need no separate solver binary.

    Parameters
    ----------
    engine : str
        Engine line of a job.
    binary : str or None
        Solver executable it needs.
    """
    from sc_neurocore.safety_cert.evidence_packet import _solver_binary

    assert _solver_binary(engine) == binary


def test_synthesis_outcomes_are_recorded_as_they_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed, failed, refused or impossible synthesis each leaves its own record.

    Parameters
    ----------
    tmp_path : Path
        Packet directories.
    monkeypatch : pytest.MonkeyPatch
        Sets ``PATH`` for the last case only.
    """
    if shutil.which("yosys") is None:
        pytest.skip("Yosys is not installed; CI installs it")
    good = synthesize(
        "module top(input wire a, input wire b, output wire y); assign y = a & b; endmodule\n",
        tmp_path / "good",
    )
    assert good["status"] == "completed" and good["reason"] is None
    report = (tmp_path / "good" / good["report"]).read_bytes()
    assert good["report_sha256"] == hashlib.sha256(report).hexdigest()
    bad = synthesize("module top(input wire a; endmodule\n", tmp_path / "bad")
    assert bad["status"] == "failed" and "Synthesis failed" in bad["reason"]
    assert (tmp_path / "bad" / "synthesis" / "yosys_report.json").is_file()
    large = synthesize("// " + "x" * (2 * 1024 * 1024) + "\n", tmp_path / "large")
    assert large["status"] == "failed" and "2 MiB" in large["reason"]
    assert not (tmp_path / "large").exists()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    missing = synthesize("module top; endmodule\n", tmp_path / "missing", "ecp5")
    assert missing["status"] == "unavailable" and missing["target"] == "ecp5"
    assert missing["reason"] == "yosys is not installed"


def test_tool_versions_and_source_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Versions come from the tools themselves; the commit only from a real checkout.

    Parameters
    ----------
    tmp_path : Path
        A directory outside any checkout.
    monkeypatch : pytest.MonkeyPatch
        Sets ``PATH`` for the last cases only.
    """
    import sc_neurocore

    versions = tool_versions()
    for name, entry in versions["tools"].items():
        assert entry["available"] is (shutil.which(name) is not None)
        assert (entry["version"] is None) is (not entry["available"])
    assert versions["sc_neurocore"] == sc_neurocore.__version__
    assert versions["python"]["version"].count(".") == 2
    identity = source_identity(ROOT)
    assert identity["checkout"] is True and isinstance(identity["dirty"], bool)
    assert source_identity(None) == {
        "package_version": sc_neurocore.__version__,
        "checkout": False,
        "reason": "not run from a source checkout",
    }
    assert source_identity(tmp_path)["reason"] == "not run from a Git checkout"
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert source_identity(ROOT)["reason"] == "git is not installed"
    assert not any(entry["available"] for entry in tool_versions()["tools"].values())


def _git(repository: Path, *arguments: str) -> str:
    """Run git in a test repository and return its output."""
    git = shutil.which("git")
    assert git is not None
    return subprocess.run(
        [git, "-C", str(repository), *arguments],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    ).stdout.strip()


def _checkout(root: Path) -> Path:
    """Create a Git checkout with one commit."""
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Tests")
    (root / "notes.txt").write_text("first\n", encoding="utf-8")
    _git(root, "add", "notes.txt")
    _git(root, "commit", "-q", "-m", "first")
    return root


def test_the_source_identity_records_the_commit_and_a_dirty_tree(tmp_path: Path) -> None:
    """A clean checkout is not dirty; a changed or untracked file makes it dirty.

    Parameters
    ----------
    tmp_path : Path
        Holds the test checkout.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not installed; CI installs it")
    root = _checkout(tmp_path / "checkout")
    clean = source_identity(root)
    assert clean["checkout"] is True and clean["dirty"] is False
    assert clean["commit"] == _git(root, "rev-parse", "HEAD")
    (root / "notes.txt").write_text("changed\n", encoding="utf-8")
    assert source_identity(root)["dirty"] is True
    _git(root, "checkout", "-q", "--", "notes.txt")
    (root / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert source_identity(root)["dirty"] is True


def test_a_packet_written_inside_the_checkout_does_not_mark_it_dirty(tmp_path: Path) -> None:
    """The source identity is read before any member is written.

    Parameters
    ----------
    tmp_path : Path
        Holds the test checkout and the packet inside it.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not installed; CI installs it")
    require_simulator()
    root = _checkout(tmp_path / "checkout")
    out = root / "packet"
    export_evidence_packet(
        dense_text(*DESIGNS["3x2"]), out, cycles=8, repository=root, pnr_timeout_s=TOOL_TIMEOUT_S
    )
    identity = read_json(out / "source.json")
    assert identity["dirty"] is False and identity["commit"] == _git(root, "rev-parse", "HEAD")
    assert source_identity(root)["dirty"] is True


def test_the_cli_exports_and_verifies_a_packet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``evidence-packet export`` writes a packet that ``evidence-packet verify`` accepts.

    Parameters
    ----------
    tmp_path : Path
        Graph file and packet directory.
    capsys : pytest.CaptureFixture[str]
        Captured output.
    """
    require_simulator()
    graph = tmp_path / "dense.sc"
    graph.write_text(dense_text(*DESIGNS["2x2"]), encoding="utf-8")
    out = tmp_path / "packet"
    arguments = ["evidence-packet", "export", str(graph), "-o", str(out), "--cycles", "32"]
    arguments += ["--pnr-timeout-seconds", str(TOOL_TIMEOUT_S)]
    assert main([*arguments, "--clock-mhz", "20"]) == 0
    printed = capsys.readouterr().out
    assert "Co-simulation:    AGREE" in printed and str(out) in printed
    assert read_json(out / "cosim" / "verdict.json")["cycles"] == 32
    routed = read_json(out / "synthesis" / "synthesis.json")["place_and_route"]
    assert routed["clock_mhz"] == 20
    assert "Place and route (ice40, 20 MHz): " in (out / "evidence.txt").read_text("utf-8")
    assert main(["evidence-packet", "verify", str(out)]) == 0
    manifest = read_json(out / MANIFEST)
    assert f"Manifest SHA-256: {manifest['manifest_sha256']}" in capsys.readouterr().out


def test_the_cli_refuses_with_status_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable file, an unsupported graph or a missing simulator exits with one.

    Parameters
    ----------
    tmp_path : Path
        Graph files and packet directories.
    capsys : pytest.CaptureFixture[str]
        Captured output.
    monkeypatch : pytest.MonkeyPatch
        Sets ``PATH`` for the last case only.
    """
    missing = tmp_path / "missing.sc"
    assert (
        main(
            [
                "evidence-packet",
                "export",
                str(missing),
                "-o",
                str(tmp_path / "a"),
                "--pnr-timeout-seconds",
                "900",
            ]
        )
        == 1
    )
    assert "cannot read IR graph" in capsys.readouterr().out
    lif = tmp_path / "lif.sc"
    builder = ScGraphBuilder("lif_graph")
    current, leak, gain, noise = (builder.input(n, "i16") for n in ("i", "leak", "gain", "n"))
    builder.output("spike", builder.lif_step(current, leak, gain, noise))
    lif.write_text(builder.build().to_text(), encoding="utf-8")
    assert (
        main(
            [
                "evidence-packet",
                "export",
                str(lif),
                "-o",
                str(tmp_path / "b"),
                "--pnr-timeout-seconds",
                "900",
            ]
        )
        == 1
    )
    assert "sc.lif_step" in capsys.readouterr().out
    dense = tmp_path / "dense.sc"
    dense.write_text(dense_text(*DESIGNS["2x2"]), encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert (
        main(
            [
                "evidence-packet",
                "export",
                str(dense),
                "-o",
                str(tmp_path / "c"),
                "--pnr-timeout-seconds",
                "900",
            ]
        )
        == 1
    )
    assert "Icarus Verilog" in capsys.readouterr().out
    assert not any((tmp_path / name).exists() for name in ("a", "b", "c"))
    with pytest.raises(SimulatorUnavailableError):
        export_evidence_packet(
            dense.read_text(encoding="utf-8"), tmp_path / "d", pnr_timeout_s=TOOL_TIMEOUT_S
        )
