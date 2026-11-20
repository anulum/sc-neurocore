# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded proofs of real user IR graphs

"""Exercise the public IR proof path with real native emission and solver runs."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
from pathlib import Path

import pytest

from sc_neurocore.cli import main
from sc_neurocore.compiler.equivalence_check import formal_tools_available, prove_equivalence
from sc_neurocore.compiler.ir_equivalence import check_graph_equivalence
from sc_neurocore.studio.resource_feedback import with_instanced_primitives
from tests.engine_requirement import require_engine

require_engine()

_DUAL_ENCODER = """sc.graph @dual_encoder {
  %0 = sc.input "first_dut" : rate
  %1 = sc.input "rst_cnt" : rate
  %2 = sc.encode %0, length=16, seed=0xACE1 : bitstream<16>
  %3 = sc.encode %1, length=16, seed=0xACE1 : bitstream<16>
  sc.output "first" %2
  sc.output "second" %3
}
"""

_LIF = """sc.graph @lif_step_refractory {
  %0 = sc.input "current" : fixed<16,8>
  %1 = sc.input "leak" : fixed<16,8>
  %2 = sc.input "gain" : fixed<16,8>
  %3 = sc.input "noise" : fixed<16,8>
  %4 = sc.lif_step %0, leak=%1, gain=%2, noise=%3, dw=16, frac=8, vr=0, reset=0, vt=256, rp=2 : (bool, fixed<16,8>)
  sc.output "spike" %4
}
"""

_DCLS = """sc.graph @dcls_delay {
  %0 = sc.input "spike_in" : bool
  %1 = sc.constant [256, 128, -64] : vec<fixed<16,8>,3>
  %2 = sc.constant 256 : fixed<16,8>
  %3 = sc.constant 512 : fixed<16,8>
  %4 = sc.dcls_layer %0, weights=%1, centre=%2, sigma=%3, taps=3, depth=4, dw=16, frac=8, ptr=2, offsets=[0;1;2] : fixed<16,8>
  sc.output "weighted" %4
}
"""

_COMPOSED = _DCLS.replace(
    'sc.output "weighted" %4',
    "%5 = sc.encode %4, length=16, seed=0xACE1 : bitstream<16>\n"
    '  sc.output "encoded" %5\n'
    '  sc.output "weighted" %4',
)

_CASES = (
    (
        _DUAL_ENCODER,
        10,
        5,
        ".x_value(encode_0_probability)",
        ".x_value(encode_1_probability)",
    ),
    (_LIF, 5, 5, ".noise_in(noise)", ".noise_in(16'sd0)"),
    (
        _DCLS,
        12,
        6,
        ".tap_offsets({2'd2, 2'd1, 2'd0})",
        ".tap_offsets({2'd1, 2'd2, 2'd0})",
    ),
    (
        _DCLS.replace("%1 = sc.constant [256, 128, -64]", '%1 = sc.input "weights"'),
        7,
        6,
        ".tap_offsets({2'd2, 2'd1, 2'd0})",
        ".tap_offsets({2'd1, 2'd2, 2'd0})",
    ),
    (
        _DCLS.replace("offsets=[0;1;2]", "offsets=[0;0;0]"),
        4,
        4,
        "assign weighted = v4;",
        "assign weighted = ~v4;",
    ),
    (_COMPOSED, 8, 7, "assign encoded = v5;", "assign encoded = ~v5;"),
)


@pytest.fixture
def proof_tools() -> None:
    """Use the installed formal toolchain and require it on the CI proof lane."""
    if formal_tools_available("yosys-abc"):
        return
    if os.environ.get("CI") == "true":
        pytest.fail("CI must install SymbiYosys, Yosys and ABC for graph equivalence")
    pytest.skip("the local optional formal toolchain is unavailable")


@pytest.mark.parametrize(("text", "depth", "minimum", "old", "new"), _CASES)
def test_public_proof_and_real_wiring_counterexample(
    proof_tools: None,
    tmp_path: Path,
    text: str,
    depth: int,
    minimum: int,
    old: str,
    new: str,
) -> None:
    """Bind the CLI result to emitted RTL and detect one changed output dependency."""
    from sc_neurocore_engine import parse_ir

    graph = parse_ir(text)
    source = tmp_path / "graph.scir"
    source.write_text(text, encoding="utf-8")
    output = tmp_path / "build"
    assert (
        main(["compile-ir", str(source), "-o", str(output), "--equivalence-depth", str(depth)]) == 0
    )
    rtl = (output / f"{graph.name}.sv").read_text(encoding="utf-8")
    summary = json.loads((output / f"{graph.name}.guarantees.json").read_bytes())
    record = summary["graph_equivalence"]
    assert (record["verdict"], record["depth"], record["minimum_depth"]) == ("PASS", depth, minimum)
    assert record["reset_cycles"] == 2
    assert record["flattened"] is True
    assert record["engine"] == "abc bmc3"
    assert summary["ir_sha256"] == hashlib.sha256(graph.to_text().encode()).hexdigest()
    assert summary["sv_sha256"] == hashlib.sha256(rtl.encode()).hexdigest()
    expanded = with_instanced_primitives(rtl)
    assert record["dut_sha256"] == hashlib.sha256(expanded.encode()).hexdigest()
    description = (output / f"{graph.name}.guarantees.txt").read_text(encoding="utf-8")
    assert f"Bounded graph equivalence PASS at depth {depth}" in description

    assert rtl.count(old) == 1
    mutated = rtl.replace(old, new, 1)
    result = check_graph_equivalence(graph.name, graph.to_text(), mutated, depth=depth)
    assert result.verdict == "FAIL" and not result.proven
    assert result.counterexample
    assert result.reference_sha256 == record["reference_sha256"]


@pytest.mark.parametrize(
    ("text", "depth", "message"),
    (
        (_LIF, 4, "too short"),
        (_DCLS, 5, "need 6"),
        (_COMPOSED, 6, "need 7"),
        (_DUAL_ENCODER.replace("0xACE1", "0xBEEF", 1), 10, "formal parameters"),
        (
            'sc.graph @wire_only {\n  %0 = sc.input "wire_in" : bool\n'
            '  sc.output "wire_out" %0\n}\n',
            10,
            "observable formal operation",
        ),
    ),
)
def test_rejects_nonobservable_or_unqualified_proofs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str, depth: int, message: str
) -> None:
    """A refused proof writes no summary with a passing or ambiguous verdict."""
    source = tmp_path / "graph.scir"
    source.write_text(text, encoding="utf-8")
    output = tmp_path / "build"
    assert (
        main(["compile-ir", str(source), "-o", str(output), "--equivalence-depth", str(depth)]) == 1
    )
    assert message in capsys.readouterr().out
    assert not output.exists()


def test_missing_tool_writes_an_explicit_unavailable_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An optional solver is only needed when the caller requests the proof."""
    monkeypatch.setenv("PATH", "")
    source = tmp_path / "graph.scir"
    source.write_text(_DUAL_ENCODER, encoding="utf-8")
    output = tmp_path / "build"
    args = ["compile-ir", str(source), "-o", str(output)]
    record = output / "dual_encoder.guarantees.json"
    assert main(args) == 0
    assert "graph_equivalence" not in json.loads(record.read_bytes())
    assert main([*args, "--equivalence-depth", "10"]) == 1
    outcome = json.loads(record.read_bytes())["graph_equivalence"]
    assert outcome["verdict"] == "UNAVAILABLE"
    assert "yosys-abc" in outcome["reason"]


def test_external_deadline_writes_an_inconclusive_verdict(
    proof_tools: None, tmp_path: Path
) -> None:
    """An actual solver invocation that exceeds its caller's budget cannot pass."""
    source = tmp_path / "graph.scir"
    source.write_text(_DUAL_ENCODER, encoding="utf-8")
    output = tmp_path / "build"
    args = [
        "compile-ir",
        str(source),
        "-o",
        str(output),
        "--equivalence-depth",
        "10",
        "--equivalence-timeout",
        "0.000001",
    ]
    assert main(args) == 1
    summary = json.loads((output / "dual_encoder.guarantees.json").read_bytes())
    outcome = summary["graph_equivalence"]
    assert outcome["verdict"] == "INCONCLUSIVE"
    assert "timed out" in outcome["reason"]
    assert outcome["counterexample"] is None


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (["--equivalence-timeout", "12"], "requires --equivalence-depth"),
        (["--equivalence-depth", "10", "--equivalence-timeout", "nan"], "positive and finite"),
    ),
)
def test_cli_rejects_invalid_proof_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], arguments: list[str], message: str
) -> None:
    """Reject orphaned and invalid proof settings before writing any artefacts."""
    source = tmp_path / "graph.scir"
    source.write_text(_DUAL_ENCODER, encoding="utf-8")
    output = tmp_path / "build"
    assert main(["compile-ir", str(source), "-o", str(output), *arguments]) == 1
    assert message in capsys.readouterr().out
    assert not output.exists()


def test_reference_checks_exact_emitted_interface() -> None:
    """Decline a reserved module or a corrupted port even before running the solver."""
    from sc_neurocore_engine import parse_ir

    graph = parse_ir(_DUAL_ENCODER)
    text = graph.to_text()
    verilog = graph.emit_sv()
    changed = verilog.replace("output wire first", "output wire wrongly_named", 1)
    assert changed != verilog
    with pytest.raises(ValueError, match="independent graph interface"):
        check_graph_equivalence(graph.name, text, changed, depth=10)
    reserved = parse_ir(_DUAL_ENCODER.replace("@dual_encoder", "@equiv_miter", 1))
    with pytest.raises(ValueError, match="unreserved SystemVerilog module name"):
        check_graph_equivalence(reserved.name, reserved.to_text(), reserved.emit_sv(), depth=10)


def test_reference_keeps_structural_outputs_in_the_public_proof(
    proof_tools: None, tmp_path: Path
) -> None:
    """Check a packed fixed-point constant and an integer wire alongside a real encoder."""
    from sc_neurocore_engine import parse_ir

    source = tmp_path / "graph.scir"
    source.write_text(
        """sc.graph @_reference_boundary {
  %0 = sc.input "probability" : rate
  %1 = sc.input "word_in" : u8
  %2 = sc.constant [1.5, -0.25] : vec<fixed<16,8>,2>
  %3 = sc.encode %0, length=16, seed=0xACE1 : bitstream<16>
  sc.output "encoded" %3
  sc.output "word_out" %1
  sc.output "packed_out" %2
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "build"
    assert main(["compile-ir", str(source), "-o", str(output), "--equivalence-depth", "5"]) == 0
    graph = parse_ir(source.read_text(encoding="utf-8"))
    verilog = (output / f"{graph.name}.sv").read_text(encoding="utf-8")
    fault = verilog.replace("assign packed_out = c2;", "assign packed_out = {c2[15:0], c2[31:16]};")
    assert fault != verilog
    outcome = check_graph_equivalence(graph.name, graph.to_text(), fault, depth=5)
    assert outcome.verdict == "FAIL"
    assert outcome.counterexample


def test_invalid_emitted_design_never_gets_a_proof(proof_tools: None) -> None:
    """An actual Yosys parser error yields an inconclusive result on the public API."""
    from sc_neurocore_engine import parse_ir

    graph = parse_ir(_DUAL_ENCODER)
    verilog = graph.emit_sv().replace("endmodule", "assign first = ;\nendmodule", 1)
    outcome = check_graph_equivalence(graph.name, graph.to_text(), verilog, depth=10)
    assert outcome.verdict == "INCONCLUSIVE"
    assert outcome.reason is not None
    assert outcome.counterexample is None


@pytest.mark.parametrize(
    ("graph_text", "old", "new", "diagnostic"),
    (
        (_DUAL_ENCODER, " : rate", " : f32", "unsupported IR equivalence type"),
        (_DCLS, "[256, 128, -64]", "[256, 128]", "does not match its declared shape"),
        (_DUAL_ENCODER, '"first_dut"', '"first.dut"', "requires plain SystemVerilog port names"),
        (
            _DUAL_ENCODER,
            '  sc.output "first" %2',
            '  unfamiliar record\n  sc.output "first" %2',
            "cannot read canonical IR equivalence operation",
        ),
        (_DUAL_ENCODER, '"first_dut"', '"rst_n"', "needs distinct port names"),
    ),
)
def test_public_checker_rejects_damaged_ir_records(
    graph_text: str, old: str, new: str, diagnostic: str
) -> None:
    """Reject an invalid transported IR record even with the valid emitted design."""
    from sc_neurocore_engine import parse_ir

    graph = parse_ir(graph_text)
    canonical = graph.to_text()
    corrupt = canonical.replace(old, new, 1)
    assert corrupt != canonical
    with pytest.raises(ValueError, match=diagnostic):
        check_graph_equivalence(graph.name, corrupt, graph.emit_sv(), depth=12)


def test_native_graph_with_no_independent_model_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A compilable stochastic XOR cannot borrow the proofs of its two encoders."""
    source = tmp_path / "graph.scir"
    source.write_text(
        _DUAL_ENCODER.replace(
            '  sc.output "first" %2\n  sc.output "second" %3',
            '  %4 = sc.xor %2, %3 : bitstream<16>\n  sc.output "first" %2\n  sc.output "second" %4',
        ),
        encoding="utf-8",
    )
    output = tmp_path / "build"
    command = ["compile-ir", str(source), "-o", str(output)]
    assert main(command) == 0
    assert main([*command, "-o", str(tmp_path / "proof"), "--equivalence-depth", "10"]) == 1
    assert "sc.xor has no independent IR equivalence reference" in capsys.readouterr().out
    assert not (tmp_path / "proof").exists()


def test_abc_rejects_the_unbounded_proof_mode(tmp_path: Path) -> None:
    """Reject an unsupported ABC check before generating a misleading solver job."""
    from sc_neurocore_engine import parse_ir

    graph = parse_ir(_DUAL_ENCODER)
    verilog = graph.emit_sv()
    reference = verilog.replace(graph.name, "reference_encoder")
    workdir = tmp_path / "proof"
    with pytest.raises(ValueError, match="ABC bounded checking requires BMC mode"):
        prove_equivalence(
            verilog,
            reference,
            [],
            dut_top=graph.name,
            ref_top="reference_encoder",
            workdir=workdir,
            mode="prove",
            abc_bmc=True,
        )
    assert not workdir.exists()


def test_real_solver_pass_with_failed_process_is_inconclusive(
    proof_tools: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run ABC and inject a failing supervisor exit after the real solver reports PASS."""
    from sc_neurocore_engine import parse_ir

    sby = shutil.which("sby")
    assert sby is not None
    wrapper = tmp_path / "sby"
    wrapper.write_text(
        f'#!/bin/sh\n{shlex.quote(sby)} "$@" || exit "$?"\nexit 65\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(tmp_path), os.environ["PATH"])))
    graph = parse_ir(_DCLS.replace("offsets=[0;1;2]", "offsets=[0;0;0]"))
    result = check_graph_equivalence(graph.name, graph.to_text(), graph.emit_sv(), depth=4)
    assert result.verdict == "INCONCLUSIVE"
    assert not result.proven
    assert result.reason == "solver returned PASS with exit status 65"


@pytest.mark.parametrize(
    ("original", "replacement", "diagnostic"),
    (
        ("[15:0] first_dut", "[+15:0] first_dut", "unsupported width expression"),
        ("[15:0] first_dut", "[$clog2(65536)-1:0] first_dut", "unparsable width expression"),
        (
            "module dual_encoder (",
            "module dual_encoder #(parameter DATA_WIDTH = (16) (",
            "unterminated port list",
        ),
    ),
)
def test_actual_emitted_interface_refuses_unsupported_header(
    original: str, replacement: str, diagnostic: str
) -> None:
    """Refuse to prove RTL when a modified top-level header cannot be read safely."""
    from sc_neurocore_engine import parse_ir

    graph = parse_ir(_DUAL_ENCODER)
    emitted = graph.emit_sv()
    assert emitted.count(original) == 1
    changed = emitted.replace(original, replacement, 1)
    with pytest.raises(ValueError, match=diagnostic):
        check_graph_equivalence(graph.name, graph.to_text(), changed, depth=10)
