# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Guarantee summary of emitted IR top levels

"""Bind the guarantee summary to the IR compiler, the tracked formal jobs and real graphs.

The classification table must cover every operation the IR prints, name exactly the
modules the emitter instantiates, and quote each formal job as it is committed. Real
graphs of every operation family are summarised and emitted, the ``compile-ir``
command writes the summary beside the top level, and where SymbiYosys and its solvers
are installed, every formal job the summary names must pass.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.cli import main
from sc_neurocore.compiler.guarantee_summary import (
    GUARANTEE_CLASSES,
    LIBRARY_MODULES,
    CosimEvidence,
    FormalEvidence,
    OPERATION_RULES,
    SCHEMA,
    parse_instances,
    render_guarantee_text,
    summarize_graph,
    summarize_ir_text,
    verilog_value,
    write_guarantee_summary,
)

ROOT = Path(__file__).resolve().parents[1]
IR = ROOT / "engine" / "src" / "ir"
_ASSERT = re.compile(r"\bassert\s*(?:property)?\s*\(")
_ASSUME = re.compile(r"\bassume\s*(?:property)?\s*\(")


def _sby_sections(job: Path) -> dict[str, list[str]]:
    """Return the non-empty lines of each section of a SymbiYosys job.

    Parameters
    ----------
    job : Path
        The ``.sby`` file.

    Returns
    -------
    dict
        Section name to its lines, comments removed.
    """
    sections: dict[str, list[str]] = {}
    current = ""
    for raw in job.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line)
    return sections


def _tracked(path: str) -> bool:
    """Tell whether a repository path is tracked by Git."""
    listing = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", path],
        capture_output=True,
        check=False,
    )
    return listing.returncode == 0


def _without_comments(source: str) -> str:
    """Return Verilog source with line comments removed."""
    return "\n".join(line.split("//", 1)[0] for line in source.splitlines())


def _compact(value: str) -> str:
    """Return a parameter value without whitespace."""
    return "".join(value.split())


def _harness_parameters(harness: str, module: str) -> dict[str, str]:
    """Return the parameters the harness instantiates its module with."""
    (instance,) = [
        parameters
        for name, parameters in parse_instances(_without_comments(harness))
        if name == module
    ]
    return instance


def _module_defaults(source: str) -> dict[str, str]:
    """Return the parameter defaults of a module declaration, as written."""
    header = re.split(r"\)\s*\(", _without_comments(source).split("#(", 1)[1], maxsplit=1)[0]
    return {
        match.group(1): match.group(2).strip()
        for match in re.finditer(
            r"parameter\s+(?:integer\s+|signed\s+)*(?:\[[^\]]*\]\s*)?(\w+)\s*=\s*([^,]+)", header
        )
    }


def test_every_operation_the_ir_prints_has_a_classification() -> None:
    """The table covers exactly the keywords of the IR printer, one per operation variant."""
    printer = (IR / "printer.rs").read_text(encoding="utf-8")
    printed = set(re.findall(r"= sc\.([a-z_]+)", printer)) | {"output"}
    graph = (IR / "graph.rs").read_text(encoding="utf-8")
    body = graph.split("pub enum ScOp {", 1)[1].split("\n}\n", 1)[0]
    variants = re.findall(r"^    ([A-Z][A-Za-z]+)\s*[{(,]", body, re.MULTILINE)
    assert set(OPERATION_RULES) == printed
    assert len(variants) == len(OPERATION_RULES)


def test_the_named_modules_are_exactly_the_ones_the_emitter_instantiates() -> None:
    """Every instantiation in the emitter is classified, and nothing else is."""
    sources = [IR / "emit_sv.rs", *sorted((IR / "emit_sv").glob("*.rs"))]
    instantiated = {
        name
        for source in sources
        for name in re.findall(r"\"    (sc_[a-z0-9_]+) #\(", source.read_text(encoding="utf-8"))
    }
    named = {rule.module for rule in OPERATION_RULES.values() if rule.module is not None}
    assert instantiated == named == set(LIBRARY_MODULES)


@pytest.mark.parametrize("module", sorted(LIBRARY_MODULES))
def test_each_module_entry_matches_the_committed_evidence(module: str) -> None:
    """A FORMAL entry quotes its tracked job; a module without one has no job to quote.

    Parameters
    ----------
    module : str
        Library module name.
    """
    evidence = LIBRARY_MODULES[module].evidence
    own_job = ROOT / "hdl" / "formal" / f"{module}.sby"
    if isinstance(evidence, CosimEvidence):
        assert not own_job.exists(), f"{module} has a formal job; classify it FORMAL"
        test = (ROOT / evidence.test).read_text(encoding="utf-8")
        assert "emit_sv" in test and ("iverilog" in test or "verilator" in test)
        return
    assert evidence.job == f"hdl/formal/{module}.sby" and _tracked(evidence.job)
    sections = _sby_sections(ROOT / evidence.job)
    options = dict(line.split(None, 1) for line in sections["options"])
    assert options["mode"] == evidence.mode
    assert int(options["depth"]) == evidence.depth
    assert sections["engines"][0] == evidence.engine
    files = sections["files"]
    assert any(Path(name).name == f"{module}.v" for name in files)
    (harness_name,) = [name for name in files if Path(name).name.endswith("_formal.v")]
    harness = (ROOT / "hdl" / "formal" / Path(harness_name).name).read_text(encoding="utf-8")
    assert len(_ASSERT.findall(harness)) == evidence.assertions
    assumptions = tuple(line.strip() for line in harness.splitlines() if _ASSUME.search(line))
    assert assumptions == evidence.assumptions
    proved = _harness_parameters(harness, module)
    assert {k: _compact(v) for k, v in proved.items()} == {
        k: _compact(v) for k, v in evidence.parameters.items()
    }
    defaults = _module_defaults((ROOT / "hdl" / f"{module}.v").read_text(encoding="utf-8"))
    assert {k: _compact(v) for k, v in defaults.items()} == {
        k: _compact(v) for k, v in evidence.defaults.items()
    }
    assert set(evidence.parameters) == set(evidence.defaults)


def test_a_stream_of_unknown_length_is_reported_as_unknown() -> None:
    """IR text whose streams are not defined in it yields no length rather than a guess."""
    summary = summarize_ir_text(
        "partial",
        'sc.graph @partial {\n  %2 = sc.and %0, %1 : bitstream\n  sc.output "y" %2\n}\n',
        "",
    )
    (guarantee,) = summary["operations"][0]["guarantees"]
    assert guarantee["parameters"] == {"stream_lengths": []}
    assert "PROBABILISTIC (stream length unknown; no error bound established)" in (
        render_guarantee_text(summary)
    )


def test_a_co_simulated_module_is_rendered_with_its_test() -> None:
    """The text names the co-simulation test of a module without its own formal job."""
    text = render_guarantee_text(summarize_graph(_graph("dense")))
    assert "CO-SIMULATED ONLY tests/test_ir_dense_rtl.py (no receipt)" in text


def test_an_unclassified_operation_is_refused() -> None:
    """An operation keyword outside the table stops the summary instead of being guessed."""
    with pytest.raises(ValueError, match=r"sc\.teleport has no guarantee classification"):
        summarize_ir_text(
            "g", 'sc.graph @g {\n  %0 = sc.teleport %1 : rate\n  sc.output "y" %0\n}\n', ""
        )


def test_an_operation_without_its_instance_is_refused() -> None:
    """A module operation whose instance is missing from the top level is not guessed."""
    with pytest.raises(ValueError, match="no instance of sc_lif_neuron for sc.lif_step"):
        summarize_ir_text("g", "sc.graph @g {\n  %4 = sc.lif_step %0, %1, %2, %3 : bool\n}\n", "")


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("256", 256),
        ("16'sd256", 256),
        ("16'hACE1", 0xACE1),
        ("16'hace1", 0xACE1),
        ("4'b1010", 10),
        ("-8'd3", -3),
        ("1_000", 1000),
        ("(1 << FRACTION)", "(1<<FRACTION)"),
        ("12ab", "12ab"),
    ],
)
def test_verilog_parameter_values_compare_by_their_number(text: str, value: int | str) -> None:
    """Literals compare by value whatever their size or base; expressions compare as text.

    Parameters
    ----------
    text : str
        Parameter value as written.
    value : int or str
        Expected comparison value.
    """
    assert verilog_value(text) == value


def test_a_formal_module_at_its_proved_parameters_is_formal() -> None:
    """The LIF builder defaults equal the harness, so the instance is covered by the proof."""
    (entry,) = [
        e for e in summarize_graph(_graph("lif"))["operations"] if e["operation"] == "sc.lif_step"
    ]
    (guarantee,) = entry["guarantees"]
    assert guarantee["class"] == "FORMAL"
    assert guarantee["instance_parameters"]["V_THRESHOLD"] == "256"
    assert "differs" not in guarantee


@pytest.mark.parametrize(
    ("family", "operation", "differs"),
    [
        ("lif_threshold", "sc.lif_step", ["V_THRESHOLD"]),
        ("encoder", "sc.encode", ["SEED_INIT"]),
        ("dcls", "sc.dcls_layer", ["DELAY_DEPTH", "N_TAPS", "PTR_WIDTH"]),
    ],
)
def test_an_instance_with_other_parameters_is_not_formal(
    family: str, operation: str, differs: list[str]
) -> None:
    """A different threshold, seed or layer size is reported with both sets, never as FORMAL.

    Parameters
    ----------
    family : str
        Graph family with at least one instance outside the proved parameters.
    operation : str
        Operation whose last instance differs.
    differs : list of str
        Parameter names expected to differ.
    """
    entries = [
        e for e in summarize_graph(_graph(family))["operations"] if e["operation"] == operation
    ]
    guarantee = entries[-1]["guarantees"][0]
    assert guarantee["class"] == "PROVEN AT OTHER PARAMETERS"
    assert guarantee["differs"] == differs
    for name in differs:
        assert verilog_value(guarantee["proved_parameters"][name]) != verilog_value(
            guarantee["instance_parameters"][name]
        )


def _encoder(b: Any) -> None:
    """Encode, multiply, count and decode two stochastic streams."""
    x = b.input("x", "rate")
    y = b.input("y", "rate")
    sx = b.encode(x, 256, 0xACE1)
    sy = b.encode(y, 256, 0xBEEF)
    product = b.bitwise_and(sx, sy)
    b.output("count", b.popcount(product))
    b.output("rate", b.decode(product))


def _lif(b: Any) -> None:
    """Build one LIF neuron step."""
    current, leak, gain, noise = (b.input(name, "i16") for name in ("i", "leak", "gain", "noise"))
    b.output("spike", b.lif_step(current, leak, gain, noise))


def _lif_threshold(b: Any) -> None:
    """Build one LIF neuron step whose threshold differs from the proved harness."""
    current, leak, gain, noise = (b.input(name, "i16") for name in ("i", "leak", "gain", "noise"))
    b.output("spike", b.lif_step(current, leak, gain, noise, v_threshold=128))


def _dense(b: Any) -> None:
    """Build a dense spiking layer."""
    x = b.input("x", "vec<rate,3>")
    w = b.input("w", "vec<rate,21>")
    leak = b.input("leak", "i16")
    gain = b.input("gain", "i16")
    b.output("spikes", b.dense_forward(x, w, leak, gain, n_inputs=3, n_neurons=7))


def _graph_forward(b: Any) -> None:
    """Build a two-node graph aggregation."""
    features = b.constant_f64_vec([0.5, 0.25, 0.125, 0.0], "vec<fixed<24,16>,4>")
    adjacency = b.constant_f64_vec([1.0, 1.0, 0.0, 1.0], "vec<fixed<24,16>,4>")
    b.output("agg", b.graph_forward(features, adjacency, 2, 2))


def _softmax(b: Any) -> None:
    """Build a one-query softmax attention."""
    q = b.constant_f64_vec([0.5, 0.25], "vec<fixed<24,16>,2>")
    k = b.constant_f64_vec([0.25, 0.5, 0.5, 0.25], "vec<fixed<24,16>,4>")
    v = b.constant_f64_vec([1.0, 0.0, 0.0, 1.0], "vec<fixed<24,16>,4>")
    b.output("attn", b.softmax_attention(q, k, v, 2))


def _kuramoto(b: Any) -> None:
    """Build a two-oscillator Kuramoto step."""
    phases = b.constant_f64_vec([0.0, 1.0], "vec<fixed<24,16>,2>")
    omega = b.constant_f64_vec([1.0, 1.0], "vec<fixed<24,16>,2>")
    coupling = b.constant_f64_vec([0.0, 0.5, 0.5, 0.0], "vec<fixed<24,16>,4>")
    b.output("phases", b.kuramoto_step(phases, omega, coupling, 0.01))


def _register(b: Any) -> None:
    """Build a counter register with an inline add and a simulation check."""
    initial = b.constant_i64(0, "u8")
    one = b.constant_i64(1, "u8")
    count = b.state(initial, "u8")
    b.store(count, b.word_binary("add", count, one, "u8"))
    b.runtime_check(b.constant_i64(1, "bool"), "overflow")
    b.output("count", count)


def _affine(b: Any) -> None:
    """Build an inline scale, offset and constant division."""
    x = b.input("x", "rate")
    b.output("w", b.div_const(b.offset(b.scale(x, 0.5), 0.25), 4))


_TEXT_GRAPHS = {
    "dcls": (
        'sc.graph @dcls {\n  %0 = sc.input "spike" : bitstream<7>\n'
        '  %1 = sc.input "weights" : vec<fixed<16,8>,1>\n'
        '  %2 = sc.input "centre" : fixed<16,8>\n  %3 = sc.input "sigma" : fixed<16,8>\n'
        "  %4 = sc.dcls_layer %0, weights=%1, centre=%2, sigma=%3, taps=1, depth=7, dw=16, "
        'frac=8, ptr=3, offsets=[6] : fixed<16,8>\n  sc.output "result" %4\n}\n'
    ),
    "xor": (
        'sc.graph @xor_stream {\n  %0 = sc.input "a" : rate\n  %1 = sc.input "b" : rate\n'
        "  %2 = sc.encode %0, length=64, seed=0xACE1 : bitstream<64>\n"
        "  %3 = sc.encode %1, length=64, seed=0xBEEF : bitstream<64>\n"
        '  %4 = sc.xor %2, %3 : bitstream\n  %5 = sc.decode %4 : rate\n  sc.output "y" %5\n}\n'
    ),
    "reduce": (
        'sc.graph @reduce {\n  %0 = sc.input "value" : vec<rate,2>\n'
        '  %1 = sc.reduce %0, mode=sum : rate\n  sc.output "result" %1\n}\n'
    ),
}

_BUILT_GRAPHS: dict[str, Callable[[Any], None]] = {
    "encoder": _encoder,
    "lif": _lif,
    "lif_threshold": _lif_threshold,
    "dense": _dense,
    "graph_forward": _graph_forward,
    "softmax": _softmax,
    "kuramoto": _kuramoto,
    "register": _register,
    "affine": _affine,
}


def _graph(family: str) -> Any:
    """Build or parse the real IR graph of one operation family."""
    ir = pytest.importorskip("sc_neurocore_engine.ir", reason="Rust engine not built")
    if family in _TEXT_GRAPHS:
        return ir.parse_ir(_TEXT_GRAPHS[family])
    builder = ir.ScGraphBuilder(family)
    _BUILT_GRAPHS[family](builder)
    return builder.build()


def _expected_classes(operation: str) -> list[str]:
    """Return the classes the table assigns to one operation keyword."""
    rule = OPERATION_RULES[operation]
    classes: list[str] = []
    if rule.module is not None:
        evidence = LIBRARY_MODULES[rule.module].evidence
        classes.append("FORMAL" if isinstance(evidence, FormalEvidence) else "CO-SIMULATED ONLY")
    if rule.stochastic:
        classes.append("PROBABILISTIC")
    return classes or ["NONE"]


@pytest.mark.parametrize("family", sorted([*_BUILT_GRAPHS, *_TEXT_GRAPHS]))
def test_a_real_graph_of_each_operation_family_is_summarised(family: str) -> None:
    """Every operation of a verified, emitted graph is classified, and its module is emitted.

    Parameters
    ----------
    family : str
        Operation family whose real graph is built or parsed.
    """
    graph = _graph(family)
    assert graph.verify() is None
    verilog = graph.emit_sv()
    text = graph.to_text()
    summary = summarize_graph(graph)
    operations = [
        line.split("sc.", 1)[1].split()[0]
        for line in text.splitlines()
        if "sc." in line and not line.lstrip().startswith("sc.graph")
    ]
    assert [entry["operation"] for entry in summary["operations"]] == [
        f"sc.{operation}" for operation in operations
    ]
    for entry in summary["operations"]:
        keyword = entry["operation"].removeprefix("sc.")
        # Whether a formal module's instance is FORMAL depends on its parameters, which
        # the dedicated parameter tests pin; here both outcomes stand for the formal job.
        found = [
            "FORMAL" if g["class"] == "PROVEN AT OTHER PARAMETERS" else g["class"]
            for g in entry["guarantees"]
        ]
        assert found == _expected_classes(keyword)
        if entry["module"] is not None:
            assert f"    {entry['module']} #(" in verilog
    assert summary["schema"] == SCHEMA and summary["graph"] == graph.name
    assert summary["ir_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert summary["sv_sha256"] == hashlib.sha256(verilog.encode("utf-8")).hexdigest()
    assert sum(summary["classes"].values()) == sum(
        len(entry["guarantees"]) for entry in summary["operations"]
    )


def test_stochastic_entries_carry_the_stream_length_and_seed() -> None:
    """Encoders report their own length and seed; consumers inherit the stream length."""
    summary = summarize_graph(_graph("encoder"))
    by_id = {entry["id"]: entry for entry in summary["operations"]}
    encode = next(e for e in by_id.values() if e["operation"] == "sc.encode")
    formal, probabilistic = encode["guarantees"]
    assert formal["job"] == "hdl/formal/sc_bitstream_encoder.sby" and formal["depth"] == 20
    assert probabilistic["parameters"] == {"stream_lengths": [256], "seed": "0xACE1"}
    assert probabilistic["bound"] is None
    for operation in ("sc.and", "sc.popcount", "sc.decode"):
        entry = next(e for e in by_id.values() if e["operation"] == operation)
        (guarantee,) = entry["guarantees"]
        assert guarantee["parameters"]["stream_lengths"] == [256]
    assert summary["classes"] == {
        "FORMAL": 1,
        "PROVEN AT OTHER PARAMETERS": 1,
        "CO-SIMULATED ONLY": 0,
        "PROBABILISTIC": 5,
        "NONE": 4,
    }


def test_the_written_summary_matches_the_graph(tmp_path: Path) -> None:
    """The JSON beside the top level is the summary; the text names every operation."""
    graph = _graph("encoder")
    sv_path = tmp_path / "encoder.sv"
    sv_path.write_text(graph.emit_sv(), encoding="utf-8")
    json_path, text_path = write_guarantee_summary(graph, sv_path)
    assert json_path.name == "encoder.guarantees.json"
    assert text_path.name == "encoder.guarantees.txt"
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary == summarize_graph(graph)
    text = text_path.read_text(encoding="utf-8")
    assert text == render_guarantee_text(summary)
    assert "FORMAL hdl/formal/sc_bitstream_encoder.sby (prove, depth 20, smtbmc" in text
    assert "PROBABILISTIC (stream length 256, seed 0xACE1; no error bound established)" in text
    assert (
        text.rstrip()
        .splitlines()[-1]
        .startswith("Classes: FORMAL 1, PROVEN AT OTHER PARAMETERS 1,")
    )
    assert set(summary["classes"]) == set(GUARANTEE_CLASSES)


def test_compile_ir_writes_the_top_level_and_its_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``sc-neurocore compile-ir`` emits the graph and writes the summary beside it."""
    graph = _graph("lif")
    source = tmp_path / "lif.ir"
    source.write_text(graph.to_text(), encoding="utf-8")
    assert main(["compile-ir", str(source), "--output", str(tmp_path / "build")]) == 0
    build = tmp_path / "build"
    assert (build / "lif.sv").read_text(encoding="utf-8") == graph.emit_sv()
    summary = json.loads((build / "lif.guarantees.json").read_text(encoding="utf-8"))
    assert summary == summarize_graph(graph)
    assert "lif.guarantees.txt" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "cannot read IR graph"),
        ("not an IR graph", "is not valid IR"),
        (
            'sc.graph @bad {\n  %0 = sc.input "spike" : bool\n  %1 = sc.input "w" : fixed<16,8>\n'
            '  %2 = sc.input "c" : fixed<16,8>\n  %3 = sc.input "s" : fixed<16,8>\n'
            "  %4 = sc.dcls_layer %0, weights=%1, centre=%2, sigma=%3, taps=2, depth=8, dw=16, "
            'frac=8, ptr=3, offsets=[0;1] : fixed<16,8>\n  sc.output "y" %4\n}\n',
            "fails IR verification",
        ),
    ],
)
def test_compile_ir_refuses_unreadable_or_invalid_graphs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None, message: str
) -> None:
    """A missing file, unparsable text or a graph failing verification writes nothing.

    Parameters
    ----------
    tmp_path : Path
        Test directory.
    capsys : pytest.CaptureFixture[str]
        Captured output.
    content : str or None
        IR text to write, or ``None`` for a missing file.
    message : str
        Expected part of the refusal.
    """
    pytest.importorskip("sc_neurocore_engine.ir", reason="Rust engine not built")
    source = tmp_path / "graph.ir"
    if content is not None:
        source.write_text(content, encoding="utf-8")
    assert main(["compile-ir", str(source), "--output", str(tmp_path / "build")]) == 1
    assert message in capsys.readouterr().out
    assert not (tmp_path / "build").exists()


def _solver_missing(engine: str) -> str | None:
    """Return why a job's solver cannot run here, or ``None`` when it can."""
    if shutil.which("sby") is None:
        return "SymbiYosys is not installed"
    solver = engine.split()[1] if len(engine.split()) > 1 else "yices"
    binary = {"yices": "yices-smt2", "cvc5": "cvc5", "z3": "z3"}[solver]
    if shutil.which(binary) is None:
        return f"the {solver} solver is not installed"
    return None


@pytest.mark.parametrize(
    "module",
    sorted(
        name
        for name, entry in LIBRARY_MODULES.items()
        if isinstance(entry.evidence, FormalEvidence)
    ),
)
def test_every_formal_job_the_summary_names_passes(module: str, tmp_path: Path) -> None:
    """A FORMAL entry is only as good as its job, so each named job must pass.

    Parameters
    ----------
    module : str
        Library module with a formal job.
    tmp_path : Path
        Work directory for the SymbiYosys run.
    """
    evidence = LIBRARY_MODULES[module].evidence
    assert isinstance(evidence, FormalEvidence)
    reason = _solver_missing(evidence.engine)
    if reason is not None:
        pytest.skip(f"{reason}; CI's test job installs the solvers and runs this job")
    # The repository's own resolver gives the directory and argument each job runs with.
    located = subprocess.run(
        [sys.executable, "-B", str(ROOT / "tools/formal_job_inputs.py"), evidence.job],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    directory, argument = located.stdout.splitlines()
    run = subprocess.run(
        ["sby", "-f", "-d", str(tmp_path / module), argument],
        cwd=ROOT / directory,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert "DONE (PASS, rc=0)" in run.stdout, run.stdout[-2000:]
