# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dense layer co-simulation against the emitted RTL

"""Bind the dense layer reference to the emitted RTL, the neuron models and the IR text.

Real dense graphs are built with the engine, emitted, and simulated with Icarus
Verilog against the Python reference on N inputs by M neurons, several state widths,
both current endpoint orders and two weight formats. Planted faults in the reference
configuration must be found at a definite cycle, and every graph outside the one
supported shape must be refused, naming its unsupported operations.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder  # noqa: E402

from sc_neurocore.safety_cert.dense_cosim import (  # noqa: E402
    PRIMITIVES,
    SCHEMA,
    TESTBENCH_MODULE,
    CosimError,
    CosimResult,
    DenseDesign,
    ScalarFormat,
    SimulatorUnavailableError,
    TraceRow,
    UnsupportedDesignError,
    cosimulate,
    default_stimulus,
    dotproduct_current,
    first_difference,
    lfsr_step,
    lif_step,
    parse_dense_design,
    probability_threshold,
    reference_trace,
    run_rtl,
    generate_testbench,
)
from sc_neurocore.neurons.fixed_point_lif import (  # noqa: E402
    FixedPointBitstreamEncoder,
    FixedPointLFSR,
    FixedPointLIFNeuron,
)

ROOT = Path(__file__).resolve().parents[1]


def dense_graph(
    n_inputs: int,
    n_neurons: int,
    width: int,
    weights: list[float],
    *,
    weight_type: str | None = None,
    y_min: int = 0,
    y_max: int = 300,
    seed: int = 0x1234,
    name: str = "dense_cosim",
) -> Any:
    """Build a real dense layer graph with the engine's public builder.

    Parameters
    ----------
    n_inputs, n_neurons, width : int
        Geometry and state width.
    weights : list of float
        Row-major weights.
    weight_type : str or None, optional
        Declared weight vector type; the builder's ``vec<rate,K>`` when omitted.
    y_min, y_max : int
        Current endpoints.
    seed : int
        Input seed base; the weight seed base is one more.
    name : str
        Graph name.

    Returns
    -------
    Any
        The built ``ScGraph``.
    """
    builder = ScGraphBuilder(name)
    inputs = builder.input("inputs", f"vec<rate,{n_inputs}>")
    if weight_type is None:
        matrix = builder.constant_f64_vec(weights)
    else:
        matrix = builder.constant_f64_vec(weights, weight_type)
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
        seed_base=seed,
        stream_length=7,
        y_min=y_min,
        y_max=y_max,
    )
    builder.output("spikes", result)
    graph = builder.build()
    assert graph.verify() is None
    return graph


def require_simulator() -> None:
    """Skip when Icarus Verilog is not installed, the only reason this module skips."""
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog (iverilog, vvp) is not installed; CI installs it")


DESIGNS = {
    "2x2 width 16, the RTL test's matrix": (2, 2, 16, [0.0, 0.0, 1.0, 1.0], None, 0, 300),
    "2x2 width 16, fractional weights": (2, 2, 16, [0.5, 0.25, 0.75, 0.9], None, 0, 300),
    "3x2 width 12": (3, 2, 12, [0.0, 0.25, 1.0, 0.5, -0.1, 0.75], None, 0, 300),
    "3x2 width 64, fixed<12,10> weights": (
        3,
        2,
        64,
        [0.5, 0.25, 1.0, 0.5, 0.9, 0.75],
        "vec<fixed<12,10>,6>",
        -200,
        900,
    ),
    "4x3 width 24, falling current range": (4, 3, 24, [0.3] * 12, None, 300, -31),
}


@pytest.mark.parametrize("label", sorted(DESIGNS))
def test_emitted_rtl_matches_the_reference_on_every_cycle(label: str) -> None:
    """Spikes, membranes and run control of the RTL equal the reference, cycle by cycle.

    Parameters
    ----------
    label : str
        Design under test.
    """
    require_simulator()
    n_inputs, n_neurons, width, weights, weight_type, y_min, y_max = DESIGNS[label]
    graph = dense_graph(
        n_inputs, n_neurons, width, weights, weight_type=weight_type, y_min=y_min, y_max=y_max
    )
    design = parse_dense_design(graph.to_text())
    assert (design.n_inputs, design.n_neurons, design.data_width) == (n_inputs, n_neurons, width)
    result = cosimulate(design, graph.emit_sv())
    assert result.agrees and result.first_difference is None
    assert len(result.rtl) == len(result.reference) == len(result.stimulus) == 256
    assert result.rtl == result.reference
    assert any(row.spikes for row in result.rtl)
    assert {row.run_done for row in result.rtl} == {0, 1}
    assert len({row.membranes for row in result.rtl}) > 2
    verdict = result.verdict()
    assert verdict["schema"] == SCHEMA and verdict["verdict"] == "AGREE"
    assert verdict["rows_at_first_difference"] is None
    assert set(verdict["primitives_sha256"]) == set(PRIMITIVES)
    assert verdict["simulator"]["iverilog"].startswith("Icarus Verilog")


@pytest.mark.parametrize(
    "fault",
    ["weight_seed_base", "input_seed_base", "weight_word", "leak", "y_max", "stream_length"],
)
def test_a_planted_fault_is_found_at_its_first_cycle(fault: str) -> None:
    """A reference configuration that differs from the emitted RTL is caught.

    Parameters
    ----------
    fault : str
        Parameter altered in the reference configuration only.
    """
    require_simulator()
    graph = dense_graph(3, 2, 16, [0.5, 0.25, 0.75, 0.9, 0.6, 0.4])
    design = parse_dense_design(graph.to_text())
    alternatives: dict[str, dict[str, Any]] = {
        "weight_seed_base": {"weight_seed_base": design.weight_seed_base + 1},
        "input_seed_base": {"input_seed_base": design.input_seed_base + 1},
        "weight_word": {"weight_words": (design.weight_words[0] // 2, *design.weight_words[1:])},
        "leak": {"leak": design.leak + 1},
        "y_max": {"y_max": design.y_max - 1},
        "stream_length": {"stream_length": design.stream_length + 1},
    }
    planted = dataclasses.replace(design, **alternatives[fault])
    result = cosimulate(planted, graph.emit_sv())
    cycle = result.first_difference
    assert cycle is not None and not result.agrees
    assert result.reference[:cycle] == result.rtl[:cycle]
    assert result.reference[cycle] != result.rtl[cycle]
    verdict = result.verdict()
    assert verdict["verdict"] == "DIFFER" and verdict["first_difference"] == cycle
    assert verdict["rows_at_first_difference"] == {
        "reference": result.reference[cycle].to_json(),
        "rtl": result.rtl[cycle].to_json(),
    }


def test_a_given_stimulus_is_simulated_as_given() -> None:
    """An explicit stimulus is used instead of the default one."""
    require_simulator()
    graph = dense_graph(2, 2, 16, [0.5, 0.25, 0.75, 0.9])
    design = parse_dense_design(graph.to_text())
    stimulus = [(256, 256)] * 12
    result = cosimulate(design, graph.emit_sv(), stimulus)
    assert result.stimulus == tuple(stimulus) and result.agrees and len(result.rtl) == 12


def test_the_reference_steps_like_the_package_neuron_and_encoder_models() -> None:
    """The layer's neuron and LFSR equal the package's bit-true models where both apply."""
    lfsr = FixedPointLFSR(seed=0xBEEF)
    state = 0xBEEF
    for _ in range(300):
        lfsr.step()
        state = lfsr_step(state)
        assert state == lfsr.reg
    encoder = FixedPointBitstreamEncoder(seed_init=0x1235)
    state = 0x1235
    for threshold in (0, 1, 40000, 65535, 32768) * 20:
        assert encoder.step(threshold) == int(state < threshold)
        state = lfsr_step(state)
    for width in (10, 16, 24, 32):
        neuron = FixedPointLIFNeuron(data_width=width, refractory_period=0)
        v = 0
        for step in range(400):
            current = ((step * 37) % 700) - 200
            leak, gain = (step % 50) - 10, 200 + step % 120
            spike, v = lif_step(v, current, leak, gain, width)
            assert (spike, v) == neuron.step(leak_k=leak, gain_k=gain, I_t=current)


@pytest.mark.parametrize(
    ("count", "n_inputs", "y_min", "y_max", "width", "current"),
    [
        (0, 3, 10, 300, 16, 10),
        (3, 3, 10, 300, 16, 300),
        (1, 3, 0, 300, 16, 100),
        (1, 3, 0, -301, 16, -100),
        (2, 3, 300, -31, 16, 80),
        (1, 2, -512, 511, 10, -1),
    ],
)
def test_the_current_truncates_toward_zero(
    count: int, n_inputs: int, y_min: int, y_max: int, width: int, current: int
) -> None:
    """``y_min + (y_max - y_min) * count / N`` divides toward zero, as Verilog does.

    Parameters
    ----------
    count, n_inputs, y_min, y_max, width : int
        Inputs of the reduction.
    current : int
        Expected current word.
    """
    assert dotproduct_current(count, n_inputs, y_min, y_max, width) == current


@pytest.mark.parametrize(
    ("word", "layout", "threshold"),
    [
        (0, ScalarFormat("rate", 16, 8, True), 0),
        (-64, ScalarFormat("rate", 16, 8, True), 0),
        (256, ScalarFormat("rate", 16, 8, True), 0xFFFF),
        (384, ScalarFormat("rate", 16, 8, True), 0xFFFF),
        (128, ScalarFormat("rate", 16, 8, True), 128 * 65535 >> 8),
        (127, ScalarFormat("fixed<8,7>", 8, 7, True), 127 * 65535 >> 7),
        (1, ScalarFormat("fixed<12,10>", 12, 10, True), 65535 >> 10),
    ],
)
def test_probability_words_become_encoder_thresholds(
    word: int, layout: ScalarFormat, threshold: int
) -> None:
    """Non-positive words never fire, words at one always clamp, the rest scale.

    Parameters
    ----------
    word : int
        Raw signed probability word.
    layout : ScalarFormat
        Its layout.
    threshold : int
        Expected encoder threshold.
    """
    assert probability_threshold(word, layout) == threshold


def test_parameters_record_every_seed_and_format() -> None:
    """The design states the formats and the seed of every one of its encoders."""
    graph = dense_graph(3, 2, 12, [0.0, 0.25, 1.0, 0.5, -0.1, 0.75], seed=0xFFFC)
    design = parse_dense_design(graph.to_text())
    parameters = design.to_parameters()
    assert parameters["seeds"]["input_encoders"] == [0xFFFC, 0x0003, 0x000A]
    assert parameters["seeds"]["weight_encoders"] == [
        [0xFFFD, (0xFFFD + 13) & 0xFFFF, (0xFFFD + 26) & 0xFFFF],
        [(0xFFFD + 39) & 0xFFFF, (0xFFFD + 52) & 0xFFFF, (0xFFFD + 65) & 0xFFFF],
    ]
    assert parameters["weights"]["words"] == [0, 64, 256, 128, -25, 192]
    assert parameters["weights"]["encoder_thresholds"] == [0, 16383, 65535, 32767, 0, 49151]
    assert parameters["q_formats"]["state"] == {"width": 12, "fraction": 8, "signed": True}
    assert parameters["q_formats"]["leak"]["type"] == "fixed<12,8>"
    assert (parameters["leak"], parameters["gain"], parameters["stream_length"]) == (16, 256, 7)
    assert parameters["instance"] == "sc_dense_matrix_layer u_dense_0"


def _text(
    *,
    inputs: str = '  %0 = sc.input "x" : vec<rate,2>\n',
    weights: str = "  %1 = sc.constant [0.5, 0.25, 1.0, 0.0] : vec<rate,4>\n",
    leak: str = "  %2 = sc.constant 16 : fixed<16,8>\n",
    gain: str = "  %3 = sc.constant 256 : fixed<16,8>\n",
    dense: str = "%4 = sc.dense_forward %0, weights=%1, leak=%2, gain=%3",
    output: str = '  sc.output "spikes" %4\n',
    name: str = "g",
) -> str:
    """Return the IR text of a dense graph with chosen lines replaced."""
    return (
        f"sc.graph @{name} {{\n{inputs}{weights}{leak}{gain}  {dense}, ni=2, nn=2, len=7, "
        "dw=16, input_seed=4660, weight_seed=4661, y_min=0, y_max=300 : vec<bool,2>\n"
        f"{output}}}\n"
    )


@pytest.mark.parametrize(
    ("leak", "gain", "words"),
    [
        ("16", "u64(256)", (16, 256)),
        ("i64(-3)", "1.5", (-3, 384)),
        ("0.0625", "256", (16, 256)),
    ],
)
def test_scalar_constants_are_read_as_the_ir_parser_reads_them(
    leak: str, gain: str, words: tuple[int, int]
) -> None:
    """Integer literals are raw words; float literals are quantised toward zero.

    Parameters
    ----------
    leak, gain : str
        Literals of the leak and gain constants.
    words : tuple of int
        Expected raw words.
    """
    design = parse_dense_design(
        _text(
            leak=f"  %2 = sc.constant {leak} : fixed<16,8>\n",
            gain=f"  %3 = sc.constant {gain} : fixed<16,8>\n",
        )
    )
    assert (design.leak, design.gain) == words


def test_rate_and_integer_typed_constants_use_their_layouts() -> None:
    """A rate literal is a float; integer vectors and ``iW`` words are raw words."""
    design = parse_dense_design(
        _text(
            weights="  %1 = sc.constant [128, 0, 256, -5] : vec<rate,4>\n",
            leak="  %2 = sc.constant 0.0625 : rate\n",
            gain="  %3 = sc.constant 256 : i16\n",
        )
    )
    assert design.weight_words == (128, 0, 256, -5)
    assert (design.leak, design.gain) == (16, 256)
    assert design.gain_format == ScalarFormat("i16", 16, 0, True)


def test_unsupported_operations_are_refused_by_name() -> None:
    """A graph with other operations is refused, and every such operation is named."""
    builder = ScGraphBuilder("lif_graph")
    current, leak, gain, noise = (builder.input(n, "i16") for n in ("i", "leak", "gain", "n"))
    builder.output("spike", builder.lif_step(current, leak, gain, noise))
    with pytest.raises(UnsupportedDesignError, match=r"sc\.lif_step") as raised:
        parse_dense_design(builder.build().to_text())
    assert raised.value.operations == ("sc.lif_step",)
    text = (
        'sc.graph @xor_stream {\n  %0 = sc.input "a" : rate\n  %1 = sc.input "b" : rate\n'
        "  %2 = sc.encode %0, length=64, seed=0xACE1 : bitstream<64>\n"
        "  %3 = sc.encode %1, length=64, seed=0xBEEF : bitstream<64>\n"
        '  %4 = sc.xor %2, %3 : bitstream\n  %5 = sc.decode %4 : rate\n  sc.output "y" %5\n}\n'
    )
    with pytest.raises(UnsupportedDesignError) as raised:
        parse_dense_design(text)
    assert raised.value.operations == ("sc.decode", "sc.encode", "sc.xor")
    assert "sc.decode, sc.encode, sc.xor" in str(raised.value)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            _text(inputs='  %0 = sc.input "x" : vec<rate,2>\n  %9 = sc.input "y" : rate\n'),
            "exactly one sc.input",
        ),
        (_text(output=""), "exactly one sc.input"),
        (_text(inputs='  %0 = sc.input "x" : vec<fixed<16,8>,2>\n'), "vec<rate,2>"),
        (
            _text(output='  sc.output "spikes" %4\n'.replace("%4", "%1")),
            "spikes of the sc.dense_forward",
        ),
        (
            _text(
                inputs='  %0 = sc.input "x" : vec<rate,2>\n',
                leak='  %2 = sc.input "leak" : fixed<16,8>\n',
            ),
            "exactly one sc.input",
        ),
        (_text(dense="%4 = sc.dense_forward %0, weights=%1, leak=%7, gain=%3"), "leak operand"),
        (_text(name=TESTBENCH_MODULE), "co-simulation module name"),
        (_text(name="sc_lif_neuron"), "co-simulation module name"),
        (_text(weights="  %1 = sc.constant 0.5 : rate\n"), "constant vector"),
        (_text(leak="  %2 = sc.constant 16 : u16\n"), "type u16"),
    ],
)
def test_other_shapes_are_refused(text: str, message: str) -> None:
    """Only one rate-vector input feeding one layer whose spikes are the output is accepted.

    Parameters
    ----------
    text : str
        IR text of a graph outside the supported shape.
    message : str
        Expected part of the refusal.
    """
    with pytest.raises(UnsupportedDesignError, match=re.escape(message)):
        parse_dense_design(text)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "sc.graph header"),
        ("module g;\n", "sc.graph header"),
        (_text().replace("ni=2", "ni=x"), "cannot read IR line"),
        (_text(leak="  %2 = sc.constant 40000 : fixed<16,8>\n"), "does not fit"),
        (_text(gain="  %3 = sc.constant 200.0 : fixed<16,8>\n"), "does not fit"),
    ],
)
def test_unreadable_text_is_refused(text: str, message: str) -> None:
    """Text that is not a readable graph, or a constant outside its word, is refused.

    Parameters
    ----------
    text : str
        IR text.
    message : str
        Expected part of the refusal.
    """
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_dense_design(text)


def test_stimulus_shapes_are_checked() -> None:
    """A stimulus needs a cycle, and each cycle one word per input."""
    design = parse_dense_design(_text())
    with pytest.raises(ValueError, match="at least one cycle"):
        default_stimulus(design, cycles=0)
    with pytest.raises(ValueError, match="has 1 words, not 2"):
        reference_trace(design, [(0, 0), (5,)])
    with pytest.raises(ValueError, match="an empty one compares nothing"):
        cosimulate(design, "module unused; endmodule\n", stimulus=[])


def test_a_missing_simulator_is_an_error_not_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``iverilog`` and ``vvp`` on the search path no trace is produced.

    Parameters
    ----------
    tmp_path : Path
        An empty directory used as the whole search path.
    monkeypatch : pytest.MonkeyPatch
        Sets ``PATH`` for this test only.
    """
    graph = dense_graph(2, 2, 16, [0.5, 0.25, 0.75, 0.9])
    design = parse_dense_design(graph.to_text())
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(SimulatorUnavailableError, match="not installed"):
        cosimulate(design, graph.emit_sv())


def test_simulator_failures_are_errors(tmp_path: Path) -> None:
    """A top level that does not compile, or prints a malformed row, raises.

    Parameters
    ----------
    tmp_path : Path
        Work directories.
    """
    require_simulator()
    graph = dense_graph(2, 2, 16, [0.5, 0.25, 0.75, 0.9])
    design = parse_dense_design(graph.to_text())
    stimulus = default_stimulus(design, cycles=3)
    broken = tmp_path / "broken"
    broken.mkdir()
    with pytest.raises(CosimError, match="iverilog"):
        run_rtl(design, "module dense_cosim(input wire clk", stimulus, broken)
    noisy = graph.emit_sv().replace("endmodule", 'initial $display("ROW 7");\nendmodule')
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    with pytest.raises(CosimError, match="malformed row: ROW 7"):
        run_rtl(design, noisy, stimulus, malformed)


def test_the_testbench_drives_and_probes_the_emitted_ports() -> None:
    """The bench packs signed words, instantiates the graph module and probes each neuron."""
    graph = dense_graph(3, 2, 12, [0.0, 0.25, 1.0, 0.5, -0.1, 0.75])
    design = parse_dense_design(graph.to_text())
    bench = generate_testbench(design, [(-64, 384, 128)])
    assert f"module {TESTBENCH_MODULE};" in bench
    assert "reg [47:0] inputs;" in bench and "wire [1:0] spikes;" in bench
    assert "inputs = 48'h00800180ffc0;" in bench
    assert "dense_cosim dut(.clk(clk), .rst_n(rst_n), .inputs(inputs), .spikes(spikes));" in bench
    assert "$signed(dut.u_dense_0.NEURON[1].v_out_j)" in bench


def test_a_short_trace_differs_where_it_ends() -> None:
    """A missing row is a difference, and the verdict shows the row that exists."""
    row = TraceRow(0, 1, (5, 6), 1, 0)
    later = TraceRow(1, 0, (7, 8), 1, 0)
    assert first_difference([row, later], [row, later], 2) is None
    assert first_difference([row, later], [row], 2) == 1
    assert first_difference([row], [row], 2) == 1
    design: DenseDesign = parse_dense_design(_text())
    result = CosimResult(design, ((0, 0), (0, 0)), (row, later), (row,), 1, "")
    assert result.verdict()["rows_at_first_difference"] == {
        "reference": later.to_json(),
        "rtl": None,
    }
    assert not result.agrees
