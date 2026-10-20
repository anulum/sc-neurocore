# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public Dense graph to real RTL

"""Check public installed graph builders against the Python bit-true reference."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir
from sc_neurocore.hdl.resources import baseline_primitive_path
from sc_neurocore.neurons.fixed_point_lif import FixedPointBitstreamEncoder, FixedPointLIFNeuron


@pytest.mark.parametrize("width", [10, 16, 24, 32])
@pytest.mark.parametrize("y_min,y_max", [(0, 300), (300, -31)])
def test_public_dense_graph_matches_python_state_and_spikes(
    tmp_path: Path, width: int, y_min: int, y_max: int
) -> None:
    """Simulate real emitted matrix RTL using the wheel's packaged dependencies."""
    builder = ScGraphBuilder("dense_python")
    inputs = builder.input("inputs", "vec<rate,2>")
    weights = builder.constant_f64_vec([0.0, 0.0, 1.0, 1.0])
    leak = builder.constant_i64(16, f"fixed<{width},8>")
    gain = builder.constant_i64(256, f"fixed<{width},8>")
    result = builder.dense_forward(
        inputs,
        weights,
        leak,
        gain,
        n_inputs=2,
        n_neurons=2,
        data_width=width,
        seed_base=0x1234,
        stream_length=7,
        y_min=y_min,
        y_max=y_max,
    )
    builder.output("spikes", result)
    graph = builder.build()
    assert graph.verify() is None
    assert graph.num_inputs == 1 and graph.num_outputs == 1
    restored = parse_ir(graph.to_text())
    assert restored.emit_sv() == graph.emit_sv()
    source = tmp_path / "top.sv"
    source.write_text(restored.emit_sv(), encoding="utf-8")
    encoders = [FixedPointBitstreamEncoder(seed_init=0x1234 + i * 7) for i in range(2)]
    weight_encoders = [FixedPointBitstreamEncoder(seed_init=0x1235 + i * 13) for i in range(4)]
    neurons = [FixedPointLIFNeuron(data_width=width, refractory_period=0) for _ in range(2)]
    inputs_bits = [0, 0]
    weight_bits = [0, 0, 0, 0]
    expected: list[tuple[int, int, int]] = []
    drive: list[str] = []
    for cycle in range(256):
        words = [(-64, 384, 128, 256, 0, 192)[(cycle // 5 + i) % 6] for i in range(2)]
        spikes = 0
        for row, neuron in enumerate(neurons):
            count = sum(inputs_bits[i] & weight_bits[row * 2 + i] for i in range(2))
            numerator = (y_max - y_min) * count
            current = y_min + (abs(numerator) // 2) * (1 if numerator >= 0 else -1)
            spike, _ = neuron.step(leak_k=16, gain_k=256, I_t=current)
            spikes |= spike << row
        expected.append((spikes, neurons[0].v, neurons[1].v))
        inputs_bits = [
            encoder.step(max(0, min(65535, word * 65535 // 256)))
            for encoder, word in zip(encoders, words)
        ]
        weight_bits = [
            encoder.step(word) for encoder, word in zip(weight_encoders, [0, 0, 65535, 65535])
        ]
        packed = sum((word & 65535) << (16 * i) for i, word in enumerate(words))
        drive.append(
            f"inputs=32'h{packed:08x}; #2; clk=1; #2; "
            '$display("ROW %0d %0d %0d", spikes, '
            "$signed(dut.u_dense_0.NEURON[0].v_out_j), "
            "$signed(dut.u_dense_0.NEURON[1].v_out_j)); clk=0; #1;"
        )
    bench = tmp_path / "tb.sv"
    bench.write_text(
        "module tb; reg clk=0,rst_n=0; reg [31:0] inputs; wire [1:0] spikes; "
        "dense_python dut(.clk(clk),.rst_n(rst_n),.inputs(inputs),.spikes(spikes)); "
        "initial begin #1; clk=1; #1; clk=0; #1; clk=1; #1; clk=0; rst_n=1;\n"
        + "\n".join(drive)
        + "\n$finish; end endmodule\n",
        encoding="utf-8",
    )
    names = [
        "sc_bitstream_encoder.v",
        "sc_bitstream_synapse.v",
        "sc_dotproduct_to_current.v",
        "sc_lif_neuron.v",
        "sc_dense_matrix_layer.v",
    ]
    primitives = [str(baseline_primitive_path(name)) for name in names]
    binary = tmp_path / "sim.vvp"
    subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "tb",
            "-o",
            str(binary),
            str(source),
            str(bench),
            *primitives,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    run = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=30
    )
    actual = [
        tuple(int(word) for word in line.removeprefix("ROW ").split())
        for line in run.stdout.splitlines()
        if line.startswith("ROW ")
    ]
    assert actual == expected
    assert any(spikes in (1, 2) for spikes, _, _ in actual)


def test_dense_defaults_and_invalid_probability_shapes_are_checked() -> None:
    """Retain signed defaults and reject unusable matrices through public methods."""
    for weights_type, endpoint in [("vec<rate,4>", 256), ("rate", 256), ("vec<rate,4>", 65535)]:
        builder = ScGraphBuilder("dense_contract")
        x = builder.input("x", "vec<rate,2>")
        w = builder.input("w", weights_type)
        leak = builder.constant_i64(16, "fixed<16,8>")
        gain = builder.constant_i64(256, "fixed<16,8>")
        result = builder.dense_forward(x, w, leak, gain, n_inputs=2, n_neurons=2, y_max=endpoint)
        builder.output("spikes", result)
        graph = builder.build()
        if weights_type == "vec<rate,4>" and endpoint == 256:
            assert graph.verify() is None
            assert "y_max=256" in graph.to_text()
            graph.emit_sv()
        else:
            assert graph.verify()
            with pytest.raises(ValueError):
                graph.emit_sv()
