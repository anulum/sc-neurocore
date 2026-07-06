# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end co-simulation of the GraphForward IR lowering

"""End-to-end co-simulation of the ``sc.graph_forward`` IR lowering.

The IR graph is built through the Rust engine's Python bindings, emitted to
SystemVerilog with :meth:`ScGraph.emit_sv`, instantiated against the hand-written
``hdl/sc_graph_forward.v`` core, and simulated with Icarus Verilog. The observed
aggregate is compared bit-for-bit against a fixed-point oracle that mirrors the
emitter quantisation and the RTL datapath exactly, and against the ideal float
degree-normalised aggregation within fixed-point resolution.
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

import pytest

pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built", exc_type=ImportError)

from sc_neurocore_engine.ir import ScGraphBuilder

# Fixed-point contract baked into hdl/sc_graph_forward.v: signed Q8.16.
DATA_WIDTH = 24
FRACTION = 16
SCALE = 1 << FRACTION
ACC_WIDTH = 2 * DATA_WIDTH + 8
DEG_WIDTH = DATA_WIDTH + 8

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_HDL = REPO_ROOT / "hdl" / "sc_graph_forward.v"


def _to_signed(value: int, bits: int) -> int:
    """Reinterpret the low ``bits`` of ``value`` as a two's-complement integer."""
    value &= (1 << bits) - 1
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _quantise(value: float) -> int:
    """Quantise a real value into signed Q8.16 (matches emit_sv's graph_fixed)."""
    return round(value * SCALE)


def _div_trunc(a: int, b: int) -> int:
    """Signed integer division truncating toward zero (Verilog `/` semantics)."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def fixed_point_forward(
    features: list[float], adjacency: list[float], n_nodes: int, n_features: int
) -> list[int]:
    """Golden fixed-point aggregate mirroring the RTL bit-for-bit."""
    feat_q = [_quantise(x) for x in features]
    adj_q = [_quantise(x) for x in adjacency]
    out: list[int] = []
    for i in range(n_nodes):
        degree = _to_signed(sum(adj_q[i * n_nodes + j] for j in range(n_nodes)), DEG_WIDTH)
        for f in range(n_features):
            num = 0
            for j in range(n_nodes):
                num += adj_q[i * n_nodes + j] * feat_q[j * n_features + f]
            num = _to_signed(num, ACC_WIDTH)
            if degree == 0:
                quot = _to_signed(num >> FRACTION, ACC_WIDTH)
            else:
                quot = _to_signed(_div_trunc(num, degree), ACC_WIDTH)
            out.append(_to_signed(quot, DATA_WIDTH))
    return out


def _float_forward(
    features: list[float], adjacency: list[float], n_nodes: int, n_features: int
) -> list[float]:
    """Ideal degree-normalised aggregation: agg[i][f] = (Σ_j A[i][j]·X[j][f]) / deg[i]."""
    out: list[float] = []
    for i in range(n_nodes):
        degree = sum(adjacency[i * n_nodes + j] for j in range(n_nodes))
        for f in range(n_features):
            acc = sum(
                adjacency[i * n_nodes + j] * features[j * n_features + f] for j in range(n_nodes)
            )
            out.append(acc / degree if degree != 0.0 else acc)
    return out


def _build_emitted_sv(
    name: str, features: list[float], adjacency: list[float], n_nodes: int, n_features: int
) -> str:
    """Construct the IR graph via the engine bindings and emit SystemVerilog."""
    builder = ScGraphBuilder(name)
    feat_id = builder.constant_f64_vec(list(features), f"vec<fixed<24,16>,{n_nodes * n_features}>")
    adj_id = builder.constant_f64_vec(list(adjacency), f"vec<fixed<24,16>,{n_nodes * n_nodes}>")
    agg_id = builder.graph_forward(feat_id, adj_id, n_nodes, n_features)
    builder.output("agg_out", agg_id)
    graph = builder.build()
    assert graph.verify() is None
    return graph.emit_sv()


def _run_cosim(
    name: str,
    features: list[float],
    adjacency: list[float],
    n_nodes: int,
    n_features: int,
    tmp_path: Path,
) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for HDL simulation tests")

    total = n_nodes * n_features
    expected = fixed_point_forward(features, adjacency, n_nodes, n_features)
    emitted = _build_emitted_sv(name, features, adjacency, n_nodes, n_features)
    assert "sc_graph_forward" in emitted
    assert "no synthesizable RTL implementation yet" not in emitted

    def lit(value: int) -> str:
        return f"-24'sd{-value}" if value < 0 else f"24'sd{value}"

    checks = "\n".join(
        f"        if ($signed(agg_out[{(k + 1) * DATA_WIDTH - 1}:{k * DATA_WIDTH}])"
        f" !== {lit(expected[k])})\n"
        f'            $fatal(1, "idx {k}: got %0d want {expected[k]}",'
        f" $signed(agg_out[{(k + 1) * DATA_WIDTH - 1}:{k * DATA_WIDTH}]));"
        for k in range(total)
    )
    testbench = f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    wire signed [{total * DATA_WIDTH - 1}:0] agg_out;

    {name} dut (
        .clk(clk),
        .rst_n(rst_n),
        .agg_out(agg_out)
    );

    initial begin
        #1;
{checks}
        $display("PASS {name}");
        $finish(0);
    end
endmodule
"""
    top_path = tmp_path / f"{name}.v"
    sim_path = tmp_path / f"{name}.out"
    top_path.write_text(emitted + testbench)

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(top_path), str(GRAPH_HDL)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    sim_result = subprocess.run([vvp, str(sim_path)], capture_output=True, text=True, check=False)
    assert sim_result.returncode == 0, sim_result.stdout + sim_result.stderr
    assert "PASS" in sim_result.stdout, sim_result.stdout

    # The fixed-point aggregate must track the ideal float aggregation within resolution.
    float_agg = _float_forward(features, adjacency, n_nodes, n_features)
    tol = (n_nodes + 4) / SCALE
    for got_fixed, want_float in zip(expected, float_agg):
        got = got_fixed / SCALE
        assert math.isclose(got, want_float, rel_tol=2e-3, abs_tol=tol), (got, want_float)


def test_single_node_self_loop(tmp_path: Path) -> None:
    # One node, self-loop degree 1: aggregate is the node's own features.
    _run_cosim("graph_one", [0.25, -0.5], [1.0], 1, 2, tmp_path)


def test_two_node_mean_aggregation(tmp_path: Path) -> None:
    # Fully-connected 2-node graph, degree 2: each node sees the feature mean.
    _run_cosim(
        "graph_mean",
        [0.1, 0.2, 0.3, 0.4],
        [1.0, 1.0, 1.0, 1.0],
        2,
        2,
        tmp_path,
    )


def test_three_node_weighted_asymmetric(tmp_path: Path) -> None:
    _run_cosim(
        "graph_three",
        [1.0, -2.0, 0.5],
        [0.0, 0.5, 0.25, 0.5, 0.0, 0.5, 0.25, 0.5, 0.0],
        3,
        1,
        tmp_path,
    )


def test_isolated_node_zero_degree(tmp_path: Path) -> None:
    # A zero row-degree leaves the (zero) aggregate un-normalised, not a divide-by-zero.
    _run_cosim("graph_zero", [0.5, -0.5], [0.0, 0.0, 0.0, 0.0], 2, 1, tmp_path)


def test_emitted_sv_instantiates_core_with_baked_parameters(tmp_path: Path) -> None:
    emitted = _build_emitted_sv("graph_params", [0.1, 0.2, 0.3, 0.4], [1.0, 1.0, 1.0, 1.0], 2, 2)
    assert ".N_NODES(2)" in emitted
    assert ".N_FEATURES(2)" in emitted
    assert ".DATA_WIDTH(24)" in emitted
    assert ".FRACTION(16)" in emitted
    assert "wire signed [95:0]" in emitted
