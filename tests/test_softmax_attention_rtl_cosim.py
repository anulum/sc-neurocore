# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end co-simulation of the SoftmaxAttention IR lowering

"""End-to-end co-simulation of the ``sc.softmax_attention`` IR lowering.

The IR graph is built through the Rust engine's Python bindings, emitted to
SystemVerilog with :meth:`ScGraph.emit_sv`, instantiated against the hand-written
``hdl/sc_softmax_attention.v`` core, and simulated with Icarus Verilog. The observed
attention output is compared bit-for-bit against a fixed-point oracle that mirrors the
emitter quantisation and the RTL datapath (including the 256-entry exp LUT) exactly,
and against the ideal float ``softmax(Q·Kᵀ/√dim_k)·V`` within the exp-LUT resolution.
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

import pytest

pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built", exc_type=ImportError)

from sc_neurocore_engine.ir import ScGraphBuilder

# Fixed-point contract baked into hdl/sc_softmax_attention.v: signed Q8.16, 256-entry exp LUT.
DW = 24
FRAC = 16
SCALE = 1 << FRAC
CAP = (1 << (DW - 1)) - 1
SCORE_ACC_W = 2 * DW + 8
SCORE_MUL_W = SCORE_ACC_W + DW
SUM_W = DW + 8
OUT_ACC_W = 2 * DW + 8

LUT_N = 256
LUT_MIN = -16.0
LUT_STEP = 0.125
EXP_SHIFT = FRAC - round(-math.log2(LUT_STEP))  # 13
EXP_MIN_ABS = round(-LUT_MIN * SCALE)  # 1048576

REPO_ROOT = Path(__file__).resolve().parent.parent
ATTN_HDL = REPO_ROOT / "hdl" / "sc_softmax_attention.v"

_EXP_TABLE = [min(round(math.exp(LUT_MIN + i * LUT_STEP) * SCALE), CAP) for i in range(LUT_N)]


def _to_signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _quantise(value: float) -> int:
    return round(value * SCALE)


def _exp_lut(arg_q: int) -> int:
    argv = _to_signed(arg_q, DW)
    offset = _to_signed(argv + EXP_MIN_ABS, DW + 1)
    raw = _to_signed(offset >> EXP_SHIFT, DW + 1)
    idx = 0 if raw < 0 else (LUT_N - 1 if raw > LUT_N - 1 else raw)
    return _EXP_TABLE[idx]


def fixed_point_attention(
    q: list[float],
    k: list[float],
    v: list[float],
    q_rows: int,
    k_rows: int,
    dim_k: int,
    v_cols: int,
    inv_temp: float,
) -> list[int]:
    """Golden fixed-point attention output mirroring the RTL bit-for-bit."""
    q_q = [_quantise(x) for x in q]
    k_q = [_quantise(x) for x in k]
    v_q = [_quantise(x) for x in v]
    it_q = _quantise(inv_temp)
    out: list[int] = []
    for i in range(q_rows):
        score = []
        for j in range(k_rows):
            raw = sum(q_q[i * dim_k + d] * k_q[j * dim_k + d] for d in range(dim_k))
            raw = _to_signed(raw, SCORE_ACC_W)
            mul = _to_signed(raw * it_q, SCORE_MUL_W)
            score.append(_to_signed(mul >> (2 * FRAC), DW))
        mx = max(score)
        e = [_exp_lut(_to_signed(s - mx, DW)) for s in score]
        sum_e = _to_signed(sum(e), SUM_W)
        w = [_to_signed((e[j] << FRAC) // sum_e, DW) for j in range(k_rows)]
        for c in range(v_cols):
            acc = _to_signed(sum(w[j] * v_q[j * v_cols + c] for j in range(k_rows)), OUT_ACC_W)
            out.append(_to_signed(acc >> FRAC, DW))
    return out


def _float_attention(
    q: list[float],
    k: list[float],
    v: list[float],
    q_rows: int,
    k_rows: int,
    dim_k: int,
    v_cols: int,
    inv_temp: float,
) -> list[float]:
    """Ideal float softmax(Q·Kᵀ·inv_temp)·V, numerically stable."""
    out: list[float] = []
    for i in range(q_rows):
        scores = [
            sum(q[i * dim_k + d] * k[j * dim_k + d] for d in range(dim_k)) * inv_temp
            for j in range(k_rows)
        ]
        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        se = sum(exps)
        w = [e / se for e in exps]
        for c in range(v_cols):
            out.append(sum(w[j] * v[j * v_cols + c] for j in range(k_rows)))
    return out


def _build_emitted_sv(
    name: str,
    q: list[float],
    k: list[float],
    v: list[float],
    dim_k: int,
) -> str:
    """Construct the IR graph via the engine bindings and emit SystemVerilog."""
    builder = ScGraphBuilder(name)
    q_id = builder.constant_f64_vec(list(q), f"vec<fixed<24,16>,{len(q)}>")
    k_id = builder.constant_f64_vec(list(k), f"vec<fixed<24,16>,{len(k)}>")
    v_id = builder.constant_f64_vec(list(v), f"vec<fixed<24,16>,{len(v)}>")
    attn_id = builder.softmax_attention(q_id, k_id, v_id, dim_k)
    builder.output("attn_out", attn_id)
    graph = builder.build()
    assert graph.verify() is None
    return graph.emit_sv()


def _run_cosim(
    name: str,
    q: list[float],
    k: list[float],
    v: list[float],
    q_rows: int,
    k_rows: int,
    dim_k: int,
    v_cols: int,
    tmp_path: Path,
) -> list[int]:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for HDL simulation tests")

    inv_temp = 1.0 / math.sqrt(dim_k)
    total = q_rows * v_cols
    expected = fixed_point_attention(q, k, v, q_rows, k_rows, dim_k, v_cols, inv_temp)
    emitted = _build_emitted_sv(name, q, k, v, dim_k)
    assert "sc_softmax_attention" in emitted
    assert "no synthesizable RTL implementation yet" not in emitted

    def lit(value: int) -> str:
        return f"-24'sd{-value}" if value < 0 else f"24'sd{value}"

    checks = "\n".join(
        f"        if ($signed(attn_out[{(m + 1) * DW - 1}:{m * DW}]) !== {lit(expected[m])})\n"
        f'            $fatal(1, "idx {m}: got %0d want {expected[m]}",'
        f" $signed(attn_out[{(m + 1) * DW - 1}:{m * DW}]));"
        for m in range(total)
    )
    testbench = f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    wire signed [{total * DW - 1}:0] attn_out;

    {name} dut (
        .clk(clk),
        .rst_n(rst_n),
        .attn_out(attn_out)
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
        [iverilog, "-g2012", "-o", str(sim_path), str(top_path), str(ATTN_HDL)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    sim_result = subprocess.run([vvp, str(sim_path)], capture_output=True, text=True, check=False)
    assert sim_result.returncode == 0, sim_result.stdout + sim_result.stderr
    assert "PASS" in sim_result.stdout, sim_result.stdout

    # The fixed-point output must track the ideal float attention within the exp-LUT
    # resolution: floor-rounding the arg to the 0.125 grid perturbs each softmax weight
    # by up to ~2*(e^0.125-1); the output is a convex combination of the V rows.
    float_out = _float_attention(q, k, v, q_rows, k_rows, dim_k, v_cols, inv_temp)
    vmax = max((abs(x) for x in v), default=1.0)
    tol = 2.0 * (math.exp(LUT_STEP) - 1.0) * vmax + (k_rows + 4) / SCALE
    for got_q, want in zip(expected, float_out):
        assert abs(got_q / SCALE - want) <= tol, (got_q / SCALE, want, tol)
    return expected


def test_single_query_selects_matching_key(tmp_path: Path) -> None:
    # Sharp query aligned with key 0: softmax collapses onto the first V row.
    _run_cosim("attn_select", [1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [10.0, 0.0], 1, 2, 2, 1, tmp_path)


def test_uniform_scores_average_values(tmp_path: Path) -> None:
    # Zero query -> equal scores -> exp(0) exact -> weights are exactly 1/k_rows, so the
    # output is the mean of the V rows, free of any exp-LUT error.
    out = _run_cosim(
        "attn_uniform", [0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.0, -4.0], 1, 2, 2, 1, tmp_path
    )
    # mean(2, -4) = -1.0; integer-division weights land within a couple of LSB.
    assert abs(out[0] / SCALE - (-1.0)) <= 4.0 / SCALE


def test_two_query_two_value_columns(tmp_path: Path) -> None:
    _run_cosim(
        "attn_two",
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [5.0, 1.0, 1.0, 5.0],
        2,
        2,
        2,
        2,
        tmp_path,
    )


def test_three_key_asymmetric(tmp_path: Path) -> None:
    _run_cosim(
        "attn_three",
        [0.5, -0.5, 0.25],
        [0.5, -0.5, 0.25, 0.1, 0.2, -0.3, -0.4, 0.6, 0.1],
        [1.0, -1.0, 2.0],
        1,
        3,
        3,
        1,
        tmp_path,
    )


def test_emitted_sv_instantiates_core_with_baked_parameters(tmp_path: Path) -> None:
    emitted = _build_emitted_sv(
        "attn_params", [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [5.0, 1.0, 1.0, 5.0], 2
    )
    assert ".Q_ROWS(2)" in emitted
    assert ".K_ROWS(2)" in emitted
    assert ".DIM_K(2)" in emitted
    assert ".V_COLS(2)" in emitted
    assert ".INV_TEMP(24'sd46341)" in emitted
    assert ".EXP_SHIFT(13)" in emitted
    assert ".EXP_MIN_ABS(1048576)" in emitted
    assert "wire signed [95:0]" in emitted
