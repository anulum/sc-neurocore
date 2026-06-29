# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded datapath PE equivalence tests

"""The combinational datapath PE is bit-exact with the per-instance module.

A single testbench drives the registered per-instance module and the
combinational PE — wrapped with an external state register under the *same*
clock/reset — from the same current, and asserts the spike trains agree on every
cycle. This is the contract the folded interconnect relies on: time-multiplexing
the PE over BRAM-held state produces exactly what one module-per-neuron would.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.hdl_gen._ident import sanitize_ident
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

_HAVE_IVERILOG = shutil.which("iverilog") is not None and shutil.which("vvp") is not None
pytestmark = pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")

# (schema, input current, n_steps) — chosen so each model actually spikes.
_MODELS = [
    ("lif", 5.0, 200),
    ("izhikevich", 10.0, 200),
    ("theta", 2.0, 200),
    ("quadratic_if", 5.0, 200),
]
_DW, _FR = 16, 8


def _dual_testbench(neuron, inst_mod: str, pe_mod: str, current: float, n_steps: int) -> str:
    """Build a TB comparing the per-instance module to the externally-clocked PE."""
    q = Q88(data_width=_DW, fraction=_FR)
    i_val = q.encode_signed_literal(current)
    svars = [sanitize_ident(v, context="state variable") for v in neuron.equations]

    inst_ports = [
        "    .clk(clk),",
        "    .rst_n(rst_n),",
        f"    .I_t({i_val}),",
        "    .spike_out(inst_spike),",
    ]
    pe_ports = [f"    .I_t({i_val}),"]
    decls, regs, inits = [], [], []
    for sv in svars:
        inst_ports.append(f"    .{sv}_out({sv}_out),")
        pe_ports.append(f"    .{sv}_reg(pe_{sv}),")
        decls.append(f"wire signed [{_DW - 1}:0] {sv}_out;")
        decls.append(f"wire signed [{_DW - 1}:0] pe_{sv}_next;")
        regs.append(f"reg signed [{_DW - 1}:0] pe_{sv};")
        init_val = q.encode_signed_literal(neuron.initial_state.get(_unsanitised(neuron, sv), 0.0))
        inits.append((f"pe_{sv}", init_val, f"pe_{sv}_next"))
    inst_ports[-1] = inst_ports[-1].rstrip(",")
    for sv in svars:
        pe_ports.append(f"    .{sv}_next_out(pe_{sv}_next),")
    pe_ports.append("    .spike_out(pe_spike_comb)")

    lines = [
        "`timescale 1ns / 1ps",
        "module tb_dual;",
        "reg clk; reg rst_n;",
        "wire inst_spike;",
        "wire pe_spike_comb;",
        "reg pe_spike;",
        *decls,
        *regs,
        "",
        f"{inst_mod} uut (",
        *inst_ports,
        ");",
        "",
        f"{pe_mod} pe (",
        *pe_ports,
        ");",
        "",
        "initial clk = 0;",
        "always #5 clk = ~clk;",
        "",
        "always @(posedge clk or negedge rst_n) begin",
        "    if (!rst_n) begin",
        "        pe_spike <= 1'b0;",
        *[f"        {reg} <= {init};" for reg, init, _ in inits],
        "    end else begin",
        "        pe_spike <= pe_spike_comb;",
        *[f"        {reg} <= {nxt};" for reg, _, nxt in inits],
        "    end",
        "end",
        "",
        "integer mismatches; integer inst_spikes; integer k;",
        "initial begin",
        "    mismatches = 0; inst_spikes = 0;",
        "    rst_n = 0;",
        "    #20; rst_n = 1;",
        "    @(posedge clk);",
        f"    for (k = 0; k < {n_steps}; k = k + 1) begin",
        "        @(posedge clk); #1;",
        "        if (inst_spike !== pe_spike) mismatches = mismatches + 1;",
        "        if (inst_spike) inst_spikes = inst_spikes + 1;",
        "    end",
        '    $display("MISMATCHES=%0d SPIKES=%0d", mismatches, inst_spikes);',
        "    $finish;",
        "end",
        "endmodule",
    ]
    return "\n".join(lines)


def _unsanitised(neuron, safe_var: str) -> str:
    """Map a sanitised state-var name back to the original equation key."""
    for var in neuron.equations:
        if sanitize_ident(var, context="state variable") == safe_var:
            return var
    return safe_var


def _run_dual(schema: str, current: float, n_steps: int) -> tuple[int, int]:
    neuron = UniversalNeuron.from_schema(schema).to_equation_neuron()
    inst_mod, pe_mod = f"sc_{schema}", f"sc_{schema}_pe"
    inst_v = compile_to_verilog(neuron, module_name=inst_mod, data_width=_DW, fraction=_FR)
    pe_v = compile_to_datapath(neuron, module_name=pe_mod, data_width=_DW, fraction=_FR)
    tb = _dual_testbench(neuron, inst_mod, pe_mod, current, n_steps)
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "inst.v").write_text(inst_v)
        (dp / "pe.v").write_text(pe_v)
        (dp / "tb.v").write_text(tb)
        out = dp / "sim"
        comp = subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-o",
                str(out),
                str(dp / "inst.v"),
                str(dp / "pe.v"),
                str(dp / "tb.v"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert comp.returncode == 0, f"iverilog failed: {comp.stderr}"
        run = subprocess.run(["vvp", str(out)], capture_output=True, text=True, timeout=120)
        m = re.search(r"MISMATCHES=(\d+) SPIKES=(\d+)", run.stdout)
        assert m, f"no result parsed from: {run.stdout}"
        return int(m.group(1)), int(m.group(2))


@pytest.mark.parametrize(("schema", "current", "n_steps"), _MODELS)
def test_datapath_pe_matches_per_instance(schema: str, current: float, n_steps: int) -> None:
    mismatches, spikes = _run_dual(schema, current, n_steps)
    assert mismatches == 0, (
        f"{schema}: PE spike train diverged from per-instance ({mismatches} cycles)"
    )
    assert spikes > 0, f"{schema}: workload should spike (otherwise equivalence is vacuous)"
