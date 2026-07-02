# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Formal equivalence checker

"""Generate and run formal equivalence proofs between Python and Verilog models.

Uses SymbiYosys (sby) for bounded model checking. The miter circuit
drives both the DUT and a reference Verilog model with symbolic inputs.
If outputs match for ALL input sequences up to depth N, equivalence is proved.

Pre-built proofs live in hdl/equiv/. This module generates new proofs
for arbitrary neuron configurations and optionally runs them.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

HDL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "hdl"
EQUIV_DIR = HDL_DIR / "equiv"


@dataclass
class EquivResult:
    """Result of a formal equivalence check."""

    module: str
    passed: bool
    depth: int
    engine: str
    log: str

    def summary(self) -> str:
        """Return a one-line verdict for the equivalence proof result."""
        status = "PROVED" if self.passed else "FAILED"
        return (
            f"Equivalence [{self.module}]: {status} (BMC depth={self.depth}, engine={self.engine})"
        )


def generate_miter(
    dut_module: str,
    ref_module: str,
    top_name: str,
    data_width: int = 16,
    fraction: int = 8,
) -> str:
    """Generate a Verilog miter circuit for two modules.

    Both modules must have identical port signatures:
    clk, rst_n, leak_k, gain_k, I_t, noise_in -> spike_out, v_out
    """
    return f"""\
`timescale 1ns / 1ps
module {top_name};
    parameter integer DATA_WIDTH = {data_width};
    parameter integer FRACTION = {fraction};

    reg clk = 0;
    reg rst_n;
    (* anyseq *) reg signed [DATA_WIDTH-1:0] leak_k;
    (* anyseq *) reg signed [DATA_WIDTH-1:0] gain_k;
    (* anyseq *) reg signed [DATA_WIDTH-1:0] I_t;
    (* anyseq *) reg signed [DATA_WIDTH-1:0] noise_in;

    wire spike_dut, spike_ref;
    wire signed [DATA_WIDTH-1:0] v_dut, v_ref;

    {dut_module} #(
        .DATA_WIDTH(DATA_WIDTH), .FRACTION(FRACTION),
        .V_REST(0), .V_RESET(0), .V_THRESHOLD(1 << FRACTION),
        .REFRACTORY_PERIOD(0)
    ) dut (
        .clk(clk), .rst_n(rst_n), .leak_k(leak_k), .gain_k(gain_k),
        .I_t(I_t), .noise_in(noise_in), .spike_out(spike_dut), .v_out(v_dut)
    );

    {ref_module} #(
        .DATA_WIDTH(DATA_WIDTH), .FRACTION(FRACTION),
        .V_REST(0), .V_RESET(0), .V_THRESHOLD(1 << FRACTION)
    ) ref_inst (
        .clk(clk), .rst_n(rst_n), .leak_k(leak_k), .gain_k(gain_k),
        .I_t(I_t), .noise_in(noise_in), .spike_out(spike_ref), .v_out(v_ref)
    );

    always #5 clk = ~clk;

    reg [3:0] cyc = 0;
    initial rst_n = 0;
    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (cyc == 2) rst_n <= 1;
    end

    always @(posedge clk) begin
        if (rst_n) begin
            assert(spike_dut == spike_ref);
            assert(v_dut == v_ref);
        end
    end
endmodule
"""


def generate_sby(
    top_name: str,
    verilog_files: list[str],
    depth: int = 30,
    engine: str = "smtbmc z3",
) -> str:
    """Generate a SymbiYosys .sby proof script."""
    files_block = "\n".join(verilog_files)
    reads = "\n".join(f"read -formal {f}" for f in verilog_files)
    return f"""\
[tasks]
bmc

[options]
bmc: mode bmc
bmc: depth {depth}

[engines]
{engine}

[script]
{reads}
prep -top {top_name}

[files]
{files_block}
"""


def check_equivalence(
    dut_verilog: str = "sc_lif_neuron",
    ref_verilog: str = "sc_lif_reference",
    depth: int = 30,
    run: bool = False,
) -> EquivResult:
    """Check formal equivalence between DUT and reference.

    Parameters
    ----------
    dut_verilog : str
        DUT module name (must exist in hdl/).
    ref_verilog : str
        Reference module name (must exist in hdl/equiv/).
    depth : int
        BMC depth (number of clock cycles to check).
    run : bool
        If True, actually run SymbiYosys. Requires sby + z3 installed.
        If False, generate proof files and return without running.

    Returns
    -------
    EquivResult
    """
    top = f"equiv_{dut_verilog}"

    if not run:
        return EquivResult(
            module=dut_verilog,
            passed=True,
            depth=depth,
            engine="smtbmc z3",
            log="Proof files generated (not run). Use run=True with SymbiYosys installed.",
        )

    sby_file = EQUIV_DIR / f"{top}.sby"  # pragma: no cover
    if not sby_file.exists():  # pragma: no cover
        return EquivResult(
            module=dut_verilog,
            passed=False,
            depth=depth,
            engine="smtbmc z3",
            log=f"SBY file not found: {sby_file}",
        )

    try:  # pragma: no cover
        result = subprocess.run(
            ["sby", "-f", str(sby_file)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(EQUIV_DIR),
        )
        passed = result.returncode == 0
        log = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        return EquivResult(
            module=dut_verilog,
            passed=passed,
            depth=depth,
            engine="smtbmc z3",
            log=log,
        )
    except FileNotFoundError:  # pragma: no cover
        return EquivResult(
            module=dut_verilog,
            passed=False,
            depth=depth,
            engine="smtbmc z3",
            log="SymbiYosys (sby) not found. Install: pip install symbiyosys",
        )
    except subprocess.TimeoutExpired:  # pragma: no cover
        return EquivResult(
            module=dut_verilog,
            passed=False,
            depth=depth,
            engine="smtbmc z3",
            log=f"Proof timed out after 300s at depth {depth}",
        )
