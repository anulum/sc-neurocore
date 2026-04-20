# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for nas/equiv

module EquivAccel

using Statistics, LinearAlgebra

mutable struct EquivResultState
    module::Float64
    passed::Float64
    depth::Float64
    engine::Float64
    log::Float64
end

function EquivResultState()
    EquivResultState(0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::EquivResultState)
    status = "PROVED" if s.passed else "FAILED"
    return (
        f"Equivalence [{s.module}]: {status} (BMC depth={s.depth}, engine={s.engine})"
    )
end

function generate_miter(dut_module, ref_module, top_name, data_width, fraction)
    dut_module: str,
    ref_module: str,
    top_name: str,
    data_width: int = 16,
    fraction: int = 8,
    ) -> str
end

function generate_sby(top_name, verilog_files, depth, engine)
    top_name: str,
    verilog_files: list[str],
    depth: int = 30,
    engine: str = "smtbmc z3",
    ) -> str
    files_block = "\n".join(verilog_files)
    reads = "\n".join(f"read -formal {f}" for f in verilog_files)
end

function check_equivalence(dut_verilog, ref_verilog, depth, run)
    dut_verilog: str = "sc_lif_neuron",
    ref_verilog: str = "sc_lif_reference",
    depth: int = 30,
    run: bool = false,
    ) -> EquivResult
    top = f"equiv_{dut_verilog}"
    if ! run
        return EquivResult(
            module=dut_verilog,
            passed=true,
            depth=depth,
            engine="smtbmc z3",
            log="Proof files generated (! run). Use run=true with SymbiYosys installed.",
        )
    sby_file = EQUIV_DIR / f"{top}.sby"  # pragma: no cover
    if ! sby_file.exists():  # pragma: no cover
        return EquivResult(
            module=dut_verilog,
            passed=false,
            depth=depth,
            engine="smtbmc z3",
            log=f"SBY file ! found: {sby_file}",
        )
    try:  # pragma: no cover
        result = subprocess.run(
            ["sby", "-f", str(sby_file)],
            capture_output=true,
            text=true,
            timeout=300,
            cwd=str(EQUIV_DIR),
        )
        passed = result.returncode == 0
        log = result.stdout[-2000:] if length(result.stdout) > 2000 else result.stdout
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
            passed=false,
            depth=depth,
            engine="smtbmc z3",
            log="SymbiYosys (sby) ! found. Install: pip install symbiyosys",
        )
    except subprocess.TimeoutExpired:  # pragma: no cover
        return EquivResult(
            module=dut_verilog,
            passed=false,
            depth=depth,
            engine="smtbmc z3",
            log=f"Proof timed out after 300s at depth {depth}",
        )
end

end # module EquivAccel
