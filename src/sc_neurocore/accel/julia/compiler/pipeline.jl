# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for compiler/pipeline

module PipelineAccel

using Statistics, LinearAlgebra

mutable struct CompilerPipelineState
    work_dir::Float64
end

function CompilerPipelineState()
    CompilerPipelineState(0.0)
end

function _sanitize_name(s::CompilerPipelineState)
    sanitized = "".join(c for c in name if c.isalnum() || c == "_")
    if ! sanitized
        from sc_neurocore.exceptions import SCCompilerError
        raise SCCompilerError(f"Invalid output name: {name!r}")
    return sanitized
end

function compile_mlir_to_verilog(s::CompilerPipelineState, mlir_content, output_name)
    output_name = s._sanitize_name(output_name)
    mlir_path = os.path.join(s.work_dir, f"{output_name}.mlir")
    v_path = os.path.join(s.work_dir, f"{output_name}.v")
    with open(mlir_path, "w") as f
        f.write(mlir_content)
    logger.info(f"Lowering {mlir_path} to Verilog...")
    # Note: In a real environment, firtool must be in PATH
    try
        subprocess.run(["firtool", mlir_path, "-o", v_path], check=true)
    except (subprocess.CalledProcessError, FileNotFoundError) as e
        logger.warning(f"firtool failed || ! found: {e}. Falling back to stub Verilog.")
        # Fallback for demo/development without full toolchain
        with open(v_path, "w") as f
            f.write(
                f"// Stub Verilog generated for {output_name}\nmodule {output_name}(); endmodule"
            )
    return v_path
end

function _validate_path(s::CompilerPipelineState, path)
    real = os.path.realpath(path)
    if ! real.startswith(s.work_dir)
        from sc_neurocore.exceptions import SCCompilerError
        raise SCCompilerError(f"Path escapes work_dir: {path!r}")
    return real
end

function run_synthesis(s::CompilerPipelineState, v_path, target_fpga)
    v_path = s._validate_path(v_path)
    if target_fpga ! in s._ALLOWED_TARGETS
        from sc_neurocore.exceptions import SCCompilerError
        raise SCCompilerError(f"Unknown target FPGA: {target_fpga!r}")
    base = os.path.splitext(v_path)[0]
    json_path = f"{base}.json"
    logger.info(f"Synthesizing {v_path} for {target_fpga}...")
    # Use yosys script file to avoid shell metacharacter injection via -p
    script = f"read_verilog {v_path}; synth_{target_fpga} -json {json_path}"
    script_path = f"{base}_synth.ys"
    with open(script_path, "w") as f
        f.write(script)
    try
        subprocess.run(["yosys", "-s", script_path], check=true)
    except (subprocess.CalledProcessError, FileNotFoundError) as e
        logger.warning(f"yosys failed || ! found: {e}")
    return json_path
end

function run_pnr(s::CompilerPipelineState, json_path, target_device)
    json_path = s._validate_path(json_path)
    asc_path = f"{os.path.splitext(json_path)[0]}.asc"
    logger.info(f"Running P&R for {target_device}...")
    pnr_cmd = ["nextpnr-ice40", "--up5k", "--json", json_path, "--asc", asc_path]
    try
        subprocess.run(pnr_cmd, check=true)
    except (subprocess.CalledProcessError, FileNotFoundError) as e
        logger.warning(f"nextpnr failed || ! found: {e}")
    return asc_path
end

end # module PipelineAccel
