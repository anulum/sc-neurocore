# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Orchestration Pipeline for sc-neurocore's Hardware Compiler.

This module provides the automated workflow to take a stochastic graph from
the MLIREmitter and compile it down to a bitstream using open-source FPGA tools:
1. CIRCT (firtool) -> Verilog
2. Yosys -> Synthesis (BLIF/JSON)
3. NextPNR -> Place & Route
4. IcePack / Project Xray -> Bitstream
"""

import os
import subprocess
import logging

logger = logging.getLogger(__name__)


class CompilerPipeline:
    """
    Automated hardware synthesis pipeline.
    """

    def __init__(self, work_dir: str = ".tmp/compiler"):
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def compile_mlir_to_verilog(self, mlir_content: str, output_name: str = "top") -> str:
        """
        Invokes 'firtool' to lower MLIR to Verilog.
        """
        mlir_path = os.path.join(self.work_dir, f"{output_name}.mlir")
        v_path = os.path.join(self.work_dir, f"{output_name}.v")

        with open(mlir_path, "w") as f:
            f.write(mlir_content)

        logger.info(f"Lowering {mlir_path} to Verilog...")
        # Note: In a real environment, firtool must be in PATH
        try:
            subprocess.run(["firtool", mlir_path, "-o", v_path], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"firtool failed or not found: {e}. Falling back to stub Verilog.")
            # Fallback for demo/development without full toolchain
            with open(v_path, "w") as f:
                f.write(
                    f"// Stub Verilog generated for {output_name}\nmodule {output_name}(); endmodule"
                )

        return v_path

    def run_synthesis(self, v_path: str, target_fpga: str = "ice40") -> str:
        """
        Invokes 'yosys' for synthesis.
        """
        base = os.path.splitext(v_path)[0]
        json_path = f"{base}.json"

        logger.info(f"Synthesizing {v_path} for {target_fpga}...")
        yosys_cmd = ["yosys", "-p", f"read_verilog {v_path}; synth_{target_fpga} -json {json_path}"]

        try:
            subprocess.run(yosys_cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"yosys failed or not found: {e}")

        return json_path

    def run_pnr(self, json_path: str, target_device: str = "up5k") -> str:
        """
        Invokes 'nextpnr' for place and route.
        """
        asc_path = f"{os.path.splitext(json_path)[0]}.asc"

        logger.info(f"Running P&R for {target_device}...")
        pnr_cmd = ["nextpnr-ice40", "--up5k", "--json", json_path, "--asc", asc_path]

        try:
            subprocess.run(pnr_cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"nextpnr failed or not found: {e}")

        return asc_path
