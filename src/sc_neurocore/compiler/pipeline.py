# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Orchestration Pipeline for sc-neurocore's Hardware Compiler

"""
Orchestration Pipeline for sc-neurocore's Hardware Compiler.

This module provides the automated workflow to take a stochastic graph from
the MLIREmitter and compile it down to a bitstream using open-source FPGA tools:
1. CIRCT (firtool) -> Verilog
2. Yosys -> Synthesis (BLIF/JSON)
3. NextPNR -> Place & Route
4. IcePack / Project Xray -> Bitstream
"""

import logging
import os
import subprocess

from sc_neurocore.exceptions import SCCompilerError

logger = logging.getLogger(__name__)


class CompilerPipeline:
    """
    Automated hardware synthesis pipeline.
    """

    def __init__(self, work_dir: str = ".tmp/compiler"):
        self.work_dir = os.path.realpath(work_dir)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Restrict output_name to alphanumeric + underscore."""
        sanitized = "".join(c for c in name if c.isalnum() or c == "_")
        if not sanitized:
            raise SCCompilerError(f"Invalid output name: {name!r}")
        return sanitized

    def compile_mlir_to_verilog(self, mlir_content: str, output_name: str = "top") -> str:
        """
        Invokes 'firtool' to lower MLIR to Verilog.
        """
        output_name = self._sanitize_name(output_name)
        mlir_path = os.path.join(self.work_dir, f"{output_name}.mlir")
        v_path = os.path.join(self.work_dir, f"{output_name}.v")

        with open(mlir_path, "w") as f:
            f.write(mlir_content)

        logger.info(f"Lowering {mlir_path} to Verilog...")
        try:
            subprocess.run(["firtool", mlir_path, "-o", v_path], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            if os.path.exists(v_path):
                os.remove(v_path)
            raise SCCompilerError(
                "firtool failed; refusing to emit fallback Verilog. "
                "Install CIRCT firtool or run MLIR bundle generation for evidence-only output."
            ) from e

        return v_path

    _ALLOWED_TARGETS = {"ice40", "ecp5", "gowin", "xilinx"}

    def _validate_path(self, path: str) -> str:
        """Ensure path resolves inside work_dir."""
        real = os.path.realpath(path)
        if not real.startswith(self.work_dir):
            raise SCCompilerError(f"Path escapes work_dir: {path!r}")
        return real

    def run_synthesis(self, v_path: str, target_fpga: str = "ice40") -> str:
        """
        Invokes 'yosys' for synthesis.
        """
        v_path = self._validate_path(v_path)
        if target_fpga not in self._ALLOWED_TARGETS:
            raise SCCompilerError(f"Unknown target FPGA: {target_fpga!r}")

        base = os.path.splitext(v_path)[0]
        json_path = f"{base}.json"

        logger.info(f"Synthesizing {v_path} for {target_fpga}...")
        # Use yosys script file to avoid shell metacharacter injection via -p
        script = f"read_verilog {v_path}; synth_{target_fpga} -json {json_path}"
        script_path = f"{base}_synth.ys"
        with open(script_path, "w") as f:
            f.write(script)

        try:
            subprocess.run(["yosys", "-s", script_path], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"yosys failed or not found: {e}")

        return json_path

    def run_pnr(self, json_path: str, target_device: str = "up5k") -> str:
        """
        Invokes 'nextpnr' for place and route.
        """
        json_path = self._validate_path(json_path)
        asc_path = f"{os.path.splitext(json_path)[0]}.asc"

        logger.info(f"Running P&R for {target_device}...")
        pnr_cmd = ["nextpnr-ice40", "--up5k", "--json", json_path, "--asc", asc_path]

        try:
            subprocess.run(pnr_cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"nextpnr failed or not found: {e}")

        return asc_path
