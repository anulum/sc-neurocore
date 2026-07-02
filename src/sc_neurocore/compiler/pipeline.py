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
import shutil

# External EDA tools are invoked with fixed argv lists and shell=False.
import subprocess  # nosec B404

from sc_neurocore.exceptions import SCCompilerError

logger = logging.getLogger(__name__)


class CompilerPipeline:
    """Coordinate MLIR lowering, synthesis, and place-and-route tool steps.

    The pipeline writes generated artifacts under ``work_dir`` and validates
    artifact paths before invoking the external EDA toolchain.
    """

    def __init__(self, work_dir: str = ".tmp/compiler") -> None:
        self.work_dir = os.path.realpath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)

    @staticmethod
    def _resolve_tool(tool_name: str) -> str:
        resolved = shutil.which(tool_name)
        if resolved is None:
            raise FileNotFoundError(tool_name)
        return os.path.realpath(resolved)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Restrict output_name to alphanumeric + underscore."""
        sanitized = "".join(c for c in name if c.isalnum() or c == "_")
        if not sanitized:
            raise SCCompilerError(f"Invalid output name: {name!r}")
        return sanitized

    def compile_mlir_to_verilog(self, mlir_content: str, output_name: str = "top") -> str:
        """Lower MLIR text to Verilog with CIRCT ``firtool``.

        Parameters
        ----------
        mlir_content:
            MLIR module text to write to the pipeline work directory.
        output_name:
            Artifact stem for the generated ``.mlir`` and ``.v`` files.

        Returns
        -------
        str
            Absolute path to the generated Verilog file.

        Raises
        ------
        SCCompilerError
            If the output stem is invalid, ``firtool`` is missing, ``firtool``
            exits unsuccessfully, or a partial Verilog output must be removed.
        """
        output_name = self._sanitize_name(output_name)
        mlir_path = os.path.join(self.work_dir, f"{output_name}.mlir")
        v_path = os.path.join(self.work_dir, f"{output_name}.v")

        with open(mlir_path, "w", encoding="utf-8") as f:
            f.write(mlir_content)

        logger.info("Lowering %s to Verilog...", mlir_path)
        try:
            firtool = self._resolve_tool("firtool")
            subprocess.run([firtool, mlir_path, "-o", v_path], check=True)  # nosec B603
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
        if os.path.commonpath([self.work_dir, real]) != self.work_dir:
            raise SCCompilerError(f"Path escapes work_dir: {path!r}")
        return real

    def run_synthesis(self, v_path: str, target_fpga: str = "ice40") -> str:
        """Run Yosys synthesis and return the expected JSON netlist path.

        Parameters
        ----------
        v_path:
            Verilog source path under ``work_dir``.
        target_fpga:
            Yosys synthesis target name.

        Returns
        -------
        str
            Expected ``.json`` netlist path under ``work_dir``.

        Raises
        ------
        SCCompilerError
            If ``v_path`` escapes ``work_dir`` or ``target_fpga`` is unknown.
        """
        v_path = self._validate_path(v_path)
        if target_fpga not in self._ALLOWED_TARGETS:
            raise SCCompilerError(f"Unknown target FPGA: {target_fpga!r}")

        base = os.path.splitext(v_path)[0]
        json_path = f"{base}.json"

        logger.info("Synthesizing %s for %s...", v_path, target_fpga)
        # Use yosys script file to avoid shell metacharacter injection via -p
        script = f"read_verilog {v_path}; synth_{target_fpga} -json {json_path}"
        script_path = f"{base}_synth.ys"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        try:
            yosys = self._resolve_tool("yosys")
            subprocess.run([yosys, "-s", script_path], check=True)  # nosec B603
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("yosys failed or not found: %s", e)

        return json_path

    def run_pnr(self, json_path: str, target_device: str = "up5k") -> str:
        """Run nextpnr place-and-route and return the expected ASC path.

        Parameters
        ----------
        json_path:
            Synthesized JSON netlist path under ``work_dir``.
        target_device:
            Human-readable target device label for telemetry.

        Returns
        -------
        str
            Expected ``.asc`` place-and-route artifact path under ``work_dir``.

        Raises
        ------
        SCCompilerError
            If ``json_path`` escapes ``work_dir``.
        """
        json_path = self._validate_path(json_path)
        asc_path = f"{os.path.splitext(json_path)[0]}.asc"

        logger.info("Running P&R for %s...", target_device)

        try:
            nextpnr = self._resolve_tool("nextpnr-ice40")
            pnr_cmd = [nextpnr, "--up5k", "--json", json_path, "--asc", asc_path]
            subprocess.run(pnr_cmd, check=True)  # nosec B603
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("nextpnr failed or not found: %s", e)

        return asc_path
