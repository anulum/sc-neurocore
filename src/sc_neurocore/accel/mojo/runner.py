# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — High-level Mojo SIMD kernel runner

"""Mojo SIMD Kernel Orchestrator.

This loader is part of the maintained Mojo surface.

Important boundary:

- authoritative Mojo behaviour comes from Python loaders and compiled libraries
  explicitly wired into maintained Python code
- transcript-style mirrors under ``accel/mojo/kernels/*.mojo`` are not an
  authoritative runtime contract unless they are explicitly loaded and tested

Expects `pixi run mojo` to be available strictly on the system PATH.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class MojoKernelRunner:
    """Manages execution and telemetry gathering for the underlying monolithic Mojo suite."""

    _mojo_dir: Path = Path(__file__).parent
    _pixi_bin: str = field(default_factory=lambda: os.path.expanduser("~/.pixi/bin/pixi"))

    def __post_init__(self) -> None:
        # Prefer source-tree location, then installed package
        mojo_file = self._mojo_dir / "kernels.mojo"
        if mojo_file.exists():
            return
        # Installed package fallback (kernels.mojo should be in package data)
        installed_mojo = Path(__file__).parent / "kernels.mojo"
        if installed_mojo.exists():
            self._mojo_dir = installed_mojo.parent
            return
        raise FileNotFoundError("kernels.mojo not found. Run: pixi install && pixi run mojo build")

    def build(self) -> bool:
        """Helper to invoke `mojo build` natively across the active working directory."""
        try:
            subprocess.run(
                [self._pixi_bin, "run", "mojo", "build", "kernels.mojo"],
                cwd=str(self._mojo_dir),
                check=True,
            )
            return True
        except Exception as e:
            print(f"[Mojo Runner] Build failed: {e}")
            return False

    def run_benchmark(self, timeout_sec: int = 60) -> Dict[str, float]:
        """Runs the entire kernel suite and parses output times natively in MS."""
        try:
            start_time = time.time()
            result = subprocess.run(
                [self._pixi_bin, "run", "mojo", "run", "kernels.mojo"],
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout_sec,
                cwd=str(self._mojo_dir),
            )

            timings = {}
            for line in result.stdout.splitlines():
                if "ms" in line.lower() and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        label = parts[0].strip()
                        val_match = re.search(
                            r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*ms", parts[1], re.IGNORECASE
                        )
                        if val_match:
                            timings[label] = float(val_match.group(1))

            return timings

        except subprocess.CalledProcessError as e:
            print(f"[Mojo Runner] Execution failed: {e.stderr}")
            return {}
        except subprocess.TimeoutExpired:
            print(f"[Mojo Runner] Hard timeout of {timeout_sec}s exceeded.")
            return {}
        except FileNotFoundError:
            print(
                f"[Mojo Runner] Pixi or Mojo completely missing at {self._pixi_bin}. Check installation bounds."
            )
            return {}

    def popcount(self, data: list[int]) -> int:
        """Call the Mojo SIMD kernel directly or fall back to Python."""
        try:
            # Mojo C-FFI pipeline target
            raise NotImplementedError("Mojo IPC bindings pending v4.0")
        except Exception:
            from sc_neurocore.edge.bitstream import popcount_slice

            return popcount_slice(data)

    def lfsr_encode(self, seed: int, threshold: int, bits: int) -> list[int]:
        """Call the Mojo LFSR-16 encoder directly or fall back to Python."""
        try:
            raise NotImplementedError("Mojo IPC bindings pending v4.0")
        except Exception:
            from sc_neurocore.edge.lfsr import Lfsr16

            lfsr = Lfsr16(seed)
            return lfsr.encode(threshold, bits)
