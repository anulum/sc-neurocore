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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from sc_neurocore.accel.mojo.isa_baseline import pin_isa

MOJO_HELPER_BACKEND: Final = "python-fallback"
"""Backend used by scalar helper methods on ``MojoKernelRunner`` today."""

MOJO_HELPER_IPC_AVAILABLE: Final = False
"""Whether scalar helper methods dispatch through direct Mojo IPC today."""

__all__ = [
    "MOJO_HELPER_BACKEND",
    "MOJO_HELPER_IPC_AVAILABLE",
    "MojoKernelRunner",
]


@dataclass
class MojoKernelRunner:
    """Run the maintained monolithic Mojo kernel suite from Python.

    The runner is a subprocess façade over ``kernels.mojo``. It discovers the
    kernel bundle at construction time, invokes the pixi-managed Mojo toolchain
    for builds and benchmark runs, and keeps Python fallbacks for scalar helper
    methods. Those scalar helpers do not attempt hidden Mojo IPC; benchmark
    execution remains the explicit Mojo subprocess surface.

    Parameters
    ----------
    _mojo_dir:
        Directory expected to contain ``kernels.mojo``. The default is the
        installed package directory.
    _pixi_bin:
        Absolute pixi executable used to launch the Mojo toolchain.
    """

    _mojo_dir: Path = Path(__file__).parent
    _pixi_bin: str = field(default_factory=lambda: os.path.expanduser("~/.pixi/bin/pixi"))

    def __post_init__(self) -> None:
        """Validate the configured Mojo kernel directory.

        Raises
        ------
        FileNotFoundError
            Raised when neither the configured directory nor the installed
            package directory contains ``kernels.mojo``.
        """
        # Prefer source-tree location, then installed package.
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
        """Build ``kernels.mojo`` through pixi.

        Returns
        -------
        bool
            ``True`` when ``pixi run mojo build kernels.mojo`` exits cleanly;
            ``False`` when the toolchain invocation raises.
        """
        try:
            subprocess.run(
                pin_isa([self._pixi_bin, "run", "mojo", "build", "kernels.mojo"]),
                cwd=str(self._mojo_dir),
                check=True,
            )
            return True
        except Exception as e:
            print(f"[Mojo Runner] Build failed: {e}")
            return False

    def run_benchmark(self, timeout_sec: int = 60) -> dict[str, float]:
        """Run the kernel benchmark suite and parse millisecond timings.

        Parameters
        ----------
        timeout_sec:
            Hard timeout in seconds for the Mojo subprocess.

        Returns
        -------
        dict[str, float]
            Mapping from benchmark labels printed by ``kernels.mojo`` to their
            measured millisecond durations. Returns an empty mapping when the
            subprocess fails, times out, or cannot be launched.
        """
        try:
            result = subprocess.run(
                pin_isa([self._pixi_bin, "run", "mojo", "run", "kernels.mojo"]),
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout_sec,
                cwd=str(self._mojo_dir),
            )

            timings: dict[str, float] = {}
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
        """Return the Python Hamming weight of packed stochastic words.

        Parameters
        ----------
        data:
            Packed integer words whose set bits are counted.

        Returns
        -------
        int
            Total number of set bits. The current maintained runtime path uses
            the Python implementation and does not attempt direct Mojo IPC.
        """
        from sc_neurocore.edge.bitstream import popcount_slice

        return popcount_slice(data)

    def lfsr_encode(self, seed: int, threshold: int, bits: int) -> list[int]:
        """Encode a threshold stream with the maintained LFSR-16 fallback.

        Parameters
        ----------
        seed:
            Non-zero 16-bit LFSR seed.
        threshold:
            Comparator threshold used to generate stochastic bits.
        bits:
            Number of output bits to generate.

        Returns
        -------
        list[int]
            Packed words generated by the Python LFSR implementation while
            direct Mojo IPC remains unavailable for scalar helper calls.
        """
        from sc_neurocore.edge.lfsr import Lfsr16

        lfsr = Lfsr16(seed)
        return lfsr.encode(threshold, bits)
