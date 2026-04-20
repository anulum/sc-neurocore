# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — High-level Mojo SIMD kernel runner

"""Mojo SIMD Kernel Orchestrator.

Spawns and manages high-performance Mojo binaries natively replacing Python bottlenecks.
Expects `pixi run mojo` to be available strictly on the system PATH.
"""

from __future__ import annotations

import os
import subprocess
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class MojoKernelRunner:
    """Manages execution and telemetry gathering for the underlying monolithic Mojo suite."""

    _mojo_dir: Path = Path(__file__).parent
    _pixi_bin: str = field(default_factory=lambda: os.path.expanduser("~/.pixi/bin/pixi"))

    def __post_init__(self):
        if not (self._mojo_dir / "kernels.mojo").exists():
            raise FileNotFoundError(f"kernels.mojo not found at {self._mojo_dir}")

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
                if "ms" in line and ":" in line:
                    parts = line.rsplit(":", 1)
                    label = parts[0].strip()
                    val_match = re.search(r"([\d.e+-]+)\s*ms", parts[1])
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
            print(f"[Mojo Runner] Pixi or Mojo completely missing at {self._pixi_bin}. Check installation bounds.")
            return {}
