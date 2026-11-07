#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cosimulation of sc_shd_int_core against the integer reference
"""Cosimulation of ``hdl/sc_shd_int_core.v`` against ``tools/shd_int_reference.py``.

Writes the network's ROM files and a stimulus file, runs ``hdl/tb_sc_shd_int_core.v`` with
Icarus Verilog (or Verilator), and compares all 20 accumulators of every sample with the
reference. A refused length (0 or above 1024) must come back as ``REFUSED 1``.

Command line: ``cosim_shd_int_core.py NETWORK_JSON FRAMES_NPZ [--verilator] [--zero-skip 0|1]``
where FRAMES_NPZ holds ``frames`` (concatenated), ``lengths`` and ``labels``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import shd_int_reference as ref  # noqa: E402

CORE = REPO_ROOT / "hdl" / "sc_shd_int_core.v"
BENCH = REPO_ROOT / "hdl" / "tb_sc_shd_int_core.v"


@dataclass(frozen=True)
class SampleResult:
    """One sample's RTL output next to the reference."""

    rtl: list[int] | None
    reference: list[int] | None
    cycles: int | None
    row_reads: int | None
    refused: bool

    @property
    def matches(self) -> bool:
        return self.rtl == self.reference


def _hex_bytes(values: np.ndarray) -> str:
    """Element 0 in the lowest byte; printed most significant first."""
    return "".join(f"{int(v) & 0xFF:02x}" for v in values[::-1])


def write_network(net: ref.IntNetwork, out_dir: Path) -> None:
    """Write the five ROM files the core reads with $readmemh."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "q1.hex").write_text(
        "".join(_hex_bytes(net.q1[:, i]) + "\n" for i in range(ref.N_INPUT))
    )
    (out_dir / "q2.hex").write_text(
        "".join(_hex_bytes(net.q2[:, j]) + "\n" for j in range(ref.N_HIDDEN))
    )
    (out_dir / "q3.hex").write_text(
        "".join(_hex_bytes(net.q3[:, j]) + "\n" for j in range(ref.N_HIDDEN))
    )
    (out_dir / "d1.hex").write_text("".join(f"{int(x):02x}\n" for x in net.d1))
    (out_dir / "d2.hex").write_text("".join(f"{int(x):02x}\n" for x in net.d2))


def write_stimulus(samples: list, path: Path) -> int:
    """Write the stimulus file; an int sample is a refused length with no frames.

    Returns the number of words written, which sizes the testbench memory.
    """
    lines = [f"{len(samples):x}"]
    for sample in samples:
        if isinstance(sample, (int, np.integer)):
            lines.append(f"{int(sample) & 0xFFFF:x}")
            continue
        counts = np.asarray(sample, dtype=np.int64)
        if (
            counts.ndim != 2
            or counts.shape[1] != ref.N_INPUT
            or counts.min(initial=0) < 0
            or counts.max(initial=0) > 15
        ):
            raise ValueError("stimulus frames must be (T, 140) counts in 0..15")
        lines.append(f"{counts.shape[0]:x}")
        lines.extend("".join(f"{int(v):x}" for v in row[::-1]) for row in counts)
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


def parameters(net: ref.IntNetwork, zero_skip: int) -> dict[str, int]:
    """Core parameters derived from the integer network (widths from its proven bounds)."""
    return {
        "FRAC_BITS": net.frac_bits,
        "THETA1": net.theta1,
        "THETA2": net.theta2,
        "V1_WIDTH": net.membrane_width(1),
        "V2_WIDTH": net.membrane_width(2),
        "ZERO_SKIP": zero_skip,
    }


def _build(
    work: Path, params: dict[str, int], verilator: bool, timeout_seconds: float
) -> list[str]:
    if verilator:
        flags = [f"-G{k}={v}" for k, v in params.items()]
        subprocess.run(
            [
                "verilator",
                "--binary",
                "--timing",
                "-Wno-fatal",
                "-Wno-lint",
                "-Wno-style",
                "--top-module",
                "tb_sc_shd_int_core",
                "-Mdir",
                str(work / "obj"),
                *flags,
                str(BENCH),
                str(CORE),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
        return [str(work / "obj" / "Vtb_sc_shd_int_core")]
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise RuntimeError("the SHD core cosimulation requires Icarus Verilog (iverilog and vvp)")
    flags = [f"-Ptb_sc_shd_int_core.{k}={v}" for k, v in params.items()]
    subprocess.run(
        [iverilog, "-g2012", "-o", str(work / "sim.vvp"), *flags, str(BENCH), str(CORE)],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return [vvp, "-n", str(work / "sim.vvp")]


def simulate(
    net: ref.IntNetwork,
    samples: list,
    *,
    timeout_seconds: float,
    zero_skip: int = 1,
    verilator: bool = False,
    work: Path | None = None,
) -> list[SampleResult]:
    """Run the core on every sample and pair each output with the reference.

    ``timeout_seconds`` bounds the build and the simulation separately; it has no default
    because the right deadline depends on the host and the stimulus length.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    with tempfile.TemporaryDirectory(prefix="shd-int-cosim-") as tmp:
        work = Path(tmp) if work is None else work
        work.mkdir(parents=True, exist_ok=True)
        write_network(net, work)
        words = write_stimulus(samples, work / "stimulus.hex")
        params = dict(parameters(net, zero_skip), MAX_WORDS=words + 1)
        command = _build(work, params, verilator, timeout_seconds)
        subprocess.run(
            command, cwd=work, check=True, stdout=subprocess.DEVNULL, timeout=timeout_seconds
        )
        lines = (work / "outputs.txt").read_text().splitlines()
    if len(lines) != len(samples):
        raise RuntimeError(f"testbench wrote {len(lines)} results for {len(samples)} samples")
    results = []
    for sample, line in zip(samples, lines):
        if isinstance(sample, (int, np.integer)):
            results.append(
                SampleResult(None, None, None, None, refused=line.strip() == "REFUSED 1")
            )
            continue
        values, counters = line.split("|")
        cycles, reads = (int(x) for x in counters.split())
        results.append(
            SampleResult(
                [int(x) for x in values.split()],
                ref.run(net, sample).tolist(),
                cycles,
                reads,
                refused=False,
            )
        )
    return results


def _load_frames(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    data = np.load(path)
    offsets = np.concatenate([[0], np.cumsum(data["lengths"])])
    frames = [
        data["frames"][offsets[i] : offsets[i + 1]].astype(np.int64)
        for i in range(len(data["lengths"]))
    ]
    return frames, data["labels"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("network", type=Path, help="integer network JSON (IntNetwork.to_json)")
    parser.add_argument("frames", type=Path, help="npz with frames, lengths and labels")
    parser.add_argument("--zero-skip", type=int, choices=(0, 1), default=1)
    parser.add_argument("--verilator", action="store_true")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        required=True,
        help="deadline for the build and for the simulation",
    )
    args = parser.parse_args(argv)
    net = ref.IntNetwork.from_json(args.network.read_text())
    frames, labels = _load_frames(args.frames)
    results = simulate(
        net,
        frames,
        timeout_seconds=args.timeout_seconds,
        zero_skip=args.zero_skip,
        verilator=args.verilator,
    )
    mismatches = [i for i, r in enumerate(results) if not r.matches]
    correct = sum(int(np.argmax(r.rtl)) == int(y) for r, y in zip(results, labels))
    print(
        json.dumps(
            {
                "samples": len(results),
                "accumulator_mismatches": len(mismatches),
                "first_mismatches": mismatches[:5],
                "rtl_correct": correct,
                "row_reads_total": sum(r.row_reads for r in results),
            }
        )
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
