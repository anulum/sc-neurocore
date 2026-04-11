#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bit-true cosim of sc_axonal_delay.v vs Python reference
"""End-to-end cosim driver for hdl/sc_axonal_delay.v.

The Python reference is the per-source loop body inside
tools/shd_q88_reference.py::AxonalDelayBuffer.step(). For a single source
neuron with a fixed `read_offset`, the per-step semantics simplify to:

    buf[head]   = spike_in
    read_idx    = (head - read_offset) mod DEPTH
    spike_out   = buf[read_idx]
    head        = (head + 1) mod DEPTH

We do not import shd_q88_reference here because that module pulls in
heavy SHD dependencies; instead we re-implement the four lines above
verbatim and verify them indirectly via the existing
test_shd_q88_reference.py suite (which pins AxonalDelayBuffer).

Cosim cases (5):
  - offset=0   passthrough          (spike_out == spike_in for all cycles)
  - offset=1   single-cycle delay   (spike_out at t == spike_in at t-1)
  - offset=15  mid-range tap        (40 cycles, exercises wraparound)
  - offset=30  max delay            (60 cycles, exercises full depth)
  - offset=7   random pattern       (80 cycles, dense spiking)

PASS criterion: bit-for-bit equality of the spike_out trace (zero
tolerance).

Run:
    python tools/cosim_axonal_delay_verilog.py
or with pytest:
    pytest tools/cosim_axonal_delay_verilog.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HDL_DIR = os.path.join(ROOT, "hdl")
DUT = os.path.join(HDL_DIR, "sc_axonal_delay.v")
TB = os.path.join(HDL_DIR, "tb_sc_axonal_delay.v")

DEPTH = 31


@dataclass(frozen=True)
class StimulusCase:
    name: str
    spikes: tuple[int, ...]
    read_offset: int


def python_reference(spikes: tuple[int, ...], read_offset: int) -> list[int]:
    """Mirror of AxonalDelayBuffer.step() for a single source neuron."""
    if not (0 <= read_offset < DEPTH):
        raise ValueError(f"read_offset out of range: {read_offset}")
    buf = [0] * DEPTH
    head = 0
    out: list[int] = []
    for s in spikes:
        s = int(s) & 1
        buf[head] = s
        read_idx = (head - read_offset) % DEPTH
        out.append(buf[read_idx])
        head = (head + 1) % DEPTH
    return out


def write_hex(path: str, spikes: tuple[int, ...]) -> None:
    """Write a stimulus file: one bit (0/1) per line, $readmemh-friendly."""
    with open(path, "w") as f:
        f.write("// auto-generated stimulus for tb_sc_axonal_delay.v\n")
        for s in spikes:
            f.write(f"{int(s) & 1}\n")


def parse_output(path: str) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"unexpected line: {line!r}")
            rows.append((int(parts[0]), int(parts[1])))
    return rows


def make_cases() -> list[StimulusCase]:
    cases: list[StimulusCase] = []

    # Case 1: passthrough — every spike must appear immediately
    cases.append(StimulusCase("passthrough_offset_0", tuple(i % 2 for i in range(20)), 0))

    # Case 2: 1-cycle delay — output should be the previous spike
    cases.append(StimulusCase("delay_1_offset_1", (1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1), 1))

    # Case 3: mid-range delay (15 cycles) — exercises one wraparound
    cases.append(StimulusCase("delay_15_offset_15", tuple((i * 7 + 3) % 2 for i in range(40)), 15))

    # Case 4: maximum delay (DEPTH-1 = 30) — exercises full depth
    cases.append(StimulusCase("delay_30_offset_30", tuple((i * 13 + 1) % 2 for i in range(60)), 30))

    # Case 5: dense random-ish pattern at offset 7
    rng = [
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
    ]
    cases.append(StimulusCase("dense_offset_7", tuple(rng), 7))

    return cases


def run_iverilog(
    input_hex: str, output_txt: str, n_samples: int, read_offset: int, work_dir: str
) -> None:
    binary = os.path.join(work_dir, "tb_axdelay_sim")
    compile_cmd = [
        "iverilog",
        "-g2012",
        "-Wall",
        "-I",
        HDL_DIR,
        "-o",
        binary,
        DUT,
        TB,
    ]
    res = subprocess.run(compile_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"iverilog compile failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )

    run_cmd = [
        "vvp",
        binary,
        f"+N={n_samples}",
        f"+READ_OFFSET={read_offset}",
        f"+INPUT_FILE={input_hex}",
        f"+OUTPUT_FILE={output_txt}",
    ]
    res = subprocess.run(run_cmd, capture_output=True, text=True, cwd=work_dir)
    if res.returncode != 0:
        raise RuntimeError(f"vvp run failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")


def cosim_one_case(case: StimulusCase) -> None:
    expected = python_reference(case.spikes, case.read_offset)

    with tempfile.TemporaryDirectory(prefix=f"axdelay_{case.name}_") as work:
        in_hex = os.path.join(work, "stim.hex")
        out_txt = os.path.join(work, "out.txt")
        write_hex(in_hex, case.spikes)
        run_iverilog(in_hex, out_txt, len(case.spikes), case.read_offset, work)
        actual = parse_output(out_txt)

    if len(actual) != len(expected):
        raise AssertionError(
            f"[{case.name}] sample count mismatch: got {len(actual)}, expected {len(expected)}"
        )

    for i, ((step, sp_v), sp_p) in enumerate(zip(actual, expected)):
        if step != i:
            raise AssertionError(f"[{case.name}] step index mismatch at row {i}: got {step}")
        if sp_v != sp_p:
            raise AssertionError(
                f"[{case.name}] step {i}: spike_out mismatch "
                f"verilog={sp_v} python={sp_p} "
                f"(spike_in={case.spikes[i]}, read_offset={case.read_offset})"
            )

    print(
        f"  {case.name}: {len(expected)} steps, "
        f"in_spikes={sum(case.spikes)}, "
        f"out_spikes={sum(expected)} — BIT-TRUE MATCH"
    )


def main() -> int:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        sys.stderr.write("ERROR: iverilog/vvp not found on PATH\n")
        return 2
    if not os.path.exists(DUT):
        sys.stderr.write(f"ERROR: missing DUT: {DUT}\n")
        return 2
    if not os.path.exists(TB):
        sys.stderr.write(f"ERROR: missing testbench: {TB}\n")
        return 2

    print("Cosim sc_axonal_delay.v vs Python reference")
    cases = make_cases()
    for case in cases:
        cosim_one_case(case)
    print(f"All {len(cases)} cases passed (bit-true).")
    return 0


# Pytest hooks ----------------------------------------------------------------
def _pytest_param_id(case: StimulusCase) -> str:
    return case.name


try:
    import pytest

    @pytest.mark.parametrize("case", make_cases(), ids=_pytest_param_id)
    def test_cosim_case(case: StimulusCase) -> None:
        if shutil.which("iverilog") is None or shutil.which("vvp") is None:
            pytest.skip("iverilog/vvp not installed")
        cosim_one_case(case)

    @pytest.mark.parametrize(
        "offset",
        [0, 1, 7, 15, 30],
        ids=lambda o: f"py_offset_{o}",
    )
    def test_python_reference_self_consistent(offset: int) -> None:
        """Sanity-check the Python reference against a hand-rolled trace."""
        spikes = tuple(i % 3 == 0 for i in range(50))
        out = python_reference(spikes, offset)
        # For each step t, the output should equal spikes[t - offset] when
        # t - offset >= 0 (the buffer was zero-initialised so anything before
        # step 0 reads 0).
        for t in range(len(spikes)):
            ref = spikes[t - offset] if t - offset >= 0 else 0
            assert out[t] == ref, f"offset={offset} t={t}: ref={ref} got={out[t]}"
except ModuleNotFoundError:
    pass


if __name__ == "__main__":
    sys.exit(main())
