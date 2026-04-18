#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bit-true cosim of sc_vmin_lif_neuron.v vs Python reference
"""End-to-end cosim driver for hdl/sc_vmin_lif_neuron.v.

For each named stimulus pattern, this script:
  1. Generates a Q8.8 input current trajectory.
  2. Runs the bit-true Python reference (vmin_lif_step_q88) step by step.
  3. Writes the inputs to a $readmemh-compatible hex file.
  4. Compiles + simulates hdl/sc_vmin_lif_neuron.v with hdl/tb_vmin_lif_neuron.v
     using iverilog + vvp (must be installed locally).
  5. Parses the simulator output (`# step spike v_out` text file).
  6. Asserts that:
       a) every spike index matches the Python reference exactly
       b) every membrane potential matches the Python reference exactly
     If anything diverges, prints the first divergence and exits non-zero.

The accepted PASS criterion is BIT-TRUE equality (zero LSB tolerance) for
both spikes and v_out across all 4 stimulus patterns. The Python reference
itself is verified by tools/test_gen_vmin_lif_lut.py (33 tests).

Run:
    python tools/cosim_vmin_lif_verilog.py
or with pytest:
    pytest tools/cosim_vmin_lif_verilog.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Make tools/ importable so we can pull the bit-true reference
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from gen_vmin_lif_lut import (  # noqa: E402  (sys.path tweak above)
    LUT_RANGE,
    LUT_SIZE,
    VminLifConfig,
    decode_q88,
    encode_q88,
    gen_softplus_lut,
    vmin_lif_step_q88,
)

HDL_DIR = os.path.join(ROOT, "hdl")
DUT = os.path.join(HDL_DIR, "sc_vmin_lif_neuron.v")
TB = os.path.join(HDL_DIR, "tb_vmin_lif_neuron.v")


@dataclass(frozen=True)
class StimulusCase:
    name: str
    samples_q88: tuple[int, ...]


def _q88_to_signed_word(v: int) -> int:
    """Convert a Python signed Q8.8 int to the unsigned 16-bit pattern
    that $readmemh would interpret as the same signed value."""
    if v < 0:
        return (v + (1 << 16)) & 0xFFFF
    return v & 0xFFFF


def _signed_word_to_q88(word: int) -> int:
    """Inverse of _q88_to_signed_word."""
    word &= 0xFFFF
    if word & 0x8000:
        return word - (1 << 16)
    return word


def write_hex(path: str, samples_q88) -> None:
    """Dump samples to a $readmemh-compatible hex file (one 16-bit value/line)."""
    with open(path, "w") as f:
        f.write("// auto-generated stimulus for tb_vmin_lif_neuron.v\n")
        for v in samples_q88:
            f.write(f"{_q88_to_signed_word(int(v)):04x}\n")


def parse_output(path: str) -> list[tuple[int, int, int]]:
    """Parse the testbench output: lines `<step> <spike> <v_out_signed>`.

    The Verilog testbench writes v_out as a signed decimal already (no
    sign-extension juggling needed)."""
    rows: list[tuple[int, int, int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"unexpected line: {line!r}")
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def python_reference(samples_q88, cfg: VminLifConfig) -> list[tuple[int, int]]:
    """Run vmin_lif_step_q88 over the trajectory and return [(spike, v)]."""
    lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
    v = 0
    out: list[tuple[int, int]] = []
    for x in samples_q88:
        v_next, spike = vmin_lif_step_q88(v, int(x), lut, cfg)
        out.append((spike, v_next))
        v = v_next
    return out


def make_cases() -> list[StimulusCase]:
    """4 stimulus patterns covering the operating regime."""
    cases: list[StimulusCase] = []

    # Case 1: zeros — should drift to v_inf and never spike
    cases.append(StimulusCase("zeros", tuple([0] * 32)))

    # Case 2: constant strong drive — should spike rapidly
    cases.append(StimulusCase("strong_drive", tuple([encode_q88(0.5)] * 32)))

    # Case 3: ramp from 0 to 1.5 over 32 steps
    ramp = tuple(encode_q88(1.5 * i / 31) for i in range(32))
    cases.append(StimulusCase("ramp", ramp))

    # Case 4: impulse train (one big spike-causing input every 5 steps)
    impulse = []
    for i in range(40):
        impulse.append(encode_q88(1.2) if i % 5 == 0 else 0)
    cases.append(StimulusCase("impulse_train", tuple(impulse)))

    return cases


def run_iverilog(input_hex: str, output_txt: str, n_samples: int, work_dir: str) -> None:
    """Compile + simulate via iverilog/vvp from inside `work_dir`."""
    binary = os.path.join(work_dir, "tb_vmin_lif_sim")
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
        f"+INPUT_FILE={input_hex}",
        f"+OUTPUT_FILE={output_txt}",
    ]
    res = subprocess.run(run_cmd, capture_output=True, text=True, cwd=work_dir)
    if res.returncode != 0:
        raise RuntimeError(f"vvp run failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")


def cosim_one_case(case: StimulusCase, cfg: VminLifConfig) -> None:
    """Run a single case end-to-end and assert bit-true equality."""
    expected = python_reference(case.samples_q88, cfg)

    with tempfile.TemporaryDirectory(prefix=f"vmin_cosim_{case.name}_") as work:
        in_hex = os.path.join(work, "stim.hex")
        out_txt = os.path.join(work, "out.txt")
        write_hex(in_hex, case.samples_q88)
        run_iverilog(in_hex, out_txt, len(case.samples_q88), work)
        actual = parse_output(out_txt)

    if len(actual) != len(expected):
        raise AssertionError(
            f"[{case.name}] sample count mismatch: got {len(actual)}, expected {len(expected)}"
        )

    for i, ((step, sp_v, vo_v), (sp_p, vo_p)) in enumerate(zip(actual, expected)):
        if step != i:
            raise AssertionError(f"[{case.name}] step index mismatch at row {i}: got {step}")
        if sp_v != sp_p:
            raise AssertionError(
                f"[{case.name}] step {i}: spike mismatch verilog={sp_v} python={sp_p} "
                f"(x_q88={case.samples_q88[i]}, v_python={vo_p})"
            )
        if vo_v != vo_p:
            raise AssertionError(
                f"[{case.name}] step {i}: v_out mismatch verilog={vo_v} python={vo_p} "
                f"(x_q88={case.samples_q88[i]}, "
                f"diff={vo_v - vo_p}, "
                f"v_python_float={decode_q88(vo_p):.6f}, "
                f"v_verilog_float={decode_q88(vo_v):.6f})"
            )

    print(
        f"  {case.name}: {len(expected)} steps, "
        f"spikes={sum(s for s, _ in expected)} — BIT-TRUE MATCH"
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

    cfg = VminLifConfig()
    print(f"Cosim sc_vmin_lif_neuron.v vs Python (cfg={cfg})")
    cases = make_cases()
    for case in cases:
        cosim_one_case(case, cfg)
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
        cosim_one_case(case, VminLifConfig())
except ModuleNotFoundError:
    pass


if __name__ == "__main__":
    sys.exit(main())
