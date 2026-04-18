#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end SHD top-level Verilog cosim vs Python reference
"""End-to-end cosim driver for hdl/sc_shd_top.v.

This script verifies that the full Verilog SHD inference network
(140 → 128 → 128 → 20 with axonal delays + Vmin_LIF + sparse int8 dense)
produces *bit-identical* `output_v_sum` to the pure-Python Q8.8 reference
in tools/shd_q88_reference.py::run_inference_q88.

For each test case the driver:
  1. Loads the FPGA artifacts emitted by tools/extract_shd_weights.py.
  2. Generates a synthetic input spike train (deterministic, seeded).
  3. Runs the Python reference and captures the FULL 20-class
     output_v_sum (not just the argmax).
  4. Materialises the artifacts + stimulus + scales into a temporary
     working directory in the layout expected by tb_sc_shd_top.v.
  5. Compiles and runs the Verilog top with iverilog/vvp.
  6. Parses the dumped outputs.txt and asserts BIT-TRUE equality of all
     20 accumulator values across both implementations.

Cases (3, ordered short-to-real for fast iteration):
  - tiny_T16    16 input timesteps, 30% spike density
  - small_T64   64 input timesteps, 15% spike density
  - shd_T100   100 input timesteps, 5% density (close to real SHD samples)

Run:
    python tools/cosim_shd_top_verilog.py
or via pytest:
    pytest tools/cosim_shd_top_verilog.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HDL_DIR = os.path.join(ROOT, "hdl")
ARTIFACTS_DIR = os.path.join(ROOT, "data/masquelier_shd/fpga_artifacts/dcls_max")

DUT_FILES = [
    os.path.join(HDL_DIR, "sc_axonal_delay.v"),
    os.path.join(HDL_DIR, "sc_dense_int8_sparse.v"),
    os.path.join(HDL_DIR, "sc_vmin_lif_neuron.v"),
    os.path.join(HDL_DIR, "sc_shd_top.v"),
    os.path.join(HDL_DIR, "tb_sc_shd_top.v"),
]

# Pull the Python Q8.8 reference (and its run_inference helper) without
# importing torch — shd_q88_reference is numpy + stdlib only.
sys.path.insert(0, HERE)
from shd_q88_reference import (  # noqa: E402
    DCLS_TIME_EXTENSION,
    AxonalDelayBuffer,
    load_artifacts,
    sparse_dense_q88,
    vmin_lif_population_step,
)


@dataclass(frozen=True)
class StimulusCase:
    name: str
    t_orig: int
    spike_density: float
    seed: int


def python_output_v_sum(net, input_spikes: np.ndarray) -> np.ndarray:
    """Mirror of run_inference_q88 but returns the full output_v_sum."""
    T_orig, n_in = input_spikes.shape
    assert n_in == 140
    T1 = T_orig + DCLS_TIME_EXTENSION
    T2 = T1 + DCLS_TIME_EXTENSION

    delay1 = AxonalDelayBuffer(140, net.delays_l1)
    v1 = np.zeros(128, dtype=np.int32)
    spikes_l1 = np.zeros((T1, 128), dtype=np.int8)
    zero_input = np.zeros(140, dtype=np.int8)
    for t in range(T1):
        in_t = input_spikes[t] if t < T_orig else zero_input
        s_in_delayed = delay1.step(in_t)
        x1 = sparse_dense_q88(net.layer1.weights_int8, net.layer1.scale, s_in_delayed)
        v1, s1 = vmin_lif_population_step(v1, x1, net.softplus_lut, net.cfg)
        spikes_l1[t] = s1

    delay2 = AxonalDelayBuffer(128, net.delays_l2)
    v2 = np.zeros(128, dtype=np.int32)
    output_v_sum = np.zeros(20, dtype=np.int64)
    zero_l1 = np.zeros(128, dtype=np.int8)
    for t in range(T2):
        s1_t = spikes_l1[t] if t < T1 else zero_l1
        s1_delayed = delay2.step(s1_t)
        x2 = sparse_dense_q88(net.layer2.weights_int8, net.layer2.scale, s1_delayed)
        v2, s2 = vmin_lif_population_step(v2, x2, net.softplus_lut, net.cfg)
        x3 = sparse_dense_q88(net.layer3.weights_int8, net.layer3.scale, s2)
        output_v_sum += x3.astype(np.int64)
    return output_v_sum


def write_int8_hex(path: str, arr: np.ndarray) -> None:
    """Write a flat row-major signed-int8 array as $readmemh hex (1 byte/line)."""
    flat = arr.flatten().astype(np.int32).tolist()
    with open(path, "w") as f:
        f.write("// auto-generated for tb_sc_shd_top.v\n")
        for v in flat:
            byte = v if v >= 0 else (256 + v)
            f.write(f"{byte:02x}\n")


def write_spikes_hex(path: str, spikes: np.ndarray) -> None:
    """Write spike vectors as one 140-bit hex value per line."""
    T_orig, n_in = spikes.shape
    nibbles = (n_in + 3) // 4
    with open(path, "w") as f:
        f.write("// auto-generated stimulus for tb_sc_shd_top.v\n")
        for t in range(T_orig):
            value = 0
            for i in range(n_in):
                if spikes[t, i]:
                    value |= 1 << i
            f.write(f"{value:0{nibbles}x}\n")


def write_scales(path: str, s1: int, s2: int, s3: int) -> None:
    with open(path, "w") as f:
        f.write(f"{s1} {s2} {s3}\n")


def parse_outputs(path: str) -> list[int]:
    rows: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"unexpected line: {line!r}")
            rows.append(int(parts[1]))
    return rows


def run_iverilog(case: StimulusCase, work_dir: str) -> None:
    binary = os.path.join(work_dir, "tb_shd_top_sim")
    compile_cmd = [
        "iverilog",
        "-g2012",
        "-Wall",
        "-I",
        HDL_DIR,
        "-o",
        binary,
        *DUT_FILES,
    ]
    res = subprocess.run(compile_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"iverilog compile failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )

    run_cmd = ["vvp", binary, f"+T={case.t_orig}"]
    res = subprocess.run(run_cmd, capture_output=True, text=True, cwd=work_dir)
    if res.returncode != 0:
        raise RuntimeError(f"vvp run failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")


def cosim_one_case(case: StimulusCase) -> None:
    if not os.path.isdir(ARTIFACTS_DIR):
        raise RuntimeError(
            f"missing artifacts dir: {ARTIFACTS_DIR}\nrun tools/extract_shd_weights.py first"
        )

    net = load_artifacts(ARTIFACTS_DIR)

    rng = np.random.default_rng(case.seed)
    spikes = (rng.random((case.t_orig, 140)) < case.spike_density).astype(np.int8)

    expected = python_output_v_sum(net, spikes)

    # Q16.16 scales rounded the same way as the Python reference
    s1 = round(net.layer1.scale * 65536)
    s2 = round(net.layer2.scale * 65536)
    s3 = round(net.layer3.scale * 65536)

    with tempfile.TemporaryDirectory(prefix=f"shd_top_{case.name}_") as work:
        write_int8_hex(os.path.join(work, "weights_layer1.hex"), net.layer1.weights_int8)
        write_int8_hex(os.path.join(work, "weights_layer2.hex"), net.layer2.weights_int8)
        write_int8_hex(os.path.join(work, "weights_layer3.hex"), net.layer3.weights_int8)
        write_int8_hex(os.path.join(work, "delays_layer1.hex"), net.delays_l1)
        write_int8_hex(os.path.join(work, "delays_layer2.hex"), net.delays_l2)
        write_spikes_hex(os.path.join(work, "spikes.hex"), spikes)
        write_scales(os.path.join(work, "scales.txt"), s1, s2, s3)
        run_iverilog(case, work)
        actual = parse_outputs(os.path.join(work, "outputs.txt"))

    if len(actual) != 20:
        raise AssertionError(f"[{case.name}] expected 20 outputs, got {len(actual)}")

    diffs: list[tuple[int, int, int]] = []
    for j in range(20):
        if actual[j] != int(expected[j]):
            diffs.append((j, actual[j], int(expected[j])))

    if diffs:
        msg_lines = [
            f"[{case.name}] mismatch on {len(diffs)}/20 classes "
            f"(T_orig={case.t_orig}, density={case.spike_density:.2f}):"
        ]
        for j, av, ev in diffs[:5]:
            msg_lines.append(f"  class {j:2d}: verilog={av:>10d}  python={ev:>10d}  diff={av - ev}")
        py_argmax = int(np.argmax(expected))
        v_argmax = int(np.argmax(actual))
        msg_lines.append(f"  argmax: verilog={v_argmax}  python={py_argmax}")
        raise AssertionError("\n".join(msg_lines))

    py_argmax = int(np.argmax(expected))
    print(
        f"  {case.name}: T={case.t_orig} density={case.spike_density:.0%} "
        f"20/20 classes BIT-TRUE MATCH (argmax={py_argmax})"
    )


def make_cases() -> list[StimulusCase]:
    return [
        StimulusCase("tiny_T16", 16, 0.30, seed=1),
        StimulusCase("small_T64", 64, 0.15, seed=2),
        StimulusCase("shd_T100", 100, 0.05, seed=3),
    ]


def main() -> int:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        sys.stderr.write("ERROR: iverilog/vvp not found on PATH\n")
        return 2

    print("Cosim sc_shd_top.v vs Python reference (output_v_sum bit-true)")
    cases = make_cases()
    for case in cases:
        cosim_one_case(case)
    print(f"All {len(cases)} cases passed (bit-true on full 20-class output).")
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
        if not os.path.isdir(ARTIFACTS_DIR):
            pytest.skip(f"missing artifacts dir: {ARTIFACTS_DIR}")
        cosim_one_case(case)
except ModuleNotFoundError:
    pass


if __name__ == "__main__":
    sys.exit(main())
