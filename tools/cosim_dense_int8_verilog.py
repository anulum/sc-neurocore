#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bit-true cosim of sc_dense_int8_sparse.v vs Python reference
"""End-to-end cosim driver for hdl/sc_dense_int8_sparse.v.

For each test case (different IN_F × OUT_F dimensions), this script:
  1. Generates random int8 weights and a per-tensor scale.
  2. Generates N random spike vectors with controllable density.
  3. Computes the expected Q8.8 outputs by mirroring
     tools/shd_q88_reference.py::sparse_dense_q88 in pure Python (no
     numpy import — keeps the test self-contained and trivially auditable).
  4. Writes weights.hex, spikes.hex, scale.txt into a fresh temp directory.
  5. Compiles the testbench with `iverilog -PIN_F=... -POUT_F=...` so a
     single source covers every layer geometry.
  6. Runs vvp from inside the temp directory and reads outputs.txt.
  7. Asserts BIT-TRUE equality (zero tolerance) of every output value
     across every sample.

Cases (5):
  - tiny_8x4_dense       — fast smoke test (8 inputs, 4 outputs, dense spikes)
  - tiny_8x4_sparse      — same shape, ~10% spike density (sparsity check)
  - shd_layer1_140x128   — real SHD layer 1 geometry, 90% sparse
  - shd_layer2_128x128   — real SHD layer 2 geometry, 90% sparse
  - shd_layer3_128x20    — real SHD output layer geometry, 90% sparse

PASS criterion: BIT-TRUE equality on every (sample, output_neuron) pair.

Run:
    python tools/cosim_dense_int8_verilog.py
or with pytest:
    pytest tools/cosim_dense_int8_verilog.py -v
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HDL_DIR = os.path.join(ROOT, "hdl")
DUT = os.path.join(HDL_DIR, "sc_dense_int8_sparse.v")
TB = os.path.join(HDL_DIR, "tb_sc_dense_int8_sparse.v")

Q88_MAX = 2**15 - 1
Q88_MIN = -(2**15)


@dataclass(frozen=True)
class StimulusCase:
    name: str
    in_features: int
    out_features: int
    n_samples: int
    spike_density: float
    scale_q16_16: int
    seed: int


def python_reference(
    weights: list[list[int]], scale_q16_16: int, spike_vec: list[int]
) -> list[int]:
    """Mirror sparse_dense_q88 in pure Python (no numpy).

    Args:
        weights: shape (OUT_FEATURES, IN_FEATURES), int8 values
        scale_q16_16: per-tensor scale already rendered as int (Q16.16)
        spike_vec: length IN_FEATURES, 0/1 values
    Returns:
        list of length OUT_FEATURES, signed Q8.8 ints (16-bit range)
    """
    out_features = len(weights)
    in_features = len(weights[0]) if out_features else 0
    if len(spike_vec) != in_features:
        raise ValueError("spike length mismatch")

    out: list[int] = []
    for j in range(out_features):
        accum = 0
        row = weights[j]
        for i in range(in_features):
            if spike_vec[i]:
                accum += row[i]
        product = accum * scale_q16_16
        v_q88 = product >> 8  # arithmetic shift on Python int
        if v_q88 > Q88_MAX:
            v_q88 = Q88_MAX
        elif v_q88 < Q88_MIN:
            v_q88 = Q88_MIN
        out.append(v_q88)
    return out


def gen_weights(out_f: int, in_f: int, rng: random.Random) -> list[list[int]]:
    """Generate random int8 weights uniformly in [-127, 127]."""
    return [[rng.randint(-127, 127) for _ in range(in_f)] for _ in range(out_f)]


def gen_spikes(in_f: int, n_samples: int, density: float, rng: random.Random) -> list[list[int]]:
    """Generate n random spike vectors with the requested density."""
    return [[1 if rng.random() < density else 0 for _ in range(in_f)] for _ in range(n_samples)]


def write_weights_hex(path: str, weights: list[list[int]]) -> None:
    """Row-major flat hex file, one signed-int8 byte per line."""
    with open(path, "w") as f:
        f.write("// auto-generated weights for tb_sc_dense_int8_sparse.v\n")
        for row in weights:
            for w in row:
                byte = w if w >= 0 else (256 + w)
                f.write(f"{byte:02x}\n")


def write_spikes_hex(path: str, spikes: list[list[int]], in_f: int) -> None:
    """One IN_F-bit hex value per line. Width auto-rounds to a byte."""
    nibbles = (in_f + 3) // 4
    with open(path, "w") as f:
        f.write("// auto-generated spikes for tb_sc_dense_int8_sparse.v\n")
        for vec in spikes:
            # Build the bit pattern with bit 0 of vec at LSB.
            value = 0
            for i, bit in enumerate(vec):
                if bit:
                    value |= 1 << i
            f.write(f"{value:0{nibbles}x}\n")


def write_scale(path: str, scale_q16_16: int) -> None:
    with open(path, "w") as f:
        f.write(f"{scale_q16_16}\n")


def parse_outputs(path: str, out_f: int) -> list[list[int]]:
    rows: list[list[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != out_f + 1:
                raise ValueError(f"expected {out_f + 1} fields, got {len(parts)} in line: {line!r}")
            rows.append([int(x) for x in parts[1:]])
    return rows


def make_cases() -> list[StimulusCase]:
    return [
        StimulusCase("tiny_8x4_dense", 8, 4, 16, 0.50, scale_q16_16=4096, seed=1),  # scale ≈ 0.0625
        StimulusCase("tiny_8x4_sparse", 8, 4, 16, 0.10, scale_q16_16=8192, seed=2),  # scale ≈ 0.125
        StimulusCase(
            "shd_layer1_140x128", 140, 128, 8, 0.10, scale_q16_16=1875, seed=3
        ),  # scale ≈ 0.0286 (realistic)
        StimulusCase("shd_layer2_128x128", 128, 128, 8, 0.10, scale_q16_16=2100, seed=4),
        StimulusCase("shd_layer3_128x20", 128, 20, 8, 0.10, scale_q16_16=3200, seed=5),
    ]


def run_iverilog(case: StimulusCase, work_dir: str) -> None:
    binary = os.path.join(work_dir, "tb_dense_sim")
    compile_cmd = [
        "iverilog",
        "-g2012",
        "-Wall",
        "-I",
        HDL_DIR,
        f"-Ptb_sc_dense_int8_sparse.IN_F={case.in_features}",
        f"-Ptb_sc_dense_int8_sparse.OUT_F={case.out_features}",
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

    run_cmd = ["vvp", binary, f"+N={case.n_samples}"]
    res = subprocess.run(run_cmd, capture_output=True, text=True, cwd=work_dir)
    if res.returncode != 0:
        raise RuntimeError(f"vvp run failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")


def cosim_one_case(case: StimulusCase) -> None:
    rng = random.Random(case.seed)
    weights = gen_weights(case.out_features, case.in_features, rng)
    spikes = gen_spikes(case.in_features, case.n_samples, case.spike_density, rng)

    expected = [python_reference(weights, case.scale_q16_16, vec) for vec in spikes]

    with tempfile.TemporaryDirectory(prefix=f"dense_{case.name}_") as work:
        write_weights_hex(os.path.join(work, "weights.hex"), weights)
        write_spikes_hex(os.path.join(work, "spikes.hex"), spikes, case.in_features)
        write_scale(os.path.join(work, "scale.txt"), case.scale_q16_16)
        run_iverilog(case, work)
        actual = parse_outputs(os.path.join(work, "outputs.txt"), case.out_features)

    if len(actual) != len(expected):
        raise AssertionError(
            f"[{case.name}] sample count mismatch: got {len(actual)}, expected {len(expected)}"
        )

    for s_idx, (act_row, exp_row) in enumerate(zip(actual, expected)):
        for j, (av, ev) in enumerate(zip(act_row, exp_row)):
            if av != ev:
                raise AssertionError(
                    f"[{case.name}] sample {s_idx}, neuron {j}: "
                    f"verilog={av} python={ev} (diff={av - ev}, "
                    f"spikes_active={sum(spikes[s_idx])})"
                )

    n_outputs = case.n_samples * case.out_features
    print(
        f"  {case.name}: IN={case.in_features} OUT={case.out_features} "
        f"N={case.n_samples} density={case.spike_density:.0%} "
        f"({n_outputs} comparisons) — BIT-TRUE MATCH"
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

    print("Cosim sc_dense_int8_sparse.v vs Python reference")
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

    def test_python_reference_known_value() -> None:
        """Hand-checked example from sparse_dense_q88 docstring:
        accum=5, scale_q16_16=1875 → (5*1875)>>8 = 9375>>8 = 36."""
        weights = [[5]]  # 1×1 layer
        spikes = [1]
        out = python_reference(weights, 1875, spikes)
        assert out == [36]

    def test_python_reference_negative_clamp() -> None:
        """Strong negative weights should saturate to Q8.8 -32768."""
        # accum = -127*100 = -12700, scale=10000
        # product = -127_000_000, >>8 = -496094 → clipped to -32768
        weights = [[-127] * 100]
        spikes = [1] * 100
        out = python_reference(weights, 10000, spikes)
        assert out == [-32768]

    def test_python_reference_zero_spikes() -> None:
        weights = [[1, 2, 3], [-4, 5, -6]]
        out = python_reference(weights, 65536, [0, 0, 0])
        assert out == [0, 0]
except ModuleNotFoundError:
    pass


if __name__ == "__main__":
    sys.exit(main())
