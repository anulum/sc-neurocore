#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Vmin_LIF softplus LUT + bit-true Python reference
"""Generate the softplus lookup table for the Vmin_LIF Verilog module
and provide a bit-true Python reference implementation that matches
what the Verilog computes step-by-step.

The Vmin_LIFNode (Masquelier SHD model) has these dynamics per step:
    v_new = v * (1 - 1/tau) + x        # neuronal_charge_no_decay_input
    v_new = v_inf + softplus(v_new - v_inf, beta)
    spike = (v_new >= v_threshold)
    v_new = v_reset * spike + (1 - spike) * v_new   # hard reset

For our SHD config:
    tau = 4.0  →  decay = 1 - 1/4 = 0.75 = Q8.8 192
    v_threshold = 1.0  →  Q8.8 256
    v_reset = 0.0  →  Q8.8 0
    v_inf = -5.0  →  Q8.8 -1280
    beta_v_inf = 1.0  →  Q8.8 256

softplus(z, beta) = (1/beta) * log(1 + exp(beta * z))

The LUT is indexed by `v - v_inf` (always non-negative because softplus
floor is monotonic in z). With v_inf = -5 and v_threshold = 1, the
expected range is z ∈ [0, ~6]. We use a 64-entry LUT covering [0, 16]
in Q8.8 (step = 16/64 = 0.25 = Q8.8 64).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Q8.8 fixed-point constants
Q88_SCALE = 256  # 2^8
Q88_MAX = 2**15 - 1  # 32767 (signed 16-bit)
Q88_MIN = -(2**15)  # -32768

# LUT configuration
LUT_SIZE = 64
LUT_RANGE = 16.0  # cover z ∈ [0, 16]
LUT_STEP = LUT_RANGE / LUT_SIZE  # 0.25 → Q8.8 step = 64


@dataclass(frozen=True)
class VminLifConfig:
    """Configuration for Vmin_LIF — matches Masquelier SHD model defaults."""

    tau: float = 4.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    v_inf: float = -5.0
    beta_v_inf: float = 1.0


# Module-level singleton for default config (frozen dataclass is safe to share).
DEFAULT_CFG = VminLifConfig()


def encode_q88(value: float) -> int:
    """Encode a float as Q8.8 signed 16-bit integer."""
    encoded = int(round(value * Q88_SCALE))
    if encoded > Q88_MAX:
        return Q88_MAX
    if encoded < Q88_MIN:
        return Q88_MIN
    return encoded


def decode_q88(value: int) -> float:
    """Decode a Q8.8 signed integer back to float."""
    return value / Q88_SCALE


def softplus_float(z: float, beta: float = 1.0) -> float:
    """Reference softplus: (1/beta) * log(1 + exp(beta * z)).

    Uses the high-threshold approximation when beta*z > 20 to match
    PyTorch's F.softplus(threshold=20) behaviour exactly.
    """
    if beta * z > 20.0:
        return z
    return (1.0 / beta) * math.log1p(math.exp(beta * z))


def gen_softplus_lut(
    beta: float = 1.0, size: int = LUT_SIZE, z_max: float = LUT_RANGE
) -> list[int]:
    """Generate a Q8.8 LUT for softplus(z, beta) over z ∈ [0, z_max]."""
    step = z_max / size
    return [encode_q88(softplus_float(i * step, beta)) for i in range(size)]


def lut_lookup(lut: list[int], z_q88: int, size: int = LUT_SIZE, z_max: float = LUT_RANGE) -> int:
    """Look up softplus value via LUT with linear interpolation.

    z_q88 is the Q8.8 encoded value of z = v - v_inf.
    For z < 0: softplus is bounded near 0 → return 0
    For z >= z_max: linear extension (softplus(z) ≈ z for large z)
    """
    if z_q88 <= 0:
        # softplus(0) = log(2)/beta — for safety return small positive
        return lut[0] if lut else 0

    z_max_q88 = encode_q88(z_max)
    if z_q88 >= z_max_q88:
        # Linear regime: softplus(z) ≈ z for large z
        return z_q88

    # Index calculation: idx = z * size / z_max
    # In Q8.8: z_q88 * size / (z_max * 256) = z_q88 * size / z_max_q88
    idx_full = (z_q88 * size) // z_max_q88
    if idx_full >= size - 1:
        return lut[size - 1]

    # Linear interpolation between lut[idx_full] and lut[idx_full+1]
    frac_num = (z_q88 * size) - (idx_full * z_max_q88)
    a = lut[idx_full]
    b = lut[idx_full + 1]
    interp = a + ((b - a) * frac_num) // z_max_q88
    return interp


def vmin_lif_step_q88(
    v_q88: int, x_q88: int, lut: list[int], cfg: VminLifConfig = DEFAULT_CFG
) -> tuple[int, int]:
    """Bit-true single step of Vmin_LIF in Q8.8 fixed-point.

    Matches PyTorch JIT eval order (jit_eval_single_step_forward_hard_reset_no_decay_input):
      1. v = v * (1 - 1/tau) + x        (neuronal_charge_no_decay_input)
      2. spike = (v >= v_threshold)     (fire — checked on CHARGED v, before floor)
      3. v = v_reset * spike + (1 - spike) * v   (hard reset)
      4. v = v_inf + softplus(v - v_inf, beta)   (vmin floor — applied AFTER reset)

    Returns (v_next_q88, spike).
    """
    v_thresh_q88 = encode_q88(cfg.v_threshold)
    v_reset_q88 = encode_q88(cfg.v_reset)
    v_inf_q88 = encode_q88(cfg.v_inf)
    decay_q88 = encode_q88(1.0 - 1.0 / cfg.tau)  # 0.75 for tau=4.0

    # Step 1: neuronal_charge_no_decay_input
    #   v_charged = v * decay + x
    # (v * decay) is Q16.16 → ARITHMETIC shift right by 8 to recover Q8.8.
    # Python's >> on int is arithmetic (matches Verilog >>>): -3 >> 1 = -2.
    v_decayed = (v_q88 * decay_q88) >> 8
    v_charged = v_decayed + x_q88
    if v_charged > Q88_MAX:
        v_charged = Q88_MAX
    elif v_charged < Q88_MIN:
        v_charged = Q88_MIN

    # Step 2: threshold check on CHARGED v (BEFORE floor)
    if v_charged >= v_thresh_q88:
        spike = 1
        v_post_reset = v_reset_q88
    else:
        spike = 0
        v_post_reset = v_charged

    # Step 3: vmin softplus floor applied to post-reset v
    z = v_post_reset - v_inf_q88
    sp = lut_lookup(lut, z)
    v_next = v_inf_q88 + sp
    if v_next > Q88_MAX:
        v_next = Q88_MAX
    elif v_next < Q88_MIN:
        v_next = Q88_MIN

    return v_next, spike


def vmin_lif_step_float(v: float, x: float, cfg: VminLifConfig = DEFAULT_CFG) -> tuple[float, int]:
    """Float reference matching PyTorch jit_eval order:
    charge -> threshold -> reset -> vmin floor.
    """
    v = v * (1.0 - 1.0 / cfg.tau) + x
    if v >= cfg.v_threshold:
        spike = 1
        v = cfg.v_reset
    else:
        spike = 0
    v = cfg.v_inf + softplus_float(v - cfg.v_inf, cfg.beta_v_inf)
    return v, spike


def emit_lut_verilog_header(lut: list[int], cfg: VminLifConfig = DEFAULT_CFG) -> str:
    """Emit Verilog `define statements for the softplus LUT."""
    lines = [
        "// SPDX-License-Identifier: AGPL-3.0-or-later",
        "// Commercial license available",
        "// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "// © Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "// ORCID: 0009-0009-3560-0851",
        "// Contact: www.anulum.li | protoscience@anulum.li",
        "// SC-NeuroCore — Auto-generated softplus LUT for Vmin_LIF — DO NOT EDIT",
        "// Generated by tools/gen_vmin_lif_lut.py",
        f"// Q8.8 fixed-point, {len(lut)} entries, range z ∈ [0, {LUT_RANGE}]",
        f"// beta = {cfg.beta_v_inf}, v_inf = {cfg.v_inf}",
        f"`define VMIN_LUT_SIZE {len(lut)}",
        f"`define VMIN_LUT_RANGE_Q88 {encode_q88(LUT_RANGE)}",
    ]
    for i, val in enumerate(lut):
        lines.append(f"`define VMIN_LUT_{i:02d} 16'sd{val}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Vmin-LIF LUT generator command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-vh", default=None, help="Output Verilog header file path")
    parser.add_argument("--print-lut", action="store_true", help="Print the LUT values to stdout")
    parser.add_argument(
        "--demo", action="store_true", help="Run a small demo: 100 steps with constant input"
    )
    args = parser.parse_args(argv)

    cfg = VminLifConfig()
    lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)

    if args.print_lut:
        print("# Vmin_LIF softplus LUT — Q8.8 fixed-point")
        print(f"# beta={cfg.beta_v_inf}, size={LUT_SIZE}, range=[0,{LUT_RANGE}]")
        for i, v in enumerate(lut):
            z = i * LUT_STEP
            print(f"  [{i:2d}]  z={z:5.2f}  softplus={decode_q88(v):8.4f}  q88={v:6d}")

    if args.out_vh:
        Path(args.out_vh).write_text(emit_lut_verilog_header(lut, cfg), encoding="utf-8")
        print(f"Written {len(lut)} LUT entries to {args.out_vh}")

    if args.demo:
        print("\n=== Demo: 20 steps, constant input x=0.3, v_init=-3 ===")
        print(f"{'step':>4} {'v_q88':>8} {'v_float':>10} {'spike':>6} {'v_ref':>10} {'err':>8}")
        v_q88 = encode_q88(-3.0)
        v_ref = -3.0
        for t in range(20):
            x_q88 = encode_q88(0.3)
            v_q88, spike = vmin_lif_step_q88(v_q88, x_q88, lut, cfg)
            v_ref, _ = vmin_lif_step_float(v_ref, 0.3, cfg)
            err = decode_q88(v_q88) - v_ref
            print(
                f"{t:>4} {v_q88:>8} {decode_q88(v_q88):>10.4f} "
                f"{spike:>6} {v_ref:>10.4f} {err:>+8.4f}"
            )

    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point.
    raise SystemExit(main())
