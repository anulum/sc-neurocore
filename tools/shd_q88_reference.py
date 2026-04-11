#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bit-true Q8.8 SHD network reference for Verilog cosim
"""End-to-end Q8.8 fixed-point simulator of the SHD network using
the artifacts produced by tools/extract_shd_weights.py.

This is the bit-true reference for Verilog cosim — every operation
matches what hand-coded Verilog modules will compute.

Architecture (Masquelier SHD model with Vmin_LIF + axonal delays):
  Input(140) -> AxonalDelay(140, max=31) -> Dense(140->128, int8)
              -> Vmin_LIF(128) -> Dropout(0)  -- inference, dropout disabled
              -> AxonalDelay(128, max=31) -> Dense(128->128, int8, sparse 90%)
              -> Vmin_LIF(128)
              -> Dense(128->20, int8, sparse 90%)
              -> Output: sum membrane voltage over time -> argmax

Per-step compute:
  1. Each input spike s_in[t] enters axonal delay buffer
  2. After delay, fed into dense weight matrix (sparse int8 mat-vec)
  3. Dense output goes through Vmin_LIF dynamics (see gen_vmin_lif_lut.py)
  4. Output spikes feed next layer
  5. Final layer accumulates membrane voltages -> classifier

All accumulators use signed 32-bit Q24.8 (24 integer bits, 8 fraction)
to avoid overflow with sparse-sum chain. Final scaling back to Q8.8
applies the per-tensor scale factor.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

# Pure-Python — only stdlib + numpy (no torch needed)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_vmin_lif_lut import (
    LUT_RANGE,
    LUT_SIZE,
    Q88_MAX,
    Q88_MIN,
    VminLifConfig,
    gen_softplus_lut,
    vmin_lif_step_q88,
)


SHD_MAX_DELAY = 31  # max_feedforward_delay from config_SHD


@dataclass
class LayerWeights:
    """Loaded weights for a single dense layer."""

    name: str
    in_features: int
    out_features: int
    weights_int8: np.ndarray  # shape (out, in)
    scale: float


@dataclass
class ShdNetwork:
    """All weights, delays, and config for the SHD network."""

    layer1: LayerWeights  # 140 -> 128
    layer2: LayerWeights  # 128 -> 128 (sparse)
    layer3: LayerWeights  # 128 -> 20  (sparse)
    delays_l1: np.ndarray  # int8, shape (140,)
    delays_l2: np.ndarray  # int8, shape (128,)
    cfg: VminLifConfig
    softplus_lut: list[int]


def load_artifacts(artifact_dir: str) -> ShdNetwork:
    """Load extracted weights, delays, scales from FPGA artifact dir."""
    with open(os.path.join(artifact_dir, "scales.json")) as f:
        scales = json.load(f)

    def _load_int8_hex(path: str, shape: tuple[int, int]) -> np.ndarray:
        rows, cols = shape
        data = np.zeros(rows * cols, dtype=np.int8)
        idx = 0
        with open(path) as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                byte = int(line, 16)
                # unsigned 8-bit -> signed
                if byte >= 128:
                    byte -= 256
                data[idx] = byte
                idx += 1
        if idx != rows * cols:
            raise ValueError(f"{path}: read {idx} bytes, expected {rows * cols}")
        return data.reshape(shape)

    def _load_delays_hex(path: str, n: int) -> np.ndarray:
        data = np.zeros(n, dtype=np.int8)
        idx = 0
        with open(path) as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                byte = int(line, 16)
                if byte >= 128:
                    byte -= 256
                data[idx] = byte
                idx += 1
        if idx != n:
            raise ValueError(f"{path}: read {idx} delays, expected {n}")
        return data

    layer1 = LayerWeights(
        name="layer1_input_to_h1",
        in_features=140,
        out_features=128,
        weights_int8=_load_int8_hex(
            os.path.join(artifact_dir, "weights_layer1_input_to_h1_int8.hex"), (128, 140)
        ),
        scale=scales["layer1_input_to_h1"],
    )
    layer2 = LayerWeights(
        name="layer2_h1_to_h2",
        in_features=128,
        out_features=128,
        weights_int8=_load_int8_hex(
            os.path.join(artifact_dir, "weights_layer2_h1_to_h2_int8.hex"), (128, 128)
        ),
        scale=scales["layer2_h1_to_h2"],
    )
    layer3 = LayerWeights(
        name="layer3_h2_output",
        in_features=128,
        out_features=20,
        weights_int8=_load_int8_hex(
            os.path.join(artifact_dir, "weights_layer3_h2_output_int8.hex"), (20, 128)
        ),
        scale=scales["layer3_h2_output"],
    )

    delays_l1 = _load_delays_hex(os.path.join(artifact_dir, "delays_layer1_input_to_h1.hex"), 140)
    delays_l2 = _load_delays_hex(os.path.join(artifact_dir, "delays_layer2_h1_to_h2.hex"), 128)

    cfg = VminLifConfig()
    softplus_lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)

    return ShdNetwork(
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        delays_l1=delays_l1,
        delays_l2=delays_l2,
        cfg=cfg,
        softplus_lut=softplus_lut,
    )


class AxonalDelayBuffer:
    """Per-source-neuron circular buffer of past spikes.

    Each source neuron has a delay value d ∈ [-15, +15]. We store the
    last (max_delay + max(|d|)) spikes and read from the offset that
    matches the per-neuron delay. For axonal delays, the SAME spike is
    delivered to all targets at one offset (the source's delay).
    """

    def __init__(self, n_sources: int, delays: np.ndarray, max_delay: int = SHD_MAX_DELAY):
        self.n = n_sources
        # Buffer depth = max_delay (covers d ∈ [-15, +15] with d_centered + 15)
        self.depth = max_delay
        self.buf = np.zeros((self.n, self.depth), dtype=np.int8)
        self.head = 0
        # DCLS conv math: output[t] = sum_k weight[k] * input[t + k - left_padding]
        # For 1-tap kernel at k = (max_delay // 2) + P:
        #   output[t] = input[t + (max_delay // 2) + P - (max_delay - 1)]
        #             = input[t + P - (max_delay - 1 - max_delay//2)]
        #             = input[t + P - 15]   (for max_delay=31)
        # So effective delay = 15 - P  (i.e. P=+15 → delay 0, P=-15 → delay 30)
        center = (max_delay - 1) // 2
        self.read_offset = center - delays.astype(np.int32)

    def step(self, spikes_in: np.ndarray) -> np.ndarray:
        """Push current spikes, return delayed spikes for this step."""
        if spikes_in.shape != (self.n,):
            raise ValueError(f"expected ({self.n},), got {spikes_in.shape}")
        # Write current input at head
        self.buf[:, self.head] = spikes_in
        # Read delayed spikes — each neuron reads from its own offset
        out = np.zeros(self.n, dtype=np.int8)
        for i in range(self.n):
            read_idx = (self.head - self.read_offset[i]) % self.depth
            out[i] = self.buf[i, read_idx]
        self.head = (self.head + 1) % self.depth
        return out


def sparse_dense_q88(weights_int8: np.ndarray, scale: float, spikes_in: np.ndarray) -> np.ndarray:
    """Sparse int8 dense matvec for binary spike inputs, output in Q8.8.

    For each output neuron j:
        accum_int = sum_i W[j,i] * spike[i]      # int32 (max ~127*140 ≈ 18000)
        v_real    = accum_int * scale            # float
        v_q88     = round(v_real * 256)          # Q8.8 representation

    Implementation uses Q16.16 fixed-point for `scale` to keep precision:
        scale_q16_16 = round(scale * 65536)
        v_q88 = (accum_int * scale_q16_16) >> 8  # equivalent to *scale*256

    For Q8.8 output v_q88 = int8_accum * scale * 256:
      e.g. accum_int=5, scale=0.0286 → v_real=0.143 → v_q88=int(36.6)=36
      Q16.16 scale = round(0.0286*65536) = 1875
      formula: (5 * 1875) >> 8 = 9375 >> 8 = 36 ✓
    """
    out_features, in_features = weights_int8.shape
    if spikes_in.shape != (in_features,):
        raise ValueError(f"expected ({in_features},), got {spikes_in.shape}")
    # Pure int matvec — only entries where spike==1 contribute
    accum_int = (weights_int8.astype(np.int32) * spikes_in.astype(np.int32)).sum(axis=1)
    # Q16.16 scale for precision (16-bit fractional part)
    scale_q16_16 = round(scale * 65536)
    out = np.zeros(out_features, dtype=np.int32)
    for j in range(out_features):
        # (accum_int * scale_q16_16) is the real value scaled by 2^16.
        # ASHR by 8 → Q8.8. Python's >> is arithmetic shift on int (matches Verilog >>>).
        product = int(accum_int[j]) * scale_q16_16
        v_q88 = product >> 8
        if v_q88 > Q88_MAX:
            v_q88 = Q88_MAX
        elif v_q88 < Q88_MIN:
            v_q88 = Q88_MIN
        out[j] = v_q88
    return out


def vmin_lif_population_step(
    v_state: np.ndarray, x_input: np.ndarray, softplus_lut: list[int], cfg: VminLifConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Vmin_LIF dynamics to a vector of neurons (Q8.8 in/out)."""
    n = len(v_state)
    new_v = np.zeros(n, dtype=np.int32)
    spikes = np.zeros(n, dtype=np.int8)
    for i in range(n):
        v_next, sp = vmin_lif_step_q88(int(v_state[i]), int(x_input[i]), softplus_lut, cfg)
        new_v[i] = v_next
        spikes[i] = sp
    return new_v, spikes


DCLS_TIME_EXTENSION = 15  # right_padding for axonal dcls_module (see config_SHD)


def _run_stage1(net: ShdNetwork, input_spikes: np.ndarray, T_orig: int, T1: int) -> np.ndarray:
    """Stage 1: input → dcls_l1 → dense_l1 → vmin_lif_l1 over T1 cycles.

    Real input for cycles [0, T_orig), zero-padded for [T_orig, T1) to
    mirror the asymmetric padding of the DCLS layer in the PyTorch model.
    """
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
    return spikes_l1


def _run_stage2(net: ShdNetwork, spikes_l1: np.ndarray, T1: int, T2: int) -> int:
    """Stage 2: spikes_l1 → dcls_l2 → dense_l2 → vmin_lif_l2 → dense_l3.

    Runs for T2 cycles, zero-padding spikes_l1 after index T1-1 to mirror
    the second DCLS layer's asymmetric padding. Returns argmax over the
    output-layer voltage sum across all T2 cycles.
    """
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
    return int(np.argmax(output_v_sum))


def run_inference_q88(net: ShdNetwork, input_spikes: np.ndarray) -> int:
    """Run one SHD sample through the network.

    The Masquelier SHD model uses asymmetric padding around its DCLS axonal
    delay layers (`config.left_padding = 30`, `config.right_padding = 15`),
    so each `dcls_module` EXTENDS the time axis by 15 cycles. With two
    such layers (input→l1 and l1→l2), the network output has T + 30
    cycles of accumulation, not T. Failing to mirror this extension was
    the dominant cause of the historical 4% Q8.8 vs PyTorch accuracy gap.
    The fix runs the network in two stages (see `_run_stage1`, `_run_stage2`).

    Args:
      net: loaded network
      input_spikes: shape (T, 140), int8 binary spikes per timestep

    Returns:
      predicted class index (0-19)
    """
    T_orig, n_in = input_spikes.shape
    assert n_in == 140
    T1 = T_orig + DCLS_TIME_EXTENSION  # length after dcls_l1
    T2 = T1 + DCLS_TIME_EXTENSION  # length after dcls_l2
    spikes_l1 = _run_stage1(net, input_spikes, T_orig, T1)
    return _run_stage2(net, spikes_l1, T1, T2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", help="Path to FPGA artifacts directory")
    parser.add_argument(
        "--smoke", action="store_true", help="Run a smoke test with a synthetic input"
    )
    args = parser.parse_args()

    print(f"Loading artifacts from {args.artifacts}")
    net = load_artifacts(args.artifacts)
    print(f"Layer 1: {net.layer1.weights_int8.shape}, scale={net.layer1.scale:.6f}")
    print(f"Layer 2: {net.layer2.weights_int8.shape}, scale={net.layer2.scale:.6f}")
    print(f"Layer 3: {net.layer3.weights_int8.shape}, scale={net.layer3.scale:.6f}")
    print(f"Delays L1: range [{net.delays_l1.min()}, {net.delays_l1.max()}]")
    print(f"Delays L2: range [{net.delays_l2.min()}, {net.delays_l2.max()}]")
    print(f"Softplus LUT: {len(net.softplus_lut)} entries")

    if args.smoke:
        np.random.seed(42)
        T = 100
        input_spikes = (np.random.random((T, 140)) > 0.95).astype(np.int8)
        print(f"\nSmoke test: T={T}, input density={input_spikes.mean():.3f}")
        pred = run_inference_q88(net, input_spikes)
        print(f"Predicted class: {pred}")
