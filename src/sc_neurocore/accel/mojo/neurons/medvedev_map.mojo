# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Medvedev 2005 1D spiking map (parity with medvedev_map.py)
#
# Build:
#   mojo build --emit shared-lib -o libmedvedev.so medvedev_map.mojo
#
# Parity contract: `medvedev_map_simulate_c` reproduces
# `sc_neurocore.neurons.models.medvedev_map.MedvedevMapNeuron.simulate`. The map
# is exact floating-point arithmetic (a multiply, an add, and a fold into
# [0, 1)). The fold is `x - floor(x)`, which equals Python's `x % 1.0`, Rust's
# `rem_euclid(1.0)` and Julia's `mod(x, 1.0)` bit-for-bit. Rust/Julia/Go
# reproduce the trace bit-for-bit; Mojo's release build can contract
# `alpha*x + current` (and `alpha*(1-x) + current`) into a fused multiply-add
# (one rounding instead of two), so each step agrees to within a couple of ULP.
# This is a chaotic expanding map (alpha > 1), so a single ULP can be amplified
# into a visible whole-trace gap — the per-step agreement stays tightly
# ULP-bounded and the spike counts match. This matches the documented Mojo
# FMA-parity precedent for wong_wang / wilson_cowan.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and
# the pointer is reconstructed inside. The caller allocates n+1 Float64 slots:
# [0, n) receive the x trace, index n the final x.
#
# Reference: Medvedev, G.S. (2005). Physica D 202:37-59.

from std.memory import UnsafePointer
from math import floor


@always_inline
fn _fold_unit(v: Float64) -> Float64:
    # Euclidean remainder modulo 1.0: v - floor(v) lands in [0, 1) and is exact.
    return v - floor(v)


@export
fn medvedev_map_simulate_c(
    x0: Float64,
    alpha: Float64,
    beta: Float64,
    x_threshold: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var x = x0
    var spikes: Int64 = 0
    for t in range(n_steps):
        var x_prev = x
        # Round the product before adding `current` so the compiler cannot
        # contract `alpha*x + current` into a single-rounding fused multiply-add
        # — that fusion is the one operation that diverges from the IEEE-754
        # two-rounding path used by Python/Rust/Go/Julia.
        var raw: Float64
        if x < beta:
            var ax = alpha * x
            raw = ax + current
        else:
            var one_minus_x = 1.0 - x
            var a_term = alpha * one_minus_x
            raw = a_term + current
        x = _fold_unit(raw)
        trace[t] = x
        if x >= x_threshold and x_prev < x_threshold:
            spikes += 1
    if n_steps > 0:
        trace[n_steps] = x
    return spikes
