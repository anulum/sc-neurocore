# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo fault injection (parity with FaultInjector.inject)

# Build:
#   mojo build --emit shared-lib -o libfault.so fault.mojo
#
# Then load from Python via ctypes:
#   import ctypes, numpy as np
#   lib = ctypes.CDLL("libfault.so")
#   lib.inject_bitflip_c.argtypes = [ctypes.c_int64, ctypes.c_int64,
#                                    ctypes.c_double, ctypes.c_uint64]
#   lib.inject_bitflip_c.restype = ctypes.c_uint64
#   bs = np.zeros(1_000_000, dtype=np.uint8)
#   n = lib.inject_bitflip_c(bs.ctypes.data, len(bs), 1e-3, 42)
#
# Mojo 0.26 FFI rules (per feedback_mojo_026_ffi_pattern):
#   - @export rejects parametric signatures, so we accept the array
#     pointer as a raw `Int` address and reconstruct the typed
#     UnsafePointer inside with explicit MutAnyOrigin.
#   - Implicit stdlib imports are deprecated; fully qualify via std.
#
# RNG: SplitMix64-style LCG seeded from `seed` so back-to-back calls
# with the same seed are reproducible. Bitwise parity with NumPy
# PCG64 / Rust Xoshiro256++ / Julia Xoshiro / Go ChaCha8 is impossible
# — the bench harness verifies statistical parity (4σ Binomial bound).

from std.memory import UnsafePointer


fn _next_u64(rng_state: UnsafePointer[UInt64, MutAnyOrigin]) -> UInt64:
    """Advance a 64-bit LCG one step and return the new value."""
    rng_state[0] = rng_state[0] * 6364136223846793005 + 1442695040888963407
    return rng_state[0]


fn _next_uniform(rng_state: UnsafePointer[UInt64, MutAnyOrigin]) -> Float64:
    """Sample a Float64 uniform in [0, 1) from the LCG."""
    var raw = _next_u64(rng_state)
    # Take top 53 bits → exact mantissa precision for Float64.
    return Float64(raw >> 11) / Float64(1 << 53)


fn _next_normal(rng_state: UnsafePointer[UInt64, MutAnyOrigin]) -> Float64:
    """Box-Muller normal sample (one of two; we discard the second
    rather than caching for simplicity — at 1 Mbit the throughput is
    still dominated by the LCG)."""
    var u1 = _next_uniform(rng_state)
    var u2 = _next_uniform(rng_state)
    # Avoid log(0).
    if u1 <= 0.0:
        u1 = 2.220446049250313e-16
    from std.math import log, sqrt, cos, pi
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


@export
fn inject_bitflip_c(addr: Int, n: Int, ber: Float64, seed: UInt64) -> UInt64:
    var ptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)
    if ber <= 0.0:
        return 0
    var rng = UInt64(seed)
    var rng_ptr = UnsafePointer(to=rng)
    var flipped: UInt64 = 0
    for i in range(n):
        if _next_uniform(rng_ptr) < ber:
            ptr[i] ^= 1
            flipped += 1
    return flipped


@export
fn inject_stuck_at_0_c(addr: Int, n: Int, ber: Float64, seed: UInt64) -> UInt64:
    var ptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)
    if ber <= 0.0:
        return 0
    var rng = UInt64(seed)
    var rng_ptr = UnsafePointer(to=rng)
    var affected: UInt64 = 0
    for i in range(n):
        if _next_uniform(rng_ptr) < ber:
            if ptr[i] != 0:
                affected += 1
            ptr[i] = 0
    return affected


@export
fn inject_stuck_at_1_c(addr: Int, n: Int, ber: Float64, seed: UInt64) -> UInt64:
    var ptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)
    if ber <= 0.0:
        return 0
    var rng = UInt64(seed)
    var rng_ptr = UnsafePointer(to=rng)
    var affected: UInt64 = 0
    for i in range(n):
        if _next_uniform(rng_ptr) < ber:
            if ptr[i] == 0:
                affected += 1
            ptr[i] = 1
    return affected


@export
fn inject_dropout_c(addr: Int, n: Int, ber: Float64, seed: UInt64) -> UInt64:
    return inject_stuck_at_0_c(addr, n, ber, seed)


@export
fn inject_gaussian_c(addr: Int, n: Int, ber: Float64, seed: UInt64) -> UInt64:
    """Add N(0, σ=ber) to bitstream cast to Float64, clip to [0,1],
    threshold at 0.5. Returns count of flipped bits."""
    var ptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)
    if ber <= 0.0:
        return 0
    var rng = UInt64(seed)
    var rng_ptr = UnsafePointer(to=rng)
    var flipped: UInt64 = 0
    for i in range(n):
        var original = ptr[i]
        var noisy = Float64(original) + ber * _next_normal(rng_ptr)
        if noisy < 0.0:
            noisy = 0.0
        elif noisy > 1.0:
            noisy = 1.0
        var new_bit: UInt8 = UInt8(1) if noisy > 0.5 else UInt8(0)
        if new_bit != original:
            flipped += 1
        ptr[i] = new_bit
    return flipped
