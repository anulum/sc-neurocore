# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo ADC-to-spike decimating rate-code encoder

# Bit-exact Mojo port of the per-window ADC-to-spike encoder in
# src/sc_neurocore/sensors/adc_to_spike_kernel.py and engine/src/adc_to_spike.rs.
#
# The per-window quantise/average/rate-code arithmetic is exact integer — no
# transcendental path — so this backend matches the Rust, Julia, Go and Python
# references bit-for-bit. Mojo `//` is floor division, so the sign-aware window
# average uses an explicit truncate-toward-zero form to match the reference.
#
# Mojo @export rejects parametric signatures, so every numpy buffer arrives as a
# raw Int address; reconstruct with `UnsafePointer[T, MutAnyOrigin](
# unsafe_from_address=addr)` inside the function body.

from std.memory import UnsafePointer


@always_inline
fn quantise_adc(
    sample: Int64,
    adc_width: Int64,
    q_int: Int64,
    q_frac: Int64,
    signed_input: Int64,
    q_min: Int64,
    q_max: Int64,
) -> Int64:
    var q_total = q_int + q_frac
    var centred: Int64
    if signed_input != 0:
        var sign_bit = Int64(1) << (adc_width - 1)
        var mask = (Int64(1) << adc_width) - 1
        var masked = sample & mask
        if (masked & sign_bit) != 0:
            centred = masked - (Int64(1) << adc_width)
        else:
            centred = masked
    else:
        centred = sample - (Int64(1) << (adc_width - 1))

    var rounded: Int64
    if q_total > adc_width:
        rounded = centred << (q_total - adc_width)
    elif adc_width > q_total:
        var shift = adc_width - q_total
        var half = Int64(1) << (shift - 1)
        if centred >= 0:
            rounded = (centred + half) >> shift
        else:
            rounded = (centred - half) >> shift
    else:
        rounded = centred

    if rounded < q_min:
        return q_min
    if rounded > q_max:
        return q_max
    return rounded


@always_inline
fn average_window(total: Int64, decimation: Int64, q_min: Int64, q_max: Int64) -> Int64:
    var half = decimation // 2  # decimation > 0, floor == truncation
    var adjusted = total + half if total >= 0 else total - half
    # Truncate toward zero (Mojo // is floor; use the magnitude form).
    var magnitude = adjusted if adjusted >= 0 else -adjusted
    var quotient = magnitude // decimation
    var averaged = quotient if adjusted >= 0 else -quotient
    if averaged < q_min:
        return q_min
    if averaged > q_max:
        return q_max
    return averaged


@export
fn adc_to_spike_windows_c(
    n_windows: Int,
    adc_width: Int,
    q_int: Int,
    q_frac: Int,
    decimation: Int,
    signed_input: Int,
    threshold_q: Int,
    samples_addr: Int,
    window_values_addr: Int,
    spike_counts_addr: Int,
    polarities_addr: Int,
) -> Int:
    var samples = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=samples_addr)
    var window_values = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=window_values_addr
    )
    var spike_counts = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=spike_counts_addr
    )
    var polarities = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=polarities_addr
    )

    if n_windows <= 0 or adc_width <= 1 or q_int <= 0 or decimation <= 0 or threshold_q <= 0:
        return 1

    var aw = Int64(adc_width)
    var qi = Int64(q_int)
    var qf = Int64(q_frac)
    var decim = Int64(decimation)
    var signed_flag = Int64(signed_input)
    var threshold = Int64(threshold_q)
    var half_q = Int64(1) << (qi + qf - 1)
    var q_min = -half_q
    var q_max = half_q - 1

    for w in range(n_windows):
        var base = w * decimation
        var total = Int64(0)
        for k in range(decimation):
            total += quantise_adc(samples[base + k], aw, qi, qf, signed_flag, q_min, q_max)
        var wq = average_window(total, decim, q_min, q_max)
        window_values[w] = Int32(wq)
        var magnitude = wq if wq >= 0 else -wq
        spike_counts[w] = Int32(magnitude // threshold)
        polarities[w] = UInt8(1) if wq < 0 else UInt8(0)

    return 0
