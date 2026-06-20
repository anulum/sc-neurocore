# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia ADC-to-spike decimating rate-code encoder

"""
Bit-exact Julia port of the per-window ADC-to-spike encoder in
`src/sc_neurocore/sensors/adc_to_spike_kernel.py` and `engine/src/adc_to_spike.rs`.

Each decimation window is quantised, sign-aware averaged (`div` truncates toward
zero) and converted into a deterministic rate code. The arithmetic is exact
integer, so this backend matches the Python, Rust, Go and Mojo references
bit-for-bit.
"""
module AdcToSpikeAccel

export adc_to_spike_windows!

@inline function quantise_adc(
    sample::Int64,
    adc_width::Int64,
    q_int::Int64,
    q_frac::Int64,
    signed_input::Int64,
    q_min::Int64,
    q_max::Int64,
)::Int64
    q_total = q_int + q_frac
    if signed_input != 0
        sign_bit = Int64(1) << (adc_width - 1)
        mask = (Int64(1) << adc_width) - 1
        masked = sample & mask
        centred = (masked & sign_bit != 0) ? masked - (Int64(1) << adc_width) : masked
    else
        centred = sample - (Int64(1) << (adc_width - 1))
    end

    if q_total > adc_width
        rounded = centred << (q_total - adc_width)
    elseif adc_width > q_total
        shift = adc_width - q_total
        half = Int64(1) << (shift - 1)
        rounded = centred >= 0 ? (centred + half) >> shift : (centred - half) >> shift
    else
        rounded = centred
    end
    return clamp(rounded, q_min, q_max)
end

@inline function average_window(total::Int64, decimation::Int64, q_min::Int64, q_max::Int64)::Int64
    half = div(decimation, 2)
    adjusted = total >= 0 ? total + half : total - half
    averaged = div(adjusted, decimation)  # truncate toward zero
    return clamp(averaged, q_min, q_max)
end

"""Per-window ADC-to-spike encode; fills the pre-allocated output buffers in place."""
function adc_to_spike_windows!(
    samples::AbstractVector{<:Integer},
    adc_width::Integer,
    q_int::Integer,
    q_frac::Integer,
    decimation::Integer,
    signed_input::Integer,
    threshold_q::Integer,
    window_values::AbstractVector{<:Integer},
    spike_counts::AbstractVector{<:Integer},
    polarities::AbstractVector{<:Integer},
)
    aw = Int64(adc_width)
    qi = Int64(q_int)
    qf = Int64(q_frac)
    decim = Int64(decimation)
    signed_flag = Int64(signed_input)
    threshold = Int64(threshold_q)
    half_q = Int64(1) << (qi + qf - 1)
    q_min = -half_q
    q_max = half_q - 1
    n_windows = length(window_values)
    @inbounds for w in 0:(n_windows - 1)
        base = w * decim
        total = Int64(0)
        for k in 0:(decim - 1)
            total += quantise_adc(Int64(samples[base + k + 1]), aw, qi, qf, signed_flag, q_min, q_max)
        end
        wq = average_window(total, decim, q_min, q_max)
        window_values[w + 1] = wq
        spike_counts[w + 1] = div(abs(wq), threshold)
        polarities[w + 1] = wq < 0 ? 1 : 0
    end
    return nothing
end

end # module AdcToSpikeAccel
