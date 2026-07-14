# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fail-closed photonic crosstalk kernel

"""Numeric coupled-waveguide hot path shared with the Python contract."""
module PhotonicEmitterAccel

export PairSpec, PairMetrics, BankMetrics, analyze_pair, analyze_bank, analyze_pairs

const ISOLATION_CEILING_DB = 300.0
const ISOLATION_RATIO_FLOOR = 1.0e-15

struct PairSpec
    index_a::Int
    index_b::Int
    gap_nm::Float64
    coupling_length_um::Float64
end

struct PairMetrics
    coupling_coefficient_per_um::Float64
    coupling_ratio::Float64
    isolation_db::Float64
end

struct BankMetrics
    num_waveguides::Int
    num_near_pairs::Int
    num_far_pairs::Int
    adjacent::PairMetrics
    next_nearest::PairMetrics
    worst_isolation_db::Float64
    mean_coupling_ratio::Float64
    max_coupling_ratio::Float64
    crosstalk_safe::Bool
end

function _nonnegative(value::Float64, name::String)::Nothing
    isfinite(value) && value >= 0.0 || throw(DomainError(value, "$name must be finite and non-negative"))
    return nothing
end

function _material(wavelength_nm::Float64, core_index::Float64, cladding_index::Float64)::Nothing
    isfinite(wavelength_nm) && wavelength_nm > 0.0 || throw(DomainError(wavelength_nm, "wavelength_nm must be finite and positive"))
    isfinite(core_index) && core_index > 0.0 || throw(DomainError(core_index, "core_index must be finite and positive"))
    isfinite(cladding_index) && cladding_index > 0.0 || throw(DomainError(cladding_index, "cladding_index must be finite and positive"))
    core_index > cladding_index || throw(DomainError((core_index, cladding_index), "core_index must be greater than cladding_index"))
    return nothing
end

"""Evaluate the Marcatili-form coupled-mode contract for one pair."""
function analyze_pair(
    gap_nm::Float64,
    coupling_length_um::Float64,
    wavelength_nm::Float64 = 1550.0,
    core_index::Float64 = 3.48,
    cladding_index::Float64 = 1.45,
)::PairMetrics
    _nonnegative(gap_nm, "gap_nm")
    _nonnegative(coupling_length_um, "coupling_length_um")
    _material(wavelength_nm, core_index, cladding_index)
    index_contrast = sqrt(core_index^2 - cladding_index^2)
    decay_length_nm = wavelength_nm / (2.0 * pi * index_contrast)
    effective_index_difference = 0.1 * exp(-gap_nm / decay_length_nm)
    coefficient = pi * effective_index_difference / (wavelength_nm * 1.0e-3)
    ratio = sin(coefficient * coupling_length_um)^2
    isolation = ratio < ISOLATION_RATIO_FLOOR ? ISOLATION_CEILING_DB : -10.0 * log10(ratio)
    return PairMetrics(coefficient, ratio, isolation)
end

"""Evaluate adjacent and next-nearest coupling in a uniform bank."""
function analyze_bank(
    num_waveguides::Int,
    gap_nm::Float64,
    coupling_length_um::Float64,
    wavelength_nm::Float64 = 1550.0,
    core_index::Float64 = 3.48,
    cladding_index::Float64 = 1.45,
)::BankMetrics
    num_waveguides >= 1 || throw(DomainError(num_waveguides, "num_waveguides must be at least one"))
    adjacent = analyze_pair(gap_nm, coupling_length_um, wavelength_nm, core_index, cladding_index)
    next_nearest = analyze_pair(2.0 * gap_nm, coupling_length_um, wavelength_nm, core_index, cladding_index)
    num_near = num_waveguides - 1
    num_far = max(0, num_waveguides - 2)
    pair_count = num_near + num_far
    if pair_count == 0
        worst = Inf
        mean_ratio = 0.0
        max_ratio = 0.0
    else
        worst = min(adjacent.isolation_db, next_nearest.isolation_db)
        mean_ratio = (num_near * adjacent.coupling_ratio + num_far * next_nearest.coupling_ratio) / pair_count
        max_ratio = max(adjacent.coupling_ratio, next_nearest.coupling_ratio)
    end
    return BankMetrics(
        num_waveguides,
        num_near,
        num_far,
        adjacent,
        next_nearest,
        worst,
        mean_ratio,
        max_ratio,
        worst > 20.0,
    )
end

"""Evaluate an arbitrary pair batch after validating the complete request."""
function analyze_pairs(
    pairs::Vector{PairSpec},
    wavelength_nm::Float64 = 1550.0,
    core_index::Float64 = 3.48,
    cladding_index::Float64 = 1.45,
)::Vector{PairMetrics}
    _material(wavelength_nm, core_index, cladding_index)
    for pair in pairs
        pair.index_a >= 0 && pair.index_b >= 0 && pair.index_a != pair.index_b ||
            throw(DomainError((pair.index_a, pair.index_b), "pairs must name distinct non-negative waveguides"))
        _nonnegative(pair.gap_nm, "gap_nm")
        _nonnegative(pair.coupling_length_um, "coupling_length_um")
    end
    return [
        analyze_pair(pair.gap_nm, pair.coupling_length_um, wavelength_nm, core_index, cladding_index)
        for pair in pairs
    ]
end

end
