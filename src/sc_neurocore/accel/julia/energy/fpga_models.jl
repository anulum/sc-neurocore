# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for energy/fpga_models

module FpgaModelsAccel

using Statistics, LinearAlgebra

mutable struct ModuleCostState
    name::Float64
    family::Float64
    total_luts::Float64
    total_bram_kb::Float64
    total_dsp::Float64
    voltage::Float64
    max_freq_mhz::Float64
    c_eff_per_lut_ff::Float64
    luts::Float64
    ffs::Float64
    bram_bits::Float64
    description::Float64
end

function ModuleCostState()
    ModuleCostState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

end # module FpgaModelsAccel
