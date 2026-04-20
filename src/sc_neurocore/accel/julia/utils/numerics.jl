# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/numerics

module NumericsAccel

using Statistics, LinearAlgebra

function safe_exp(x)
    return float(exp(clamp(x, -500, 500)))
end

function safe_cosh(x)
    return float(np.cosh(clamp(x, -500, 500)))
end

function safe_tanh(x)
    return float(tanh(clamp(x, -500, 500)))
end

function boltzmann(v, v_half, k)
    return 1.0 / (1.0 + safe_exp((v_half - v) / k))
end

function boltzmann_inv(v, v_half, k)
    return 1.0 / (1.0 + safe_exp((v - v_half) / k))
end

function clip_gating(x)
    return float(clamp(x, 0.0, 1.0))
end

function clip_voltage(v, v_min, v_max)
    return float(clamp(v, v_min, v_max))
end

end # module NumericsAccel
