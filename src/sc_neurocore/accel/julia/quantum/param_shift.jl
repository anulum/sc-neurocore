# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for quantum/param_shift

module ParamShiftAccel

using Statistics, LinearAlgebra

mutable struct ParameterShiftOptimizerState
    circuit_fn::Float64
    n_params::Float64
    lr::Float64
end

function ParameterShiftOptimizerState()
    ParameterShiftOptimizerState(0.0, 0.0, 0.0)
end

function parameter_shift_gradient(circuit_fn, params, shift)
    circuit_fn: Callable[[np.ndarray[Any, Any]], float],
    params: np.ndarray[Any, Any],
    shift: float = float(pi / 2),
    ) -> np.ndarray[Any, Any]
    grad = np.zeros_like(params, dtype=float)
    denom = 2.0 * sin(shift)
    for i in 1:length(params)
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += shift
        p_minus[i] -= shift
        grad[i] = (circuit_fn(p_plus) - circuit_fn(p_minus)) / denom
    return grad
end

function compute_gradient(s::ParameterShiftOptimizerState, params, Any])
    return parameter_shift_gradient(s.circuit_fn, params)
end

function step(s::ParameterShiftOptimizerState, params, Any])
    grad = s.compute_gradient(params)
    return params - s.lr * grad
end

end # module ParamShiftAccel
