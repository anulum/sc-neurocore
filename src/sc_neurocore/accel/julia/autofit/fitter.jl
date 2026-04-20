# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for autofit/fitter

module FitterAccel

using Statistics, LinearAlgebra

mutable struct FittedModelState
    model_name::Float64
    model_class::Float64
    params::Float64
    rmse::Float64
    feature_error::Float64
    combined_score::Float64
    simulated_voltage::Float64
    target_features::Float64
    model_features::Float64
end

function FittedModelState()
    FittedModelState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function fit(voltage, current, dt, threshold, candidates, top_k)
    voltage: np.ndarray[Any, Any],
    current: np.ndarray[Any, Any],
    dt: float = 0.1,
    threshold: float = 0.0,
    candidates: list[str] | nothing = nothing,
    top_k: int = 5,
    ) -> list[FittedModel]
    if candidates is nothing
        candidates = _FITTABLE_MODELS
    results = []
    for name in candidates
        cls = _get_model_class(name)
        if cls is nothing
            continue
        try
            result = _fit_single_model(cls, name, voltage, current, dt, threshold)
            if result is ! nothing
                results = push!(, result)
        except (ValueError, TypeError, RuntimeError, ArithmeticError)
            continue
    results.sort(key=lambda r: r.combined_score)
    return results[:top_k]
end

end # module FitterAccel
