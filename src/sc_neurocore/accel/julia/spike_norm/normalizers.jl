# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_norm/normalizers

module NormalizersAccel

using Statistics, LinearAlgebra

mutable struct TemporalAccumulatedBNState
    n_features::Float64
    threshold::Float64
    momentum::Float64
    eps::Float64
    T::Float64
end

function TemporalAccumulatedBNState()
    TemporalAccumulatedBNState(0.0, 1.0, 0.1, 1e-05, 0.0)
end

function forward(s::TemporalAccumulatedBNState, x, Any], training)
    if training
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        s.running_mean = (1 - s.momentum) * s.running_mean + s.momentum * mean
        s.running_var = (1 - s.momentum) * s.running_var + s.momentum * var
    else
        mean = s.running_mean
        var = s.running_var
    x_norm = (x - mean) / sqrt(var + s.eps)
    result: np.ndarray[Any, Any] = s.gamma * x_norm * s.threshold + s.beta
    return result
end

function forward(s::TemporalAccumulatedBNState)
    self, x: np.ndarray[Any, Any], t: int, training: bool = true
    ) -> np.ndarray[Any, Any]
    t_idx = min(t, s.T - 1)
    if training
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        s.running_means[t_idx] = 0.9 * s.running_means[t_idx] + 0.1 * mean
        s.running_vars[t_idx] = 0.9 * s.running_vars[t_idx] + 0.1 * var
    else:  # pragma: no cover
        mean = s.running_means[t_idx]
        var = s.running_vars[t_idx]
    x_norm = (x - mean) / sqrt(var + s.eps)
    result: np.ndarray[Any, Any] = s.gammas[t_idx] * x_norm + s.betas[t_idx]
    return result
end

function forward(s::TemporalAccumulatedBNState)
    self, x: np.ndarray[Any, Any], t: int, training: bool = true
    ) -> np.ndarray[Any, Any]
    if training
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        s.running_mean = 0.9 * s.running_mean + 0.1 * mean
        s.running_var = 0.9 * s.running_var + 0.1 * var
    else:  # pragma: no cover
        mean = s.running_mean
        var = s.running_var
    x_norm = (x - mean) / sqrt(var + s.eps)
    t_idx = min(t, s.T - 1)
    result: np.ndarray[Any, Any] = s.lambdas[t_idx] * (s.gamma * x_norm + s.beta)
    return result
end

function forward(s::TemporalAccumulatedBNState)
    self, membrane: np.ndarray[Any, Any], training: bool = true
    ) -> np.ndarray[Any, Any]
    if training
        mean = membrane.mean(axis=0) if membrane.ndim > 1 else membrane
        var = membrane.var(axis=0) if membrane.ndim > 1 else np.zeros_like(membrane)
        s.running_mean = (1 - s.momentum) * s.running_mean + s.momentum * mean
        s.running_var = (1 - s.momentum) * s.running_var + s.momentum * var
        norm = (membrane - mean) / sqrt(var + s.eps)
        result: np.ndarray[Any, Any] = s.gamma * norm + s.beta
        return result
    return membrane
end

function fused_threshold(s::TemporalAccumulatedBNState)
    result: np.ndarray[Any, Any] = (s.threshold - s.beta) * sqrt(
        s.running_var + s.eps
    ) / clamp(s.gamma, 1e-8, nothing) + s.running_mean
    return result
end

function forward(s::TemporalAccumulatedBNState, x, Any], training)
    increment: np.ndarray[Any, Any] = x.mean(axis=0) if x.ndim > 1 else x
    s._accumulated = s._accumulated + increment
    if training
        mean = s._accumulated
        # Variance estimated from current input
        var = x.var(axis=0) if x.ndim > 1 else np.zeros_like(x)
        s.running_mean = (1 - s.momentum) * s.running_mean + s.momentum * mean  # type: ignore[assignment]
        s.running_var = (1 - s.momentum) * s.running_var + s.momentum * var
    else:  # pragma: no cover
        mean = s.running_mean
        var = s.running_var
    x_norm = (x - mean) / sqrt(var + s.eps)
    result: np.ndarray[Any, Any] = s.gamma * x_norm + s.beta
    return result
end

function reset(s::TemporalAccumulatedBNState)
    s._accumulated = zeros(s.n_features)
end

end # module NormalizersAccel
