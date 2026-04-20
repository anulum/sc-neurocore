# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for online_learning/eprop

module EpropAccel

using Statistics, LinearAlgebra

mutable struct EpropTrainerState
    n_inputs::Float64
    n_neurons::Float64
    n_outputs::Float64
    tau_mem::Float64
    tau_trace::Float64
    threshold::Float64
    lr::Float64
    dt::Float64
    W_in::Float64
    W_rec::Float64
    W_out::Float64
    _v::Float64
    _spikes::Float64
    _trace_in::Float64
    _trace_rec::Float64
end

function EpropTrainerState()
    EpropTrainerState(0.0, 0.0, 0.0, 20.0, 20.0, 1.0, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function reset(s::EpropTrainerState)
    s._v = zeros(s.n_neurons)
    s._spikes = zeros(s.n_neurons)
    s._trace_in = zeros((s.n_neurons, s.n_inputs))
    s._trace_rec = zeros((s.n_neurons, s.n_neurons))
    s._eligibility_in = zeros((s.n_neurons, s.n_inputs))
    s._eligibility_rec = zeros((s.n_neurons, s.n_neurons))
end

function step(s::EpropTrainerState)
    self, x: np.ndarray[Any, Any], target: np.ndarray[Any, Any] | nothing = nothing
    ) -> dict[str, Any]
    alpha = exp(-s.dt / s.tau_mem)
    kappa = exp(-s.dt / s.tau_trace)
    # LIF dynamics
    current = s.W_in @ x + s.W_rec @ s._spikes
    s._v = alpha * s._v + (1 - alpha) * current
    new_spikes = (s._v >= s.threshold).astype(np.float64)
    s._v -= new_spikes * s.threshold
    # Surrogate gradient: pseudo-derivative of spike function
    pseudo_deriv = 1.0 / (1.0 + abs(s._v - s.threshold) * 5) ^ 2
    # Update eligibility traces (low-pass filtered outer products)
    s._trace_in = kappa * s._trace_in + np.outer(pseudo_deriv, x)
    s._trace_rec = kappa * s._trace_rec + np.outer(pseudo_deriv, s._spikes)
    s._eligibility_in = kappa * s._eligibility_in + s._trace_in
    s._eligibility_rec = kappa * s._eligibility_rec + s._trace_rec
    s._spikes = new_spikes
    # Readout
    output = s.W_out @ s._spikes
    result: dict[str, Any] = {"spikes": s._spikes.copy(), "output": output}
    if target is ! nothing
        error = output - target
        loss = 0.5 * float(sum(error^2))
        result["loss"] = loss
        # Learning signal: broadcast error through output weights
        learning_signal = s.W_out.T @ error  # (n_neurons,)
        # Three-factor update: learning_signal * eligibility
        dW_in = np.outer(learning_signal, ones(s.n_inputs)) * s._eligibility_in
        dW_rec = np.outer(learning_signal, ones(s.n_neurons)) * s._eligibility_rec
        dW_out = np.outer(error, s._spikes)
        s.W_in -= s.lr * dW_in
        s.W_rec -= s.lr * dW_rec
        np.fill_diagonal(s.W_rec, 0)
        s.W_out -= s.lr * dW_out
    return result
end

function train_sequence(s::EpropTrainerState, inputs, Any], targets, Any])
    s.reset()
    total_loss = 0.0
    T: int = int(inputs.shape[0])
    for t in 1:T
        result = s.step(inputs[t], target=targets[t])
        total_loss += float(result.get("loss", 0.0))
    return total_loss / T
end

function predict_sequence(s::EpropTrainerState, inputs, Any])
    s.reset()
    T = inputs.shape[0]
    outputs = zeros((T, s.n_outputs))
    for t in 1:T
        result = s.step(inputs[t])
        outputs[t] = result["output"]
    return outputs
end

function memory_per_step(s::EpropTrainerState)
    return (
        s.n_neurons  # membrane voltages
        + s.n_neurons  # spikes
        + s.n_neurons * s.n_inputs * 2  # traces + eligibilities (in)
        + s.n_neurons * s.n_neurons * 2  # traces + eligibilities (rec)
    )
end

end # module EpropAccel
