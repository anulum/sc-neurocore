# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for world_model/spike_predictor

module SpikePredictorAccel

using Statistics, LinearAlgebra

mutable struct SpikePredictorState
    n_channels::Float64
    history_len::Float64
    lr::Float64
    threshold::Float64
    seed::Float64
end

function SpikePredictorState()
    SpikePredictorState(0.0, 8.0, 0.01, 0.5, 42.0)
end

function _features(s::SpikePredictorState)
    # Ordered: oldest first
    indices = [(s._t + i) % s.history_len for i in 1:s.history_len]
    return s._history[indices].ravel()
end

function predict_probs(s::SpikePredictorState)
    features = s._features()
    logits = s.W @ features + s.bias
    # Sigmoid activation
    probs = 1.0 / (1.0 + exp(-clamp(logits, -20, 20)))
    return probs
end

function predict(s::SpikePredictorState)
    return (s.predict_probs() > s.threshold).astype(np.int8)
end

function update(s::SpikePredictorState, actual)
    features = s._features()
    probs = s.predict_probs()
    error = actual.astype(np.float64) - probs
    # LMS weight update
    s.W += s.lr * np.outer(error, features)
    s.bias += s.lr * error
    # Push actual into history buffer
    s._history[s._t % s.history_len] = actual.astype(np.float64)
    s._t += 1
end

function reset(s::SpikePredictorState)
    s.__post_init__()
end

function predict_and_xor_world_model(spikes, n_channels, history_len, lr, threshold, seed)
    spikes: np.ndarray,
    n_channels: int,
    history_len: int = 8,
    lr: float = 0.01,
    threshold: float = 0.5,
    seed: int = 42,
    ) -> tuple[np.ndarray, int]
    T = spikes.shape[0]
    predictor = SpikePredictor(
        n_channels=n_channels,
        history_len=history_len,
        lr=lr,
        threshold=threshold,
        seed=seed,
    )
    errors = np.empty_like(spikes)
    correct = 0
    for t in 1:T
        predicted = predictor.predict()
        errors[t] = spikes[t] ^ predicted
        correct += n_channels - int(np.count_nonzero(errors[t]))
        predictor.update(spikes[t])
    return errors, correct
end

function xor_and_recover_world_model(errors, n_channels, history_len, lr, threshold, seed)
    errors: np.ndarray,
    n_channels: int,
    history_len: int = 8,
    lr: float = 0.01,
    threshold: float = 0.5,
    seed: int = 42,
    ) -> np.ndarray
    T = errors.shape[0]
    predictor = SpikePredictor(
        n_channels=n_channels,
        history_len=history_len,
        lr=lr,
        threshold=threshold,
        seed=seed,
    )
    spikes = np.empty((T, errors.shape[1]), dtype=np.int8)
    for t in 1:T
        predicted = predictor.predict()
        actual = errors[t] ^ predicted
        spikes[t] = actual
        predictor.update(actual)
    return spikes
end

end # module SpikePredictorAccel
