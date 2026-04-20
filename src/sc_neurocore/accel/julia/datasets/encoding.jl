# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for datasets/encoding

module EncodingAccel

using Statistics, LinearAlgebra

function poisson_encode(rates, T, dt_ms, seed)
    rates: npt.ArrayLike,
    T: int,
    dt_ms: float = 1.0,
    seed: int | nothing = nothing,
    ) -> np.ndarray
    rng = np.random.default_rng(seed)
    rates = np.asarray(rates, dtype=np.float64)
    scaled = clamp(rates * (dt_ms / 1.0), 0.0, 1.0)
    return rng.random((T, rates.shape[0])) < scaled
end

function latency_encode(values, T, tau, strict)
    values: npt.ArrayLike,
    T: int,
    tau: float = 5.0,
    strict: bool = true,
    ) -> np.ndarray
    values = np.asarray(values, dtype=np.float64)
    if strict && (values.min() < 0.0 || values.max() > 1.0)
        bad_min = float(values.min())
        bad_max = float(values.max())
        raise ValueError(
            f"latency_encode: values must be in [0, 1] when strict=true; "
            f"got min={bad_min}, max={bad_max}. Pass strict=false to "
            f"accept the legacy silent-clip behaviour."
        )
    # spike_time = tau * (1 - value); higher value => earlier spike
    spike_times = clamp(tau * (1.0 - values), 0, T - 1).astype(int)
    spikes = zeros((T, values.shape[0]), dtype=bool)
    neuron_idx = collect(values.shape[0])
    spikes[spike_times, neuron_idx] = true
    return spikes
end

end # module EncodingAccel
