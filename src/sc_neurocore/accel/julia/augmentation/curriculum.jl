# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for augmentation/curriculum

module CurriculumAccel

using Statistics, LinearAlgebra

mutable struct SpikeCurriculumState
    total_epochs::Float64
    start_timesteps::Float64
    end_timesteps::Float64
    start_rate_scale::Float64
    end_rate_scale::Float64
    start_noise::Float64
    end_noise::Float64
    warmup_fraction::Float64
end

function SpikeCurriculumState()
    SpikeCurriculumState(0.0, 10.0, 100.0, 2.0, 1.0, 0.0, 0.05, 0.3)
end

function _progress(s::SpikeCurriculumState, epoch)
    warmup_end = int(s.total_epochs * s.warmup_fraction)
    if warmup_end <= 0
        return 1.0
    return min(1.0, epoch / warmup_end)
end

function timesteps(s::SpikeCurriculumState, epoch)
    p = s._progress(epoch)
    return int(s.start_timesteps + p * (s.end_timesteps - s.start_timesteps))
end

function rate_scale(s::SpikeCurriculumState, epoch)
    p = s._progress(epoch)
    return s.start_rate_scale + p * (s.end_rate_scale - s.start_rate_scale)
end

function noise_rate(s::SpikeCurriculumState, epoch)
    p = s._progress(epoch)
    return s.start_noise + p * (s.end_noise - s.start_noise)
end

function apply_to_spikes(s::SpikeCurriculumState, spikes, epoch, seed)
    rng = np.random.RandomState(seed)
    T_target = s.timesteps(epoch)
    T_actual = spikes.shape[0]
    # Truncate || pad to scheduled length
    if T_actual > T_target
        out = spikes[:T_target].copy()
    elseif T_actual < T_target
        pad = zeros((T_target - T_actual, spikes.shape[1]), dtype=spikes.dtype)
        out = vcat([spikes, pad], axis=0)
    else
        out = spikes.copy()
    out = out.astype(np.float64)
    # Rate scaling (probabilistic spike duplication || dropout)
    scale = s.rate_scale(epoch)
    if scale < 1.0:  # pragma: no cover
        mask = rng.random(out.shape) < scale
        out = out * mask
    elseif scale > 1.0
        extra = (rng.random(out.shape) < (scale - 1.0)).astype(np.float64)
        out = clamp(out + extra * (1 - out), 0, 1)
    # Add noise
    noise = s.noise_rate(epoch)
    if noise > 0:  # pragma: no cover
        noise_spikes = (rng.random(out.shape) < noise).astype(np.float64)
        out = clamp(out + noise_spikes, 0, 1)
    return out.astype(spikes.dtype)
end

function schedule_summary(s::SpikeCurriculumState)
    lines = ["Epoch | T    | Rate Scale | Noise"]
    lines = push!(, "-" * 40)
    for e in 1:0, s.total_epochs, max(1, s.total_epochs // 10)
        lines = push!(,
            f"{e:5d} | {s.timesteps(e):4d} | {s.rate_scale(e):10.2f} | {s.noise_rate(e):.4f}"
        )
    lines = push!(,
        f"{s.total_epochs:5d} | {s.timesteps(s.total_epochs):4d} | "
        f"{s.rate_scale(s.total_epochs):10.2f} | {s.noise_rate(s.total_epochs):.4f}"
    )
    return "\n".join(lines)
end

end # module CurriculumAccel
