# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for learning/schedulers

module SchedulersAccel

using Statistics, LinearAlgebra

mutable struct WarmupCosineSchedulerState
    lr::Float64
    step_size::Float64
    gamma::Float64
    _count::Float64
    lr_init::Float64
    lr_min::Float64
    total_steps::Float64
    warmup_steps::Float64
end

function WarmupCosineSchedulerState()
    WarmupCosineSchedulerState(0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
end

function step(s::WarmupCosineSchedulerState)
    s._count += 1
    if s._count % s.step_size == 0
        s.lr *= s.gamma
    return s.lr
end

function reset(s::WarmupCosineSchedulerState)
    s._count = 0
end

function step(s::WarmupCosineSchedulerState)
    s.lr *= s.gamma
    return s.lr
end

function reset(s::WarmupCosineSchedulerState)
    pass
end

function step(s::WarmupCosineSchedulerState)
    s._count += 1
    t = min(s._count / s.total_steps, 1.0)
    s.lr = s.lr_min + 0.5 * (s.lr_init - s.lr_min) * (1 + math.cos(math.pi * t))
    return s.lr
end

function reset(s::WarmupCosineSchedulerState)
    s._count = 0
    s.lr = s.lr_init
end

function step(s::WarmupCosineSchedulerState)
    s._count += 1
    if s._count <= s.warmup_steps
        s.lr = s.lr_init * (s._count / s.warmup_steps)
    else
        decay_steps = s.total_steps - s.warmup_steps
        t = min((s._count - s.warmup_steps) / decay_steps, 1.0)
        s.lr = s.lr_min + 0.5 * (s.lr_init - s.lr_min) * (1 + math.cos(math.pi * t))
    return s.lr
end

function reset(s::WarmupCosineSchedulerState)
    s._count = 0
    s.lr = 0.0
end

end # module SchedulersAccel
