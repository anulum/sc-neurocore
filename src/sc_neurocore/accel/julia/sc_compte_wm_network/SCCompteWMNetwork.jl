# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""
Full Julia runtime for the separately named `SC-COMPTE-WM-NETWORK` model.

The module executes the SC 2,048-pyramidal plus 512-interneuron ring. It does
not replace the preserved source-bounded scalar Compte cell. Its transition
matches the Python/Rust v1 contract: counter-addressed Poisson input, midpoint
RK2 channel flow, FFT circular excitation, optional structured E-to-I input,
sampled threshold/reset/refractory behavior, explicit-event co-simulation, and
bounded run statistics. Execution is not by itself evidence of a persistent
bump or distractor resistance.
"""
module SCCompteWMNetwork

using FFTW
using SHA

export N_EXCITATORY, N_INHIBITORY, DT_MS
export SCCompteWMNetworkSpec, SCCompteWMNetworkState, SCCompteWMNetworkRuntime
export SCCompteWMStepReceipt, SCCompteWMActivityStatistics
export SCCompteWMWindowReceipt, SCCompteWMRunReceipt, SCCompteWMStimulus
export counter_poisson_counts, validate_state, state_sha256
export step!, step_with_events!, run!, reset!, cue_current_pa, summarize_activity

"""Fixed pyramidal population size for SC-COMPTE-WM-NETWORK v1."""
const N_EXCITATORY = 2048
"""Fixed inhibitory population size for SC-COMPTE-WM-NETWORK v1."""
const N_INHIBITORY = 512
"""Fixed source timestep in milliseconds."""
const DT_MS = 0.02

const GOLDEN = UInt64(0x9e3779b97f4a7c15)
const STEP_MIX = UInt64(0xd1b54a32d192ed03)
const STREAM_MIX = UInt64(0x94d049bb133111eb)
const V_MIN = -200.0
const V_MAX = 100.0
const GATE_MAX = 1.0e6

"""
Frozen configuration for the Julia v1 network transition.

`modulated=true` scales recurrent NMDA by 1.2 and GABAA by 1.4.
`structured_ei=true` selects the tuned E-to-I footprint. Recurrent E-to-E and
I-to-I autapses are disabled by default.
"""
Base.@kwdef struct SCCompteWMNetworkSpec
    seed::UInt64 = UInt64(42)
    structured_ei::Bool = false
    modulated::Bool = false
    allow_recurrent_autapses::Bool = false
end

"""Complete mutable state of all 2,560 cells and their synaptic gates."""
mutable struct SCCompteWMNetworkState
    step_index::UInt64
    v_exc_mv::Vector{Float64}
    v_inh_mv::Vector{Float64}
    refractory_exc_ms::Vector{Float64}
    refractory_inh_ms::Vector{Float64}
    external_ampa_exc::Vector{Float64}
    external_ampa_inh::Vector{Float64}
    recurrent_nmda::Vector{Float64}
    recurrent_nmda_rise::Vector{Float64}
    recurrent_gabaa::Vector{Float64}
end

"""Construct the leak-equilibrium, zero-gate network state."""
function SCCompteWMNetworkState()
    SCCompteWMNetworkState(
        UInt64(0), fill(-70.0, N_EXCITATORY), fill(-70.0, N_INHIBITORY),
        zeros(N_EXCITATORY), zeros(N_INHIBITORY), zeros(N_EXCITATORY),
        zeros(N_INHIBITORY), zeros(N_EXCITATORY), zeros(N_EXCITATORY),
        zeros(N_INHIBITORY),
    )
end

"""Events and input totals emitted by one successful atomic step."""
struct SCCompteWMStepReceipt
    step_index::UInt64
    excitatory_spikes::BitVector
    inhibitory_spikes::BitVector
    excitatory_input_events::UInt64
    inhibitory_input_events::UInt64
    input_sha256::String
    state_sha256::String
end

"""Population rates and circular bump observables for one bounded window."""
struct SCCompteWMActivityStatistics
    excitatory_rate_hz::Float64
    inhibitory_rate_hz::Float64
    bump_angle_deg::Float64
    resultant_length::Float64
    circular_width_deg::Union{Nothing,Float64}
end

"""One explicitly bounded run-window receipt."""
struct SCCompteWMWindowReceipt
    start_ms::Float64
    end_ms::Float64
    excitatory_spikes::Int
    inhibitory_spikes::Int
    statistics::Union{Nothing,SCCompteWMActivityStatistics}
end

"""Aggregate deterministic evidence returned by `run!`."""
struct SCCompteWMRunReceipt
    specification_version::String
    seed::UInt64
    duration_ms::Float64
    steps::Int
    excitatory_spikes::Int
    inhibitory_spikes::Int
    windows::Vector{SCCompteWMWindowReceipt}
    input_sha256::String
    spike_sha256::String
    final_state_sha256::String
end

"""
One excitatory current epoch in source pA units.

`kind` is `:localized_cue` or `:global_current`. Localized cues require a
finite `center_deg`; global currents require `center_deg=nothing`.
"""
struct SCCompteWMStimulus
    start_ms::Float64
    duration_ms::Float64
    current_pa::Float64
    kind::Symbol
    center_deg::Union{Nothing,Float64}
    function SCCompteWMStimulus(start_ms, duration_ms, current_pa;
                               kind::Symbol=:localized_cue, center_deg=0.0)
        values = Float64[start_ms, duration_ms, current_pa]
        all(isfinite, values) || throw(ArgumentError("stimulus values must be finite"))
        start_ms >= 0 || throw(ArgumentError("stimulus start must be non-negative"))
        duration_ms > 0 || throw(ArgumentError("stimulus duration must be positive"))
        current_pa > 0 || throw(ArgumentError("stimulus current must be positive"))
        kind in (:localized_cue, :global_current) ||
            throw(ArgumentError("stimulus kind must be :localized_cue or :global_current"))
        if kind == :localized_cue
            center_deg !== nothing && isfinite(center_deg) ||
                throw(ArgumentError("localized cues require finite center_deg"))
        elseif center_deg !== nothing
            throw(ArgumentError("global currents require center_deg=nothing"))
        end
        new(Float64(start_ms), Float64(duration_ms), Float64(current_pa), kind,
            center_deg === nothing ? nothing : Float64(center_deg))
    end
end

"""Julia executor with cached FFT spectra and complete public state."""
mutable struct SCCompteWMNetworkRuntime
    spec::SCCompteWMNetworkSpec
    state::SCCompteWMNetworkState
    ee_kernel::Vector{Float64}
    ee_spectrum::Vector{ComplexF64}
    ei_spectrum::Union{Nothing,Vector{ComplexF64}}
end

"""Construct and validate a Julia runtime from an optional checkpoint."""
function SCCompteWMNetworkRuntime(spec::SCCompteWMNetworkSpec=SCCompteWMNetworkSpec();
                                  state::SCCompteWMNetworkState=SCCompteWMNetworkState())
    copied = deepcopy(state)
    ee_kernel = footprint(1.62, 18.0)
    runtime = SCCompteWMNetworkRuntime(
        spec, copied, ee_kernel, fft(ee_kernel),
        spec.structured_ei ? fft(footprint(1.25, 18.0)) : nothing,
    )
    validate_state(runtime.state)
    runtime
end

"""Restore zero-gate leak equilibrium while retaining runtime configuration."""
function reset!(runtime::SCCompteWMNetworkRuntime)
    runtime.state = SCCompteWMNetworkState()
    nothing
end

@inline function splitmix64(value::UInt64)
    z = value + GOLDEN
    z = (z ⊻ (z >> 30)) * UInt64(0xbf58476d1ce4e5b9)
    z = (z ⊻ (z >> 27)) * UInt64(0x94d049bb133111eb)
    z ⊻ (z >> 31)
end

"""
Return portable per-cell aggregate Poisson counts for one counter address.

The inverse CDF terminates below a `1e-15` residual tail and accepts means no
larger than 32 events per cell and timestep.
"""
function counter_poisson_counts(population_size::Integer, rate_hz::Real, dt_ms::Real,
                                seed::UInt64, stream::UInt64, step_index::UInt64)
    population_size > 0 || throw(ArgumentError("population size must be positive"))
    isfinite(rate_hz) && rate_hz >= 0 || throw(ArgumentError("rate must be finite and non-negative"))
    isfinite(dt_ms) && dt_ms > 0 || throw(ArgumentError("dt must be finite and positive"))
    mean = Float64(rate_hz) * Float64(dt_ms) / 1000.0
    mean <= 32.0 || throw(ArgumentError("counter-Poisson mean exceeds safety envelope"))
    probability = exp(-mean)
    cumulative = probability
    cdf = Float64[cumulative]
    count = 0
    while cumulative < 1.0 - 1.0e-15
        count += 1
        count <= 255 || throw(ArgumentError("counter-Poisson inverse CDF exceeded event range"))
        probability *= mean / count
        cumulative += probability
        push!(cdf, min(1.0, cumulative))
    end
    cdf[end] = 1.0
    counts = Vector{UInt64}(undef, population_size)
    for cell in 0:(population_size - 1)
        counter = seed + step_index * STEP_MIX + stream * STREAM_MIX + UInt64(cell) * GOLDEN
        bits = splitmix64(counter)
        uniform = (Float64(bits >> 11) + 0.5) * 2.0^-53
        counts[cell + 1] = UInt64(searchsortedfirst(cdf, uniform) - 1)
    end
    counts
end

"""Fail closed unless every state array has the fixed shape and safety bounds."""
function validate_state(state::SCCompteWMNetworkState)
    lengths = (length(state.v_exc_mv), length(state.v_inh_mv),
        length(state.refractory_exc_ms), length(state.refractory_inh_ms),
        length(state.external_ampa_exc), length(state.external_ampa_inh),
        length(state.recurrent_nmda), length(state.recurrent_nmda_rise),
        length(state.recurrent_gabaa))
    lengths == (N_EXCITATORY, N_INHIBITORY, N_EXCITATORY, N_INHIBITORY,
                N_EXCITATORY, N_INHIBITORY, N_EXCITATORY, N_EXCITATORY,
                N_INHIBITORY) || throw(ArgumentError("invalid SC Compte network state shape"))
    all(v -> isfinite(v) && V_MIN <= v <= V_MAX, state.v_exc_mv) ||
        throw(ArgumentError("invalid excitatory voltage state"))
    all(v -> isfinite(v) && V_MIN <= v <= V_MAX, state.v_inh_mv) ||
        throw(ArgumentError("invalid inhibitory voltage state"))
    gates = (state.refractory_exc_ms, state.refractory_inh_ms,
             state.external_ampa_exc, state.external_ampa_inh,
             state.recurrent_nmda_rise, state.recurrent_gabaa)
    all(values -> all(v -> isfinite(v) && 0.0 <= v <= GATE_MAX, values), gates) ||
        throw(ArgumentError("invalid refractory or channel state"))
    all(v -> isfinite(v) && 0.0 <= v <= 1.0, state.recurrent_nmda) ||
        throw(ArgumentError("invalid recurrent NMDA state"))
    nothing
end

"""Return a canonical SHA-256 digest of every state scalar and Float64 array."""
function state_sha256(state::SCCompteWMNetworkState)
    io = IOBuffer()
    write(io, htol(state.step_index))
    for values in (state.v_exc_mv, state.v_inh_mv, state.refractory_exc_ms,
                   state.refractory_inh_ms, state.external_ampa_exc,
                   state.external_ampa_inh, state.recurrent_nmda,
                   state.recurrent_nmda_rise, state.recurrent_gabaa)
        for value in values
            write(io, htol(reinterpret(UInt64, value)))
        end
    end
    bytes2hex(sha256(take!(io)))
end

function footprint(j_plus::Float64, sigma_deg::Float64)
    gaussian = [exp(-0.5 * (((index * 360.0 / N_EXCITATORY + 180.0) % 360.0 - 180.0) /
                           sigma_deg)^2) for index in 0:(N_EXCITATORY - 1)]
    gaussian_mean = sum(gaussian) / N_EXCITATORY
    j_minus = (1.0 - j_plus * gaussian_mean) / (1.0 - gaussian_mean)
    weights = j_minus .+ (j_plus - j_minus) .* gaussian
    weights ./ (sum(weights) / N_EXCITATORY)
end

@inline mg_block(voltage::Float64) = 1.0 / (1.0 + exp(clamp(-0.062 * voltage, -700.0, 700.0)) / 3.57)

function circular_sum(source::Vector{Float64}, spectrum::Vector{ComplexF64})
    real.(ifft(fft(source) .* spectrum))
end

function aggregates(runtime::SCCompteWMNetworkRuntime, nmda, gabaa)
    ee = circular_sum(nmda, runtime.ee_spectrum)
    runtime.spec.allow_recurrent_autapses || (ee .-= runtime.ee_kernel[1] .* nmda)
    ei = if runtime.ei_spectrum === nothing
        fill(sum(nmda), N_INHIBITORY)
    else
        circular_sum(nmda, runtime.ei_spectrum)[1:4:end]
    end
    total_gabaa = sum(gabaa)
    ie = fill(total_gabaa, N_EXCITATORY)
    ii = fill(total_gabaa, N_INHIBITORY)
    runtime.spec.allow_recurrent_autapses || (ii .-= gabaa)
    ee, ei, ie, ii
end

function derivatives(runtime, v_exc, v_inh, ext_exc, ext_inh, nmda, nmda_rise,
                     gabaa, current_na, active_exc, active_inh)
    ee, ei, ie, ii = aggregates(runtime, nmda, gabaa)
    nmda_scale = runtime.spec.modulated ? 1.2 : 1.0
    gaba_scale = runtime.spec.modulated ? 1.4 : 1.0
    dv_exc = similar(v_exc)
    dv_inh = similar(v_inh)
    @inbounds for index in eachindex(v_exc)
        v = v_exc[index]
        dv_exc[index] = active_exc[index] ?
            (-0.025 * (v + 70.0) - 0.0031 * ext_exc[index] * v -
             0.000381 * nmda_scale * ee[index] * mg_block(v) * v -
             0.001336 * gaba_scale * ie[index] * (v + 70.0) + current_na[index]) / 0.5 : 0.0
    end
    @inbounds for index in eachindex(v_inh)
        v = v_inh[index]
        dv_inh[index] = active_inh[index] ?
            (-0.020 * (v + 70.0) - 0.00238 * ext_inh[index] * v -
             0.000292 * nmda_scale * ei[index] * mg_block(v) * v -
             0.001024 * gaba_scale * ii[index] * (v + 70.0)) / 0.2 : 0.0
    end
    (dv_exc, dv_inh, -ext_exc ./ 2.0, -ext_inh ./ 2.0,
     -nmda ./ 100.0 .+ 0.5 .* nmda_rise .* (1.0 .- nmda),
     -nmda_rise ./ 2.0, -gabaa ./ 10.0)
end

function digest_inputs(exc_events, inh_events, current_pa)
    io = IOBuffer()
    for event in exc_events
        write(io, htol(UInt64(event)))
    end
    for event in inh_events
        write(io, htol(UInt64(event)))
    end
    for value in current_pa
        write(io, htol(reinterpret(UInt64, Float64(value))))
    end
    bytes2hex(sha256(take!(io)))
end

"""Advance one atomic step using canonical counter-addressed Poisson input."""
function step!(runtime::SCCompteWMNetworkRuntime,
               direct_exc_current_pa::AbstractVector{<:Real}=zeros(N_EXCITATORY))
    exc_events = counter_poisson_counts(N_EXCITATORY, 1800.0, DT_MS,
                                        runtime.spec.seed, UInt64(0), runtime.state.step_index)
    inh_events = counter_poisson_counts(N_INHIBITORY, 1800.0, DT_MS,
                                        runtime.spec.seed, UInt64(1), runtime.state.step_index)
    step_with_events!(runtime, direct_exc_current_pa, exc_events, inh_events)
end

"""
Advance one atomic step with explicit per-cell event counts.

This boundary bypasses the counter generator for independent-oracle and
co-simulation tests. All input and candidate validation precedes mutation.
"""
function step_with_events!(runtime::SCCompteWMNetworkRuntime,
                           direct_exc_current_pa::AbstractVector{<:Real},
                           external_exc_events::AbstractVector{<:Integer},
                           external_inh_events::AbstractVector{<:Integer})
    validate_state(runtime.state)
    length(direct_exc_current_pa) == N_EXCITATORY || throw(ArgumentError("invalid current shape"))
    length(external_exc_events) == N_EXCITATORY || throw(ArgumentError("invalid excitatory event shape"))
    length(external_inh_events) == N_INHIBITORY || throw(ArgumentError("invalid inhibitory event shape"))
    all(isfinite, direct_exc_current_pa) || throw(ArgumentError("current must be finite"))
    all(>=(0), external_exc_events) || throw(ArgumentError("events must be non-negative"))
    all(>=(0), external_inh_events) || throw(ArgumentError("events must be non-negative"))
    current_pa = Float64.(direct_exc_current_pa)
    exc_events = UInt64.(external_exc_events)
    inh_events = UInt64.(external_inh_events)
    state = runtime.state
    start = (copy(state.v_exc_mv), copy(state.v_inh_mv),
             state.external_ampa_exc .+ exc_events, state.external_ampa_inh .+ inh_events,
             copy(state.recurrent_nmda), copy(state.recurrent_nmda_rise),
             copy(state.recurrent_gabaa))
    active_exc = state.refractory_exc_ms .<= 0.0
    active_inh = state.refractory_inh_ms .<= 0.0
    current_na = current_pa ./ 1000.0
    k1 = derivatives(runtime, start..., current_na, active_exc, active_inh)
    midpoint = map((value, slope) -> value .+ 0.5 * DT_MS .* slope, start, k1)
    k2 = derivatives(runtime, midpoint..., current_na, active_exc, active_inh)
    candidate = map((value, slope) -> value .+ DT_MS .* slope, start, k2)
    v_exc, v_inh, ext_exc, ext_inh, nmda, nmda_rise, gabaa = candidate
    ref_exc = max.(0.0, state.refractory_exc_ms .- DT_MS)
    ref_inh = max.(0.0, state.refractory_inh_ms .- DT_MS)
    v_exc[.!active_exc] .= -60.0
    v_inh[.!active_inh] .= -60.0
    exc_spikes = BitVector(active_exc .& (v_exc .>= -50.0))
    inh_spikes = BitVector(active_inh .& (v_inh .>= -50.0))
    v_exc[exc_spikes] .= -60.0
    v_inh[inh_spikes] .= -60.0
    ref_exc[exc_spikes] .= 2.0
    ref_inh[inh_spikes] .= 1.0
    nmda_rise .+= exc_spikes
    gabaa .+= inh_spikes
    state.step_index == typemax(UInt64) && throw(ArgumentError("step counter overflow"))
    next_state = SCCompteWMNetworkState(
        state.step_index + UInt64(1), v_exc, v_inh, ref_exc, ref_inh,
        ext_exc, ext_inh, nmda, nmda_rise, gabaa,
    )
    validate_state(next_state)
    receipt = SCCompteWMStepReceipt(
        state.step_index, exc_spikes, inh_spikes, sum(exc_events), sum(inh_events),
        digest_inputs(exc_events, inh_events, current_pa), state_sha256(next_state),
    )
    runtime.state = next_state
    receipt
end

"""Return the compact SC raised-cosine cue current on the excitatory ring."""
function cue_current_pa(center_deg::Real; peak_pa::Real=200.0)
    isfinite(center_deg) || throw(ArgumentError("center must be finite"))
    isfinite(peak_pa) && peak_pa > 0 || throw(ArgumentError("peak must be finite and positive"))
    current = Vector{Float64}(undef, N_EXCITATORY)
    for index in 0:(N_EXCITATORY - 1)
        angle = index * 360.0 / N_EXCITATORY
        distance = abs((angle - center_deg + 180.0) % 360.0 - 180.0)
        phase = min(distance / 18.0, 1.0)
        current[index + 1] = distance >= 18.0 ? 0.0 : 0.5 * peak_pa * (1.0 + cospi(phase))
    end
    current
end

"""Compute firing rates and circular observables for one non-empty window."""
function summarize_activity(exc_counts::AbstractVector{<:Integer},
                            inh_counts::AbstractVector{<:Integer}, window_ms::Real)
    length(exc_counts) == N_EXCITATORY || throw(ArgumentError("invalid excitatory count shape"))
    length(inh_counts) == N_INHIBITORY || throw(ArgumentError("invalid inhibitory count shape"))
    all(>=(0), exc_counts) && all(>=(0), inh_counts) || throw(ArgumentError("counts must be non-negative"))
    isfinite(window_ms) && window_ms > 0 || throw(ArgumentError("window must be finite and positive"))
    total_exc = sum(exc_counts)
    total_exc > 0 || throw(ArgumentError("bump statistics require an excitatory spike"))
    vector = sum(exc_counts[index + 1] * cis(2pi * index / N_EXCITATORY)
                 for index in 0:(N_EXCITATORY - 1))
    angle = mod(rad2deg(Base.angle(vector)), 360.0)
    resultant = min(1.0, abs(vector) / total_exc)
    width = resultant <= 0.0 ? nothing : rad2deg(sqrt(-2.0 * log(resultant)))
    seconds = window_ms / 1000.0
    SCCompteWMActivityStatistics(total_exc / (N_EXCITATORY * seconds),
        sum(inh_counts) / (N_INHIBITORY * seconds), angle, resultant, width)
end

function stimulus_current(time_ms, stimuli)
    current = zeros(N_EXCITATORY)
    for stimulus in stimuli
        if stimulus.start_ms <= time_ms < stimulus.start_ms + stimulus.duration_ms
            if stimulus.kind == :global_current
                current .+= stimulus.current_pa
            else
                current .+= cue_current_pa(something(stimulus.center_deg); peak_pa=stimulus.current_pa)
            end
        end
    end
    current
end

"""Execute integral timesteps and return bounded input, spike, state, and window receipts."""
function run!(runtime::SCCompteWMNetworkRuntime, duration_ms::Real;
              stimuli::AbstractVector{SCCompteWMStimulus}=SCCompteWMStimulus[],
              statistics_window_ms::Real=500.0)
    isfinite(duration_ms) && duration_ms > 0 || throw(ArgumentError("duration must be finite and positive"))
    raw_steps = duration_ms / DT_MS
    steps = round(Int, raw_steps)
    isapprox(raw_steps, steps; rtol=0.0, atol=1.0e-10) || throw(ArgumentError("duration must contain integral timesteps"))
    raw_window = statistics_window_ms / DT_MS
    window_steps = round(Int, raw_window)
    isfinite(statistics_window_ms) && statistics_window_ms > 0 &&
        isapprox(raw_window, window_steps; rtol=0.0, atol=1.0e-10) ||
        throw(ArgumentError("statistics window must contain integral timesteps"))
    all(s -> s.start_ms + s.duration_ms <= duration_ms + 1.0e-12, stimuli) ||
        throw(ArgumentError("stimulus lies outside run"))
    input_context = SHA2_256_CTX()
    spike_context = SHA2_256_CTX()
    exc_window = zeros(Int, N_EXCITATORY)
    inh_window = zeros(Int, N_INHIBITORY)
    total_exc = 0
    total_inh = 0
    windows = SCCompteWMWindowReceipt[]
    window_start = 0
    for offset in 0:(steps - 1)
        receipt = step!(runtime, stimulus_current(offset * DT_MS, stimuli))
        update!(input_context, hex2bytes(receipt.input_sha256))
        update!(spike_context, UInt8.(receipt.excitatory_spikes))
        update!(spike_context, UInt8.(receipt.inhibitory_spikes))
        exc_window .+= receipt.excitatory_spikes
        inh_window .+= receipt.inhibitory_spikes
        total_exc += count(receipt.excitatory_spikes)
        total_inh += count(receipt.inhibitory_spikes)
        if (offset + 1) % window_steps == 0 || offset + 1 == steps
            elapsed_ms = (offset + 1 - window_start) * DT_MS
            statistics = sum(exc_window) == 0 ? nothing :
                summarize_activity(exc_window, inh_window, elapsed_ms)
            push!(windows, SCCompteWMWindowReceipt(window_start * DT_MS,
                (offset + 1) * DT_MS, sum(exc_window), sum(inh_window), statistics))
            fill!(exc_window, 0)
            fill!(inh_window, 0)
            window_start = offset + 1
        end
    end
    SCCompteWMRunReceipt("sc-neurocore.sc-compte-wm-network.v1", runtime.spec.seed,
        Float64(duration_ms), steps, total_exc, total_inh, windows,
        bytes2hex(digest!(input_context)), bytes2hex(digest!(spike_context)),
        state_sha256(runtime.state))
end

end # module
