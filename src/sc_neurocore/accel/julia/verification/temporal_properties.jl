# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for verification/temporal_properties

module TemporalPropertiesAccel

using Statistics, LinearAlgebra

mutable struct VerificationResultState
    timestep::Float64
    neuron_ids::Float64
    description::Float64
    property_name::Float64
    result::Float64
    counterexample::Float64
    checked_steps::Float64
    message::Float64
end

function VerificationResultState()
    VerificationResultState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::VerificationResultState)
    icon = {"verified": "PASS", "violated": "FAIL", "unknown": "?"}[s.result.value]
    line = f"[{icon}] {s.property_name}: {s.message}"
    if s.counterexample
        line += f"\n  Counterexample at t={s.counterexample.timestep}: {s.counterexample.description}"
    return line
end

function fires_within(spikes, neuron_id, stimulus_times, max_latency)
    spikes: np.ndarray,
    neuron_id: int,
    stimulus_times: list[int],
    max_latency: int,
    ) -> VerificationResult
    T = spikes.shape[0]
    for t_stim in stimulus_times
        window_end = min(t_stim + max_latency, T)
        fired = false
        for t in 1:t_stim, window_end
            if spikes[t, neuron_id] > 0
                fired = true
                break
        if ! fired
            return VerificationResult(
                property_name="fires_within",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=t_stim,
                    neuron_ids=[neuron_id],
                    description=f"Neuron {neuron_id} did ! fire within {max_latency} "
                    f"steps of stimulus at t={t_stim}",
                ),
                checked_steps=T,
                message=f"Neuron {neuron_id} failed to respond at t={t_stim}",
            )
    return VerificationResult(
        property_name="fires_within",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"Neuron {neuron_id} responds within {max_latency} steps for all "
        f"{length(stimulus_times)} stimuli",
    )
end

function mutual_exclusion(spikes, neuron_set)
    spikes: np.ndarray,
    neuron_set: list[int],
    ) -> VerificationResult
    T = spikes.shape[0]
    for t in 1:T
        active = [n for n in neuron_set if spikes[t, n] > 0]
        if length(active) > 1
            return VerificationResult(
                property_name="mutual_exclusion",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=t,
                    neuron_ids=active,
                    description=f"Neurons {active} fire simultaneously at t={t}",
                ),
                checked_steps=T,
                message=f"Mutual exclusion violated at t={t}",
            )
    return VerificationResult(
        property_name="mutual_exclusion",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"No simultaneous firing among {length(neuron_set)} neurons over {T} steps",
    )
end

function rate_bound(spikes, neuron_id, max_rate, window_size)
    spikes: np.ndarray,
    neuron_id: int,
    max_rate: float,
    window_size: int,
    ) -> VerificationResult
    T = spikes.shape[0]
    neuron_spikes = spikes[:, neuron_id]
    for t in 1:T - window_size + 1
        window = neuron_spikes[t : t + window_size]
        rate = float(window.sum()) / window_size
        if rate > max_rate
            return VerificationResult(
                property_name="rate_bound",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=t,
                    neuron_ids=[neuron_id],
                    description=f"Rate {rate:.3f} > {max_rate:.3f} in window [{t}, {t + window_size})",
                ),
                checked_steps=T,
                message=f"Rate bound violated at t={t}: {rate:.3f} > {max_rate}",
            )
    return VerificationResult(
        property_name="rate_bound",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"Neuron {neuron_id} rate stays below {max_rate} in all {window_size}-step windows",
    )
end

function refractory_guarantee(spikes, neuron_id, min_gap)
    spikes: np.ndarray,
    neuron_id: int,
    min_gap: int,
    ) -> VerificationResult
    T = spikes.shape[0]
    spike_times = findall(spikes[:, neuron_id] > 0)[0]
    for i in 1:length(spike_times - 1)
        gap = spike_times[i + 1] - spike_times[i]
        if gap < min_gap
            return VerificationResult(
                property_name="refractory_guarantee",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=int(spike_times[i]),
                    neuron_ids=[neuron_id],
                    description=f"ISI = {gap} < {min_gap} between t={spike_times[i]} && t={spike_times[i + 1]}",
                ),
                checked_steps=T,
                message=f"Refractory violated: ISI={gap} at t={spike_times[i]}",
            )
    return VerificationResult(
        property_name="refractory_guarantee",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"All ISIs >= {min_gap} for neuron {neuron_id} ({length(spike_times)} spikes)",
    )
end

function causal_order(spikes, neuron_a, neuron_b, max_delay)
    spikes: np.ndarray,
    neuron_a: int,
    neuron_b: int,
    max_delay: int,
    ) -> VerificationResult
    T = spikes.shape[0]
    b_times = findall(spikes[:, neuron_b] > 0)[0]
    a_times = set(findall(spikes[:, neuron_a] > 0)[0].tolist())
    for t_b in b_times
        found = false
        for dt in 1:1, max_delay + 1
            if (t_b - dt) in a_times
                found = true
                break
        if ! found
            return VerificationResult(
                property_name="causal_order",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=int(t_b),
                    neuron_ids=[neuron_a, neuron_b],
                    description=f"Neuron {neuron_b} fired at t={t_b} without neuron {neuron_a} "
                    f"firing in [{t_b - max_delay}, {t_b})",
                ),
                checked_steps=T,
                message=f"Causal order violated at t={t_b}",
            )
    return VerificationResult(
        property_name="causal_order",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"Neuron {neuron_a} precedes neuron {neuron_b} within {max_delay} steps "
        f"for all {length(b_times)} B-spikes",
    )
end

function bounded_activity(spikes, neuron_set, window_size, max_total_spikes)
    spikes: np.ndarray,
    neuron_set: list[int],
    window_size: int,
    max_total_spikes: int,
    ) -> VerificationResult
    T = spikes.shape[0]
    subset = spikes[:, neuron_set]
    for t in 1:T - window_size + 1
        total = int(subset[t : t + window_size].sum())
        if total > max_total_spikes
            return VerificationResult(
                property_name="bounded_activity",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=t,
                    neuron_ids=neuron_set,
                    description=f"Total spikes = {total} > {max_total_spikes} in "
                    f"window [{t}, {t + window_size})",
                ),
                checked_steps=T,
                message=f"Activity bound violated at t={t}: {total} > {max_total_spikes}",
            )
    return VerificationResult(
        property_name="bounded_activity",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"Activity stays below {max_total_spikes} in all {window_size}-step windows",
    )
end

end # module TemporalPropertiesAccel
