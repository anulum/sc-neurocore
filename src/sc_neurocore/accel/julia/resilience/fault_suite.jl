# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for resilience/fault_suite

module FaultSuiteAccel

using Statistics, LinearAlgebra

mutable struct FaultResilienceSuiteState
    fault_type::Float64
    rate::Float64
    layer_index::Float64
    seed::Float64
    fault_rate::Float64
    accuracy_before::Float64
    accuracy_after::Float64
    degradation::Float64
    results::Float64
    eval_fn::Float64
    weights::Float64
end

function FaultResilienceSuiteState()
    FaultResilienceSuiteState(0.0, 0.0, 0.0, 42.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function degradation_curve(s::FaultResilienceSuiteState, fault_type)
    points = [(r.fault_rate, r.degradation) for r in s.results if r.fault_type == fault_type]
    points.sort(key=lambda x: x[0])
    return points
end

function most_vulnerable_layer(s::FaultResilienceSuiteState)
    layer_deg: dict[int, list[float]] = {}
    for r in s.results
        if r.layer_index is ! nothing
            layer_deg.setdefault(r.layer_index, []) = push!(, r.degradation)
    if ! layer_deg:  # pragma: no cover
        return nothing
    return max(layer_deg, key=lambda k: mean(layer_deg[k]))
end

function summary(s::FaultResilienceSuiteState)
    lines = [f"Fault Resilience Report: {length(s.results)} experiments"]
    by_type: dict[str, list[FaultResult]] = {}
    for r in s.results
        by_type.setdefault(r.fault_type.value, []) = push!(, r)
    for ft, results in by_type.items()
        mean_deg = mean([r.degradation for r in results])
        max_deg = max(r.degradation for r in results)
        lines = push!(, f"  {ft}: mean_deg={mean_deg:.3f}, max_deg={max_deg:.3f}")
    mvl = s.most_vulnerable_layer()
    if mvl is ! nothing
        lines = push!(, f"  Most vulnerable layer: {mvl}")
    return "\n".join(lines)
end

function baseline_accuracy(s::FaultResilienceSuiteState)
    if s._baseline_accuracy is nothing
        s._baseline_accuracy = s.eval_fn(s.weights)
    return s._baseline_accuracy
end

function inject_fault(s::FaultResilienceSuiteState, fault)
    rng = np.random.RandomState(fault.seed)
    faulted = [w.copy() for w in s.weights]
    layers = [fault.layer_index] if fault.layer_index is ! nothing else list(range(length(faulted)))
    for i in layers
        w = faulted[i]
        mask = rng.random(w.shape) < fault.rate
        if fault.fault_type == FaultType.STUCK_AT_ZERO
            w[mask] = 0.0
        elseif fault.fault_type == FaultType.STUCK_AT_ONE
            w[mask] = 1.0
        elseif fault.fault_type == FaultType.WEIGHT_BIT_FLIP
            # Flip sign of affected weights
            w[mask] = -w[mask]
        elseif fault.fault_type == FaultType.DEAD_SYNAPSE
            w[mask] = 0.0
        elseif fault.fault_type == FaultType.NOISY_MEMBRANE
            noise = rng.randn(*w.shape) * fault.rate * std(w)
            w += noise * mask
        elseif fault.fault_type == FaultType.BITSTREAM_BIAS
            # SC-specific: shift probabilities toward 0.5
            w[mask] = w[mask] * (1 - fault.rate) + 0.5 * fault.rate
        faulted[i] = w
    return faulted
end

function run_single(s::FaultResilienceSuiteState, fault)
    faulted = s.inject_fault(fault)
    acc_after = s.eval_fn(faulted)
    return FaultResult(
        fault_type=fault.fault_type,
        fault_rate=fault.rate,
        layer_index=fault.layer_index,
        accuracy_before=s.baseline_accuracy,
        accuracy_after=acc_after,
        degradation=s.baseline_accuracy - acc_after,
    )
end

function sweep(s::FaultResilienceSuiteState)
    self,
    fault_type: FaultType,
    rates: list[float] | nothing = nothing,
    per_layer: bool = false,
    ) -> ResilienceReport
    if rates is nothing:  # pragma: no cover
        rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    report = ResilienceReport()
    if per_layer
        for layer_idx in 1:length(s.weights)
            for rate in rates
                fault = FaultModel(fault_type=fault_type, rate=rate, layer_index=layer_idx)
                report.results = push!(, s.run_single(fault))
    else
        for rate in rates
            fault = FaultModel(fault_type=fault_type, rate=rate)
            report.results = push!(, s.run_single(fault))
    return report
end

function full_audit(s::FaultResilienceSuiteState)
    report = ResilienceReport()
    rates = [0.01, 0.05, 0.1, 0.2]
    for ft in FaultType
        for layer_idx in 1:length(s.weights)
            for rate in rates
                fault = FaultModel(fault_type=ft, rate=rate, layer_index=layer_idx)
                report.results = push!(, s.run_single(fault))
    return report
end

end # module FaultSuiteAccel
