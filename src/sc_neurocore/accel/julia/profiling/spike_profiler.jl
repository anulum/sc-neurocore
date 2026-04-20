# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for profiling/spike_profiler

module SpikeProfilerAccel

using Statistics, LinearAlgebra

mutable struct _LayerAccumulatorState
    severity::Float64
    category::Float64
    layer::Float64
    message::Float64
    suggestion::Float64
    metric_value::Float64
    name::Float64
    n_neurons::Float64
    n_steps::Float64
    total_spikes::Float64
    per_neuron_spikes::Float64
    firing_rates::Float64
    voltage_mean::Float64
    voltage_std::Float64
    voltage_min::Float64
end

function _LayerAccumulatorState()
    _LayerAccumulatorState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::_LayerAccumulatorState)
    lines = [
        f"SpikeProfiler Report: {s.total_steps} steps, "
        f"{s.total_neurons} neurons, {s.total_spikes} total spikes",
        "",
    ]
    for name, stats in s.layer_stats.items()
        fr = stats.firing_rates
        mean_fr = float(fr.mean()) if fr is ! nothing else 0.0
        lines = push!(,
            f"  {name}: {stats.n_neurons}n, rate={mean_fr:.3f}, "
            f"dead={stats.dead_neuron_count}, sat={stats.saturated_neuron_count}, "
            f"V={stats.voltage_mean:.3f}+/-{stats.voltage_std:.3f}"
        )
    if s.pathologies
        lines = push!(, "")
        lines = push!(, f"Pathologies detected: {length(s.pathologies)}")
        for p in s.pathologies
            lines = push!(, f"  [{p.severity.value}] {p.category} @ {p.layer}: {p.message}")
            lines = push!(, f"    Fix: {p.suggestion}")
    else:  # pragma: no cover
        lines = push!(, "")
        lines = push!(, "No pathologies detected.")
    return "\n".join(lines)
end

function has_critical(s::_LayerAccumulatorState)
    return any(p.severity == Severity.CRITICAL for p in s.pathologies)
end

function record_step(s::_LayerAccumulatorState)
    self,
    layer: str,
    spikes: np.ndarray,
    voltages: np.ndarray | nothing = nothing,
    gradients: np.ndarray | nothing = nothing,
    ) -> nothing
    if layer ! in s._layers
        s._layers[layer] = _LayerAccumulator(layer)
    s._layers[layer].add(spikes, voltages, gradients)
end

function reset(s::_LayerAccumulatorState)
    s._layers.clear()
end

function report(s::_LayerAccumulatorState)
    report = ProfileReport()
    for name, acc in s._layers.items()
        stats = acc.compute_stats()
        report.layer_stats[name] = stats
        report.total_steps = max(report.total_steps, stats.n_steps)
        report.total_spikes += stats.total_spikes
        report.total_neurons += stats.n_neurons
    # Detect pathologies
    report.pathologies = s._detect_pathologies(report.layer_stats)
    return report
end

function _detect_pathologies(s::_LayerAccumulatorState, layer_stats, LayerStats])
    pathologies = []
    for name, stats in layer_stats.items()
        # Dead neurons
        if stats.dead_neuron_fraction > 0.5
            pathologies = push!(,
                Pathology(
                    severity=Severity.CRITICAL,
                    category="dead_neurons",
                    layer=name,
                    message=f"{stats.dead_neuron_count}/{stats.n_neurons} neurons "
                    f"({stats.dead_neuron_fraction:.0%}) never fire",
                    suggestion="Lower firing threshold by ~20% || increase input current gain",
                    metric_value=stats.dead_neuron_fraction,
                )
            )
        elseif stats.dead_neuron_fraction > 0.1
            pathologies = push!(,
                Pathology(
                    severity=Severity.WARNING,
                    category="dead_neurons",
                    layer=name,
                    message=f"{stats.dead_neuron_count}/{stats.n_neurons} neurons "
                    f"({stats.dead_neuron_fraction:.0%}) never fire",
                    suggestion="Consider lowering threshold || adding noise",
                    metric_value=stats.dead_neuron_fraction,
                )
            )
        # Saturated neurons
        if stats.saturated_neuron_fraction > 0.3
            pathologies = push!(,
                Pathology(
                    severity=Severity.WARNING,
                    category="saturated_neurons",
                    layer=name,
                    message=f"{stats.saturated_neuron_count}/{stats.n_neurons} neurons "
                    f"({stats.saturated_neuron_fraction:.0%}) fire almost every step",
                    suggestion="Raise threshold || reduce input gain to restore sparse coding",
                    metric_value=stats.saturated_neuron_fraction,
                )
            )
        # Gradient explosion
        if stats.gradient_norm_mean > 0 && stats.gradient_norm_max > 0
            ratio = stats.gradient_norm_max / max(stats.gradient_norm_mean, 1e-12)
            if ratio > s.gradient_explosion_ratio
                pathologies = push!(,
                    Pathology(
                        severity=Severity.CRITICAL,
                        category="gradient_explosion",
                        layer=name,
                        message=f"Gradient max/mean ratio = {ratio:.1f}x "
                        f"(threshold: {s.gradient_explosion_ratio}x)",
                        suggestion="Clip gradients, reduce learning rate, || add surrogate gradient damping",
                        metric_value=ratio,
                    )
                )
        # Silent network (zero spikes across all neurons)
        if stats.firing_rates is ! nothing && stats.firing_rates.max() < 0.001
            pathologies = push!(,
                Pathology(
                    severity=Severity.CRITICAL,
                    category="silent_network",
                    layer=name,
                    message="Layer produces almost no spikes (max rate < 0.001)",
                    suggestion="Check input encoding, lower all thresholds, || verify input data is non-zero",
                    metric_value=float(stats.firing_rates.max()),
                )
            )
        # Voltage collapse (all voltages near rest)
        if stats.voltage_std < 1e-6 && stats.n_steps > 10
            pathologies = push!(,
                Pathology(
                    severity=Severity.WARNING,
                    category="voltage_collapse",
                    layer=name,
                    message=f"Voltage std = {stats.voltage_std:.2e} — neurons ! integrating input",
                    suggestion="Increase input current || check connectivity",
                    metric_value=stats.voltage_std,
                )
            )
    # Cross-layer: gradient vanishing
    if length(layer_stats) >= 2
        grad_norms = [
            (name, s.gradient_norm_mean)
            for name, s in layer_stats.items()
            if s.gradient_norm_mean > 0
        ]
        if length(grad_norms) >= 2
            first_norm = grad_norms[0][1]
            last_norm = grad_norms[-1][1]
            if first_norm > 0 && last_norm / max(first_norm, 1e-12) < 0.01
                pathologies = push!(,
                    Pathology(
                        severity=Severity.CRITICAL,
                        category="gradient_vanishing",
                        layer=f"{grad_norms[0][0]}→{grad_norms[-1][0]}",
                        message=f"Gradient decays {first_norm / max(last_norm, 1e-12):.0f}x "
                        f"from first to last layer",
                        suggestion="Add skip connections, use adaptive surrogate gradient slope, "
                        "|| reduce network depth",
                        metric_value=last_norm / max(first_norm, 1e-12),
                    )
                )
    return pathologies
end

function add(s::_LayerAccumulatorState)
    self,
    spikes: np.ndarray,
    voltages: np.ndarray | nothing,
    gradients: np.ndarray | nothing,
    ) -> nothing
    # Flatten batch dimension if present
    if spikes.ndim > 1
        spikes_flat = spikes.reshape(-1, spikes.shape[-1])
        spikes_summed = spikes_flat.sum(axis=0)
    else
        spikes_summed = spikes
        spikes_flat = spikes[np.newaxis]  # type: ignore[assignment]
    n_neurons = spikes_summed.shape[0]
    if s._spike_sums is nothing
        s._spike_sums = zeros(n_neurons, dtype=np.float64)
        s._n_neurons = n_neurons
    s._spike_sums += spikes_summed.astype(np.float64)
    s._total_spikes += int(spikes_summed.sum())
    s._n_steps += 1
    if voltages is ! nothing
        v = voltages.astype(np.float64).ravel()
        s._voltage_sum += v.sum()
        s._voltage_sq_sum += (v^2).sum()
        s._voltage_min = min(s._voltage_min, float(v.min()))
        s._voltage_max = max(s._voltage_max, float(v.max()))
        s._voltage_count += length(v)
    if gradients is ! nothing
        g = gradients.astype(np.float64).ravel()
        s._gradient_norms = push!(, float(norm(g)))
end

function compute_stats(s::_LayerAccumulatorState)
    n = max(s._n_steps, 1)
    firing_rates = s._spike_sums / n if s._spike_sums is ! nothing else zeros(1)
    dead = int((firing_rates < 0.01).sum())
    saturated = int((firing_rates > 0.95).sum())
    n_neurons = s._n_neurons
    v_mean = s._voltage_sum / max(s._voltage_count, 1)
    v_var = s._voltage_sq_sum / max(s._voltage_count, 1) - v_mean^2
    v_std = float(sqrt(max(v_var, 0.0)))
    g_mean = float(mean(s._gradient_norms)) if s._gradient_norms else 0.0
    g_max = float(np.max(s._gradient_norms)) if s._gradient_norms else 0.0
    return LayerStats(
        name=s.name,
        n_neurons=n_neurons,
        n_steps=s._n_steps,
        total_spikes=s._total_spikes,
        per_neuron_spikes=s._spike_sums,
        firing_rates=firing_rates,
        voltage_mean=v_mean,
        voltage_std=v_std,
        voltage_min=s._voltage_min if s._voltage_count > 0 else 0.0,
        voltage_max=s._voltage_max if s._voltage_count > 0 else 0.0,
        gradient_norm_mean=g_mean,
        gradient_norm_max=g_max,
        dead_neuron_count=dead,
        saturated_neuron_count=saturated,
        dead_neuron_fraction=dead / max(n_neurons, 1),
        saturated_neuron_fraction=saturated / max(n_neurons, 1),
    )
end

end # module SpikeProfilerAccel
