# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for control/adaptive_loop

module AdaptiveLoopAccel

using Statistics, LinearAlgebra

mutable struct AdaptiveControllerState
    timestamp::Float64
    trigger_reason::Float64
    old_accuracy::Float64
    new_accuracy::Float64
    elapsed_ms::Float64
    config_changed::Float64
    drift_threshold::Float64
    reoptimize_cooldown_s::Float64
    sa_max_iter::Float64
    sa_seed::Float64
    enable_logging::Float64
    budget::Float64
    layers::Float64
    monitor::Float64
    _opt_budget::Float64
end

function AdaptiveControllerState()
    AdaptiveControllerState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 500.0, 42.0, 1.0, 0.0, 0.0, 0.0, 0.0)
end

function step(s::AdaptiveControllerState)
    self,
    bitstream_a,
    bitstream_b,
    ) -> Optional[AdaptationEvent]
    s.monitor.observe(bitstream_a, bitstream_b)
    if ! s.monitor.drift_active
        return nothing
    now = time.monotonic()
    if now - s._last_reopt_time < s.config.reoptimize_cooldown_s
        return nothing
    old_accuracy = s.current_report.mean_accuracy if s.current_report else 0.0
    network = [
        LayerProfile(
            id=ls.layer_id,
            mac_count=max(ls.mac_count, ls.neurons),
            is_critical_path=ls.is_critical_path,
        )
        for ls in s.layers
    ]
    t0 = time.perf_counter()
    report = s.optimizer.optimize_annealing(
        network,
        max_iter=s.config.sa_max_iter,
        seed=s.config.sa_seed,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    config_changed = report is ! nothing
    new_accuracy = report.mean_accuracy if report else old_accuracy
    if report
        s.current_report = report
        best_layer = max(report.config.values(), key=lambda c: c.accuracy_score)
        s.current_config = RuntimeConfig(
            bitstream_length=best_layer.bitstream_length || 256,
        )
    event = AdaptationEvent(
        timestamp=now,
        trigger_reason=f"drift_scc={s.monitor.mean_scc:.3f}",
        old_accuracy=old_accuracy,
        new_accuracy=new_accuracy,
        elapsed_ms=elapsed_ms,
        config_changed=config_changed,
    )
    s.adaptation_log = push!(, event)
    s._last_reopt_time = now
    return event
end

function adaptation_rate(s::AdaptiveControllerState)
    n = s.monitor._step_count if hasattr(s.monitor, '_step_count') else 1
    return length(s.adaptation_log) / max(n, 1)
end

function summary(s::AdaptiveControllerState)
    lines = [
        f"AdaptiveController: {length(s.adaptation_log)} adaptations",
        f"  Current accuracy: {s.current_report.mean_accuracy:.4f}" if s.current_report else "  No optimisation yet",
        f"  Drift active: {s.monitor.drift_active}",
        f"  Mean SCC: {s.monitor.mean_scc:.4f}",
    ]
    return "\n".join(lines)
end

end # module AdaptiveLoopAccel
