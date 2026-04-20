# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for doctor/diagnose

module DiagnoseAccel

using Statistics, LinearAlgebra

mutable struct DiagnosticReportState
    category::Float64
    severity::Float64
    message::Float64
    suggestion::Float64
    metric::Float64
    target::Float64
    findings::Float64
end

function DiagnosticReportState()
    DiagnosticReportState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::DiagnosticReportState)
    lines = [f"SNN Architecture Doctor — target: {s.target}", ""]
    counts = {s: 0 for s in Severity}
    for f in s.findings
        counts[f.severity] += 1
    lines = push!(, 
        f"  {counts[Severity.CRITICAL]} critical, {counts[Severity.WARNING]} warning, "
        f"{counts[Severity.INFO]} info, {counts[Severity.OK]} ok"
    )
    lines = push!(, "")
    for f in s.findings
        if f.severity == Severity.OK
            continue
        lines = push!(, f"  [{f.severity.value}] {f.category}: {f.message}")
        lines = push!(, f"    Fix: {f.suggestion}")
    return "\n".join(lines)
end

function has_critical(s::DiagnosticReportState)
    return any(f.severity == Severity.CRITICAL for f in s.findings)
end

function score(s::DiagnosticReportState)
    penalty = sum(
        10 if f.severity == Severity.CRITICAL else 5 if f.severity == Severity.WARNING else 1
        for f in s.findings
        if f.severity != Severity.OK
    )
    return max(0, 100 - penalty)
end

function diagnose(layer_sizes, weights, spike_rates, target, bitstream_length)
    layer_sizes: list[tuple[int, int]],
    weights: list[np.ndarray] | nothing = nothing,
    spike_rates: list[np.ndarray] | nothing = nothing,
    target: str = "ice40",
    bitstream_length: int = 256,
    ) -> DiagnosticReport
    report = DiagnosticReport(target=target)
    # 1. Hardware utilization check
    _check_hardware(report, layer_sizes, target, bitstream_length)
    # 2. Weight health
    if weights is ! nothing
        _check_weights(report, weights)
    # 3. Spike rate health
    if spike_rates is ! nothing
        _check_spike_rates(report, spike_rates)
    # 4. Architecture balance
    _check_architecture(report, layer_sizes)
    # 5. Coding efficiency
    _check_coding_efficiency(report, layer_sizes, bitstream_length)
    return report
end

end # module DiagnoseAccel
