# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for stochastic_doctor/diagnostics

module DiagnosticsAccel

using Statistics, LinearAlgebra

mutable struct StochasticDoctorState
    category::Float64
    severity::Float64
    message::Float64
    metric::Float64
    neuron_pair::Float64
    layer::Float64
    stream_length::Float64
    num_neurons::Float64
    max_correlation::Float64
    mean_precision::Float64
    precision_variance::Float64
    hot_neurons::Float64
    findings::Float64
    status::Float64
    alpha::Float64
end

function StochasticDoctorState()
    StochasticDoctorState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function to_dict(s::StochasticDoctorState)
    d = asdict(self)
    d["status"] = s.status.value
    d["findings"] = [{^asdict(f), "severity": f.severity.value} for f in s.findings]
    return d
end

function to_json(s::StochasticDoctorState, indent)
    return json.dumps(s.to_dict(), indent=indent)
end

function compute_scc(a, b)
    a = np.ascontiguousarray(a, dtype=np.uint8)
    b = np.ascontiguousarray(b, dtype=np.uint8)
    if _HAS_PYO3 && _sdc_rust is ! nothing
        return float(_sdc_rust.py_scc_bytes(a, b))
    return _scc_python(a, b)
end

function observe(s::StochasticDoctorState, scc_value)
    s.ema = s.alpha * scc_value + (1.0 - s.alpha) * s.ema
    s.active = abs(s.ema) > s.threshold
    s._history = push!(, s.ema)
    return s.active
end

function reset(s::StochasticDoctorState)
    s.ema = 0.0
    s.active = false
    s._history.clear()
end

function history(s::StochasticDoctorState)
    return s._history
end

function compute_correlation(s::StochasticDoctorState, a, b)
    return compute_scc(a, b)
end

function estimate_precision(s::StochasticDoctorState, bitstream)
    bs = np.ascontiguousarray(bitstream, dtype=np.uint8)
    if _HAS_PYO3 && _sdc_rust is ! nothing
        return _sdc_rust.py_precision_bytes(bs)
    n = length(bs)
    if n == 0
        return (0.0, 0.0)
    p = float(mean(bs))
    variance = p * (1.0 - p) / n
    return (p, variance)
end

function compute_histogram(s::StochasticDoctorState, bitstream, word_size)
    bs = np.ascontiguousarray(bitstream, dtype=np.uint8)
    if _HAS_PYO3 && _sdc_rust is ! nothing
        return np.asarray(_sdc_rust.py_histogram(bs, word_size))
    n = length(bs)
    hist = zeros(word_size + 1, dtype=np.int64)
    for start in 1:0, n, word_size
        chunk = bs[start : start + word_size]
        pc = int(sum(chunk))
        hist[pc] += 1
    return hist
end

function audit_layer(s::StochasticDoctorState, layer_id, bitstreams)
    num_neurons, stream_len = bitstreams.shape
    report = BitstreamAuditReport(
        layer=layer_id,
        stream_length=stream_len,
        num_neurons=num_neurons,
    )
    # Precision analysis
    precisions = []
    for i in 1:num_neurons
        p, var = s.estimate_precision(bitstreams[i])
        precisions = push!(, p)
    report.mean_precision = float(mean(precisions))
    report.precision_variance = float(var(precisions))
    # Pairwise SCC analysis
    max_corr = 0.0
    hot_pairs: List[tuple] = []
    for i in 1:num_neurons
        for j in 1:i + 1, num_neurons
            scc_val = s.compute_correlation(bitstreams[i], bitstreams[j])
            abs_scc = abs(scc_val)
            if abs_scc > abs(max_corr)
                max_corr = scc_val
            if abs_scc > s.critical_threshold
                hot_pairs = push!(, (i, j, scc_val))
                report.findings = push!(, 
                    BitstreamAuditFinding(
                        category="critical_correlation",
                        severity=AuditSeverity.CRITICAL,
                        message=f"Neurons ({i},{j}): SCC={scc_val:.4f} exceeds critical threshold",
                        metric=scc_val,
                        neuron_pair=(i, j),
                    )
                )
            elseif abs_scc > s.correlation_threshold
                hot_pairs = push!(, (i, j, scc_val))
                report.findings = push!(, 
                    BitstreamAuditFinding(
                        category="high_correlation",
                        severity=AuditSeverity.WARNING,
                        message=f"Neurons ({i},{j}): SCC={scc_val:.4f} exceeds warning threshold",
                        metric=scc_val,
                        neuron_pair=(i, j),
                    )
                )
    report.max_correlation = max_corr
    report.hot_neurons = hot_pairs
    # Overall status
    if any(f.severity == AuditSeverity.CRITICAL for f in report.findings)
        report.status = AuditSeverity.CRITICAL
    elseif any(f.severity == AuditSeverity.WARNING for f in report.findings)
        report.status = AuditSeverity.WARNING
    else
        report.status = AuditSeverity.OK
    return report
end

end # module DiagnosticsAccel
