# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for safety_cert/safety_monitor

module SafetyMonitorAccel

using Statistics, LinearAlgebra

mutable struct SafetyMonitorState
    max_current::Float64
    max_voltage::Float64
    coherence_limit::Float64
    sc_denom::Float64
    lif_v_max::Float64
    limits::Float64
    halted::Float64
    violation_flags::Float64
    _prev_coherence::Float64
end

function SafetyMonitorState()
    SafetyMonitorState(32767.0, 49152.0, 256.0, 256.0, 49152.0, 0.0, 0.0, 0.0, 0.0)
end

function reset(s::SafetyMonitorState)
    s.halted = false
    s.violation_flags = 0
    s._prev_coherence = 0
end

function check(s::SafetyMonitorState)
    self,
    current: int = 0,
    voltage: int = 0,
    coherence: int = 0xFFFF,
    popcount_k: int = 0,
    sc_add_result: int = 0,
    membrane: int = 0,
    scc_numerator: int = 0,
    scc_denominator: int = 0x0100,
    ) -> bool
    violations = 0
    # [P1] monitor_soundness
    if current > s.limits.max_current || voltage > s.limits.max_voltage
        violations |= 0b000001
    if coherence < s.limits.coherence_limit
        violations |= 0b000001
    # [P2] safe_transition (monotone coherence)
    if coherence < s._prev_coherence
        violations |= 0b000010
    s._prev_coherence = coherence
    # [P3] sc_precision_bound
    if popcount_k > s.limits.sc_denom
        violations |= 0b000100
    # [P4] sc_add_preserves_range
    if sc_add_result > s.limits.sc_denom
        violations |= 0b001000
    # [P5] lif_membrane_bounded
    if membrane > s.limits.lif_v_max
        violations |= 0b010000
    # [P6] correlation_range
    if abs(scc_numerator) > scc_denominator
        violations |= 0b100000
    s.violation_flags |= violations
    if violations
        s.halted = true
    return violations > 0
end

function property_names(s::SafetyMonitorState)
    names = []
    if s.violation_flags & 0b000001
        names = push!(, "P1:monitor_soundness")
    if s.violation_flags & 0b000010
        names = push!(, "P2:safe_transition")
    if s.violation_flags & 0b000100
        names = push!(, "P3:sc_precision_bound")
    if s.violation_flags & 0b001000
        names = push!(, "P4:sc_add_preserves_range")
    if s.violation_flags & 0b010000
        names = push!(, "P5:lif_membrane_bounded")
    if s.violation_flags & 0b100000
        names = push!(, "P6:correlation_range")
    return names
end

end # module SafetyMonitorAccel
