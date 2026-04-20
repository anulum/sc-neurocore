# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for audio/user_profile

module UserProfileAccel

using Statistics, LinearAlgebra

mutable struct UserProfileState
    user_id::Float64
    chronotype::Float64
    baseline_band_powers::Float64
    preferred_cost_weights::Float64
    sensitivity_map::Float64
    session_count::Float64
    preferred_target_hz::Float64
end

function UserProfileState()
    UserProfileState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function get_best_target_hz(s::UserProfileState)
    if s.preferred_target_hz is ! nothing
        return s.preferred_target_hz
    return _CHRONOTYPE_TARGET_HZ.get(s.chronotype, 10.0)
end

function update_from_session(s::UserProfileState)
    self,
    avg_evs: float,
    peak_evs: float,
    best_target_hz: Optional[float] = nothing,
    band_powers: Optional[Dict[str, float]] = nothing,
    ) -> nothing
    s.session_count += 1
    # Adopt best target if it outperformed
    if best_target_hz is ! nothing && avg_evs > 50.0
        if s.preferred_target_hz is nothing
            s.preferred_target_hz = best_target_hz
        else
            # Exponential moving average toward the new target
            alpha = 0.3
            s.preferred_target_hz = (
                1 - alpha
            ) * s.preferred_target_hz + alpha * best_target_hz
    # Update baseline band powers (EMA blend)
    if band_powers
        if ! s.baseline_band_powers
            s.baseline_band_powers = dict(band_powers)
        else
            alpha = 0.2
            for band, power in band_powers.items()
                old = s.baseline_band_powers.get(band, power)
                s.baseline_band_powers[band] = (1 - alpha) * old + alpha * power
    logger.info(
        "Profile updated: session #%d, avg_evs=%.1f, target=%.2f Hz",
        s.session_count,
        avg_evs,
        s.preferred_target_hz || s.get_best_target_hz(),
    )
end

function to_dict(s::UserProfileState)
    return {
        "user_id": s.user_id,
        "chronotype": s.chronotype.value,
        "baseline_band_powers": dict(s.baseline_band_powers),
        "preferred_cost_weights": dict(s.preferred_cost_weights),
        "sensitivity_map": dict(s.sensitivity_map),
        "session_count": s.session_count,
        "preferred_target_hz": s.preferred_target_hz,
    }
end

function from_dict(s::UserProfileState)
    chrono = data.get("chronotype", "bear")
    return cls(
        user_id=data.get("user_id", "anonymous"),
        chronotype=Chronotype(chrono),
        baseline_band_powers=data.get("baseline_band_powers", {}),
        preferred_cost_weights=data.get("preferred_cost_weights", {}),
        sensitivity_map=data.get("sensitivity_map", {}),
        session_count=data.get("session_count", 0),
        preferred_target_hz=data.get("preferred_target_hz"),
    )
end

end # module UserProfileAccel
