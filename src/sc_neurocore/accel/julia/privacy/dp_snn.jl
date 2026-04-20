# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for privacy/dp_snn

module DpSnnAccel

using Statistics, LinearAlgebra

mutable struct MembershipAuditState
    target_epsilon::Float64
    target_delta::Float64
    _spent_epsilon::Float64
    _steps::Float64
    epsilon::Float64
    mechanism::Float64
    _rng::Float64
    flip_prob::Float64
    keep_prob::Float64
    run_fn::Float64
end

function MembershipAuditState()
    MembershipAuditState(1.0, 1e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function record_step(s::MembershipAuditState, step_epsilon)
    s._spent_epsilon += step_epsilon
    s._steps += 1
end

function spent_epsilon(s::MembershipAuditState)
    return s._spent_epsilon
end

function remaining_epsilon(s::MembershipAuditState)
    return max(0.0, s.target_epsilon - s._spent_epsilon)
end

function budget_exhausted(s::MembershipAuditState)
    return s._spent_epsilon >= s.target_epsilon
end

function summary(s::MembershipAuditState)
    return (
        f"Privacy: epsilon={s._spent_epsilon:.4f}/{s.target_epsilon} "
        f"({s._steps} steps), delta={s.target_delta}"
    )
end

function privatize(s::MembershipAuditState, spikes)
    if s.mechanism == "randomized_response"
        flip_mask = s._rng.random(spikes.shape) < s.flip_prob
        privatized = spikes.copy().astype(np.int8)
        privatized[flip_mask] = 1 - privatized[flip_mask]
        return privatized
    else
        keep_mask = s._rng.random(spikes.shape) < s.keep_prob
        return (spikes * keep_mask).astype(spikes.dtype)
end

function per_step_epsilon(s::MembershipAuditState)
    return s.epsilon
end

function audit(s::MembershipAuditState)
    self,
    member_samples: list[np.ndarray],
    non_member_samples: list[np.ndarray],
    ) -> dict[str, Any]
    member_scores = [float(abs(s.run_fn(s)).mean()) for s in member_samples]
    non_member_scores = [float(abs(s.run_fn(s)).mean()) for s in non_member_samples]
    mean_member = float(mean(member_scores))
    mean_non = float(mean(non_member_scores))
    # Threshold-based inference: predict member if score > midpoint
    threshold = (mean_member + mean_non) / 2
    correct = 0
    total = length(member_scores) + length(non_member_scores)
    for s in member_scores
        if s >= threshold
            correct += 1
    for s in non_member_scores
        if s < threshold
            correct += 1
    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "member_confidence": mean_member,
        "non_member_confidence": mean_non,
        "vulnerable": accuracy > 0.6,
    }
end

end # module DpSnnAccel
