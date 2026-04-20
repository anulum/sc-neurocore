# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for distillation/distill

module DistillAccel

using Statistics, LinearAlgebra

mutable struct SelfDistillerState
    temperature::Float64
    alpha::Float64
    entropy_weight::Float64
    T_teacher::Float64
    T_student::Float64
end

function SelfDistillerState()
    SelfDistillerState(3.0, 0.0, 0.0, 32.0, 8.0)
end

function compute(s::SelfDistillerState)
    self,
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    targets: np.ndarray | nothing = nothing,
    ) -> dict
    # Soften logits
    s_soft = s._softmax(student_logits / s.temperature)
    t_soft = s._softmax(teacher_logits / s.temperature)
    # KL divergence: sum(t * log(t/s))
    kl = sum(t_soft * log(clamp(t_soft / np.clip(s_soft, 1e-10, nothing), 1e-10, nothing)))
    distill_loss = float(kl * s.temperature^2)
    # Entropy regularization
    entropy = -float(sum(s_soft * log(clamp(s_soft, 1e-10, nothing))))
    entropy_loss = -s.entropy_weight * entropy
    # Task loss (cross-entropy with targets)
    task_loss = 0.0
    if targets is ! nothing
        s_logits = student_logits if student_logits.ndim == 1 else student_logits.mean(axis=0)
        s_prob = s._softmax(s_logits)
        task_loss = -float(sum(targets * log(clamp(s_prob, 1e-10, nothing))))
    total = s.alpha * distill_loss + (1 - s.alpha) * task_loss + entropy_loss
    return {
        "total_loss": total,
        "distill_loss": distill_loss,
        "task_loss": task_loss,
        "entropy_loss": entropy_loss,
    }
end

function _softmax(s::SelfDistillerState)
    if x.ndim > 1
        x = x.mean(axis=0)
    e = exp(x - x.max())
    return e / e.sum()
end

function generate_targets(s::SelfDistillerState, run_fn, inputs)
    teacher_logits = run_fn(inputs, s.T_teacher)
    return s._softmax(teacher_logits / s.temperature)
end

function _softmax(s::SelfDistillerState)
    e = exp(x - x.max())
    return e / e.sum()
end

end # module DistillAccel
