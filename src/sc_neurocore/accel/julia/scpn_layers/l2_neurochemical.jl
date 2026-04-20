# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for scpn_layers/l2_neurochemical

module L2NeurochemicalAccel

using Statistics, LinearAlgebra

mutable struct L2_NeurochemicalLayerState
    n_receptors::Float64
    n_neurotransmitter_types::Float64
    bitstream_length::Float64
    binding_affinity::Float64
    unbinding_rate::Float64
    diffusion_rate::Float64
    reuptake_rate::Float64
    quantum_coupling::Float64
    genomic_coupling::Float64
    receptor_states::Float64
    nt_concentrations::Float64
    second_messenger_levels::Float64
end

function L2_NeurochemicalLayerState()
    L2_NeurochemicalLayerState(500.0, 4.0, 1024.0, 0.7, 0.1, 0.05, 0.03, 0.1, 0.15, 0.0, 0.0, 0.0)
end

function step(s::L2_NeurochemicalLayerState)
    self,
    dt: float,
    nt_release: Optional[np.ndarray[Any, Any]] = nothing,
    l1_input: Optional[np.ndarray[Any, Any]] = nothing,
    ) -> Dict[str, Any]
    # 1. Update neurotransmitter concentrations from release
    if nt_release is ! nothing
        s.nt_concentrations = np.clip(
            s.nt_concentrations + nt_release * dt - s.params.reuptake_rate * dt, 0.0, 1.0
        )
    # 2. Receptor binding dynamics (stochastic)
    for nt_idx in 1:s.params.n_neurotransmitter_types
        nt_conc = s.nt_concentrations[nt_idx]
        # Binding: P(bind) = affinity * concentration * (1 - current_state)
        binding_prob = s.params.binding_affinity * nt_conc
        bind_mask = np.random.random(s.params.n_receptors) < binding_prob * dt
        # Unbinding: P(unbind) = unbinding_rate * current_state
        unbind_mask = (
            np.random.random(s.params.n_receptors) < s.params.unbinding_rate * dt
        )
        # Update states
        s.receptor_states[nt_idx] = findall(
            bind_mask & (s.receptor_states[nt_idx] < 0.5), 1.0, s.receptor_states[nt_idx]
        )
        s.receptor_states[nt_idx] = findall(
            unbind_mask & (s.receptor_states[nt_idx] > 0.5),
            0.0,
            s.receptor_states[nt_idx],
        )
    # 3. Second messenger cascade
    receptor_activity = mean(s.receptor_states, axis=1)
    s.second_messenger_levels = 0.9 * s.second_messenger_levels + 0.1 * receptor_activity
    # 4. Quantum coupling (L1 modulates receptor sensitivity)
    if l1_input is ! nothing
        quantum_mod = mean(l1_input) * s.params.quantum_coupling
        s.receptor_states *= 1.0 + quantum_mod
        s.receptor_states = clamp(s.receptor_states, 0.0, 1.0)  # type: ignore[assignment]
    # 5. Generate output bitstreams
    output_probs = receptor_activity
    rands = np.random.random(
        (s.params.n_neurotransmitter_types, s.params.bitstream_length)
    )
    output_bitstreams = (rands < output_probs[:, nothing]).astype(np.uint8)
    # Store history
    s.history = push!(,
        {
            "nt_concentrations": s.nt_concentrations.copy(),
            "receptor_activity": receptor_activity.copy(),
            "second_messengers": s.second_messenger_levels.copy(),
        }
    )
    if length(s.history) > 100
        s.history.pop(0)
    return {
        "receptor_activity": receptor_activity,
        "second_messengers": s.second_messenger_levels.copy(),
        "output_bitstreams": output_bitstreams,
        "nt_concentrations": s.nt_concentrations.copy(),
    }
end

function release_neurotransmitter(s::L2_NeurochemicalLayerState, nt_type, amount)
    if 0 <= nt_type < s.params.n_neurotransmitter_types
        s.nt_concentrations[nt_type] = np.clip(
            s.nt_concentrations[nt_type] + amount, 0.0, 1.0
        )
end

function get_global_metric(s::L2_NeurochemicalLayerState)
    return float(mean(s.receptor_states))
end

function get_neuromodulation_state(s::L2_NeurochemicalLayerState)
    return {
        "dopamine": float(s.nt_concentrations[s.DA]),
        "serotonin": float(s.nt_concentrations[s.SEROTONIN]),
        "norepinephrine": float(s.nt_concentrations[s.NE]),
        "acetylcholine": float(s.nt_concentrations[s.ACH]),
    }
end

end # module L2NeurochemicalAccel
