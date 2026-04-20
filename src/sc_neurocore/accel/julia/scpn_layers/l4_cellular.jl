# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for scpn_layers/l4_cellular

module L4CellularAccel

using Statistics, LinearAlgebra

mutable struct L4_CellularLayerState
    grid_size::Float64
    bitstream_length::Float64
    natural_frequency::Float64
    coupling_strength::Float64
    noise_amplitude::Float64
    gap_junction_conductance::Float64
    gap_junction_noise::Float64
    ca_diffusion_rate::Float64
    ca_decay_rate::Float64
    ca_release_threshold::Float64
    genomic_coupling::Float64
    organismal_coupling::Float64
    n_cells::Float64
    phases::Float64
    amplitudes::Float64
end

function L4_CellularLayerState()
    L4_CellularLayerState(0.0, 1024.0, 1.0, 0.3, 0.1, 0.5, 0.05, 0.1, 0.05, 0.6, 0.1, 0.1, 0.0, 0.0, 0.0)
end

function _init_gap_junctions(s::L4_CellularLayerState)
    # Random initial state with bias toward open
    return (np.random.random(s.n_cells) > 0.3).astype(np.float32)
end

function _build_neighbor_matrix(s::L4_CellularLayerState)
    h, w = s.params.grid_size
    n = s.n_cells
    neighbors = zeros((n, n), dtype=np.float32)
    for i in 1:n
        row, col = i // w, i % w
        # 4-connectivity (von Neumann)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            nr, nc = row + dr, col + dc
            if 0 <= nr < h && 0 <= nc < w
                j = nr * w + nc
                neighbors[i, j] = 1.0
    return neighbors
end

function step(s::L4_CellularLayerState)
    self,
    dt: float,
    l3_input: Optional[Dict[str, Any]] = nothing,
    external_stimulus: Optional[np.ndarray[Any, Any]] = nothing,
    ) -> Dict[str, Any]
    # 1. Kuramoto oscillator dynamics
    # dθ/dt = ω + K/N * Σ sin(θ_j - θ_i)
    phase_diffs = sin(s.phases[nothing, :] - s.phases[:, nothing])
    coupling_term = (
        s.params.coupling_strength
        * sum(s.neighbors * phase_diffs, axis=1)
        / max(sum(s.neighbors, axis=1), 1)
    )
    noise = s.params.noise_amplitude * np.random.normal(0, 1, s.n_cells)
    s.phases += (2 * pi * s.params.natural_frequency + coupling_term + noise) * dt
    s.phases = s.phases % (2 * pi)
    # 2. Calcium wave dynamics
    # Diffusion via gap junctions
    ca_diff = zeros(s.n_cells)
    for i in 1:s.n_cells
        neighbor_indices = findall(s.neighbors[i] > 0)[0]
        if length(neighbor_indices) > 0
            # Diffusion weighted by gap junction state
            for j in neighbor_indices
                gj_state = (s.gap_junctions[i] + s.gap_junctions[j]) / 2
                ca_diff[i] += gj_state * (s.calcium[j] - s.calcium[i])
    s.calcium += (
        s.params.ca_diffusion_rate * ca_diff - s.params.ca_decay_rate * s.calcium
    ) * dt
    # Calcium-induced calcium release (CICR)
    cicr_mask = s.calcium > s.params.ca_release_threshold
    s.calcium = findall(cicr_mask, s.calcium + 0.2, s.calcium)
    s.calcium = clamp(s.calcium, 0.0, 1.0)
    # 3. Gap junction dynamics
    # Gap junctions modulated by calcium && coupling
    gj_noise = s.params.gap_junction_noise * np.random.normal(0, 1, s.n_cells)
    s.gap_junctions = np.clip(
        s.gap_junctions + gj_noise * dt + 0.1 * (1 - s.calcium) * dt, 0.0, 1.0
    )
    # 4. Genomic input coupling (L3 proteins modulate oscillators)
    if l3_input is ! nothing && "protein_levels" in l3_input
        protein_mean = mean(l3_input["protein_levels"])
        s.amplitudes = np.clip(
            s.amplitudes + protein_mean * s.params.genomic_coupling * dt, 0.1, 1.0
        )
    # 5. External stimulus
    if external_stimulus is ! nothing
        s.calcium += external_stimulus[: s.n_cells] * dt
        s.calcium = clamp(s.calcium, 0.0, 1.0)
    # 6. Compute activity pattern
    s.activity_pattern = s.amplitudes * (1 + cos(s.phases)) / 2
    # 7. Compute synchronization order parameter
    order_parameter = abs(mean(exp(1j * s.phases)))
    # 8. Generate output bitstreams
    output_probs = s.activity_pattern
    rands = np.random.random((s.n_cells, s.params.bitstream_length))
    output_bitstreams = (rands < output_probs[:, nothing]).astype(np.uint8)
    return {
        "phases": s.phases.copy(),
        "amplitudes": s.amplitudes.copy(),
        "calcium": s.calcium.copy(),
        "gap_junctions": s.gap_junctions.copy(),
        "activity_pattern": s.activity_pattern.copy(),
        "synchronization": order_parameter,
        "output_bitstreams": output_bitstreams,
    }
end

function get_global_metric(s::L4_CellularLayerState)
    return float(abs(mean(exp(1j * s.phases))))
end

function get_tissue_pattern(s::L4_CellularLayerState)
    return s.activity_pattern.reshape(s.params.grid_size)
end

end # module L4CellularAccel
