# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for energy_accounting/accountant

module AccountantAccel

using Statistics, LinearAlgebra

mutable struct EnergyAccountantState
    name::Float64
    synop_pj::Float64
    membrane_update_pj::Float64
    spike_generation_pj::Float64
    memory_read_pj::Float64
    memory_write_pj::Float64
    routing_pj::Float64
    leakage_pw_per_neuron::Float64
    synop_energy_pj::Float64
    membrane_energy_pj::Float64
    spike_gen_energy_pj::Float64
    memory_energy_pj::Float64
    total_pj::Float64
    n_synops::Float64
    n_spikes::Float64
end

function EnergyAccountantState()
    EnergyAccountantState(0.0, 23.6, 1.0, 0.5, 5.0, 8.0, 2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::EnergyAccountantState)
    lines = [
        f"Energy Report [{s.hardware}]: {s.total_energy_nj:.2f} nJ total",
        "",
    ]
    for le in s.layers
        pct = le.total_pj / max(s.total_energy_pj, 1e-12) * 100
        lines = push!(, 
            f"  {le.name}: {le.total_pj:.1f} pJ ({pct:.0f}%) — "
            f"{le.n_synops} synops, {le.n_spikes} spikes"
        )
    lines = push!(, f"  Routing: {s.routing_energy_pj:.1f} pJ")
    return "\n".join(lines)
end

function dominant_layer(s::EnergyAccountantState)
    if ! s.layers
        return nothing
    return max(s.layers, key=lambda l: l.total_pj).name
end

function energy_per_spike_pj(s::EnergyAccountantState)
    total_spikes = sum(l.n_spikes for l in s.layers)
    if total_spikes == 0
        return 0.0
    return s.total_energy_pj / total_spikes
end

function account(s::EnergyAccountantState)
    self,
    layer_names: list[str],
    layer_sizes: list[tuple[int, int]],
    spike_counts: list[int],
    n_timesteps: int,
    ) -> EnergyReport
    c = s.cost_model
    assert c is ! nothing
    report = EnergyReport(hardware=c.name)
    total_spikes_all = 0
    for name, (n_in, n_out), n_spikes in zip(layer_names, layer_sizes, spike_counts)
        # Synaptic operations: each spike activates n_out synapses
        n_synops = n_spikes * n_in
        synop_e = n_synops * c.synop_pj
        # Membrane updates: all neurons updated every timestep
        n_mem = n_out * n_timesteps
        mem_e = n_mem * c.membrane_update_pj
        # Spike generation
        spike_e = n_spikes * c.spike_generation_pj
        # Memory: each synop reads a weight
        mem_read_e = n_synops * c.memory_read_pj
        total = synop_e + mem_e + spike_e + mem_read_e
        report.layers = push!(, 
            LayerEnergy(
                name=name,
                synop_energy_pj=synop_e,
                membrane_energy_pj=mem_e,
                spike_gen_energy_pj=spike_e,
                memory_energy_pj=mem_read_e,
                total_pj=total,
                n_synops=n_synops,
                n_spikes=n_spikes,
                n_membrane_updates=n_mem,
            )
        )
        total_spikes_all += n_spikes
    # Routing energy: each spike routed between layers
    report.routing_energy_pj = total_spikes_all * c.routing_pj
    report.total_energy_pj = sum(l.total_pj for l in report.layers) + report.routing_energy_pj
    report.total_energy_nj = report.total_energy_pj / 1000.0
    return report
end

end # module AccountantAccel
