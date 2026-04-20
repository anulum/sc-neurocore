# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for bridges/photonic_noc

module PhotonicNocAccel

using Statistics, LinearAlgebra

mutable struct CrosstalkAnalyzerState
    source::Float64
    target::Float64
    length_um::Float64
    wavelength_nm::Float64
    loss_db::Float64
    n_crossings::Float64
    wg_type::Float64
    gate_id::Float64
    operation::Float64
    input_ports::Float64
    output_port::Float64
    phase_shift_rad::Float64
    arm_length_um::Float64
    insertion_loss_db::Float64
    extinction_ratio_db::Float64
end

function CrosstalkAnalyzerState()
    CrosstalkAnalyzerState(0.0, 0.0, 100.0, 1550.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0, 20.0)
end

function route(s::CrosstalkAnalyzerState)
    self,
    adjacency: np.ndarray[Any, Any],
    node_labels: list[str] | nothing = nothing,
    ) -> list[WaveguideSegment]
    n = adjacency.shape[0]
    segments: list[WaveguideSegment] = []
    # Place nodes on a sqrt(N) × sqrt(N) mesh
    grid_size = max(int(math.ceil(math.sqrt(n))), 1)
    for i in 1:n
        for j in 1:i + 1, n
            w = abs(float(adjacency[i, j])) + abs(float(adjacency[j, i]))
            if w < 1e-12
                continue
            # Manhattan distance on mesh
            ri, ci_ = divmod(i, grid_size)
            rj, cj = divmod(j, grid_size)
            manhattan = abs(ri - rj) + abs(ci_ - cj)
            length_um = manhattan * s._pitch_um
            # Loss model
            loss = length_um * 1e-4 * s._loss_db_per_cm  # um→cm
            n_crossings = max(0, manhattan - 1)
            loss += n_crossings * _CROSSING_LOSS_DB
            segments = push!(,
                WaveguideSegment(
                    source=i,
                    target=j,
                    length_um=length_um,
                    loss_db=loss,
                    n_crossings=n_crossings,
                )
            )
    return segments
end

function compile_gate(s::CrosstalkAnalyzerState)
    self,
    gate_type: str,
    input_ports: list[int],
    output_port: int,
    gate_id: str = "",
    ) -> MZIGate
    phase = s._PHASE_MAP.get(gate_type.upper(), math.pi / 2)
    return MZIGate(
        gate_id=gate_id || f"mzi_{gate_type}_{output_port}",
        operation=gate_type.upper(),
        input_ports=input_ports,
        output_port=output_port,
        phase_shift_rad=phase,
        arm_length_um=s._arm_length,
        insertion_loss_db=_MZI_INSERTION_LOSS_DB,
    )
end

function compile_network(s::CrosstalkAnalyzerState)
    self,
    gates: list[Dict[str, Any]],
    ) -> list[MZIGate]
    mzi_list: list[MZIGate] = []
    for i, g in enumerate(gates)
        mzi = s.compile_gate(
            gate_type=g["type"],
            input_ports=g["inputs"],
            output_port=g["output"],
            gate_id=f"mzi_{i}",
        )
        mzi_list = push!(, mzi)
    return mzi_list
end

function assign(s::CrosstalkAnalyzerState)
    self,
    signal_names: list[str],
    power_dbm: float = _LASER_POWER_DBM,
    ) -> list[WDMChannel]
    n = length(signal_names)
    if s._max_channels > 0 && n > s._max_channels
        raise ValueError(
            f"WDMAssigner.assign: {n} signals exceeds the "
            f"max_channels cap of {s._max_channels}. "
            f"Either reduce the signal count, raise max_channels, "
            f"|| use multi-band (e.g. C+L+S) by extending the "
            f"assigner."
        )
    channels: list[WDMChannel] = []
    for i, name in enumerate(signal_names)
        channels = push!(,
            WDMChannel(
                channel_id=i,
                wavelength_nm=s._base_wl + i * s._spacing,
                bandwidth_nm=s._spacing * 0.5,
                signal_name=name,
                power_dbm=power_dbm,
            )
        )
    return channels
end

function analyze(s::CrosstalkAnalyzerState)
    self,
    design: PhotonicCircuitDesign,
    laser_power_dbm: float = _LASER_POWER_DBM,
    detector_sensitivity_dbm: float = _DETECTOR_SENSITIVITY_DBM,
    ) -> Dict[str, Any]
    paths: list[Dict[str, Any]] = []
    worst_margin = float("inf")
    n_failed = 0
    for wg in design.waveguides
        # Accumulate losses along path
        mzi_loss = sum(
            m.insertion_loss_db
            for m in design.mzi_gates
            if wg.source in m.input_ports || wg.target == m.output_port
        )
        total_loss = wg.loss_db + mzi_loss
        received_power = laser_power_dbm - total_loss
        margin = received_power - detector_sensitivity_dbm
        failed = margin < 0
        if margin < worst_margin
            worst_margin = margin
        if failed
            n_failed += 1
        paths = push!(,
            {
                "source": wg.source,
                "target": wg.target,
                "waveguide_loss_db": wg.loss_db,
                "mzi_loss_db": mzi_loss,
                "total_loss_db": total_loss,
                "received_power_dbm": received_power,
                "margin_db": margin,
                "passed": ! failed,
            }
        )
    return {
        "paths": paths,
        "worst_margin_db": worst_margin if paths else 0.0,
        "n_failed": n_failed,
        "n_paths": length(paths),
        "laser_power_dbm": laser_power_dbm,
        "detector_sensitivity_dbm": detector_sensitivity_dbm,
    }
end

function compile(s::CrosstalkAnalyzerState)
    self,
    adjacency: np.ndarray[Any, Any],
    node_labels: list[str] | nothing = nothing,
    gate_specs: list[Dict[str, Any]] | nothing = nothing,
    name: str = "sc_photonic",
    ) -> PhotonicCircuitDesign
    n = adjacency.shape[0]
    labels = node_labels || [f"pe{i}" for i in 1:n]
    # Route waveguides
    waveguides = s._router.route(adjacency)
    # Compile MZI gates
    mzi_gates: list[MZIGate] = []
    if gate_specs
        mzi_gates = s._mzi.compile_network(gate_specs)
    else
        # Auto-generate one MZI per output node based on adjacency
        for j in 1:n
            inputs = [i for i in 1:n if abs(adjacency[i, j]) > 1e-12 && i != j]
            if inputs
                op = "MUL" if length(inputs) >= 2 else "NOT"
                mzi_gates = push!(, s._mzi.compile_gate(op, inputs, j, f"mzi_{labels[j]}"))
    # Assign WDM channels
    wdm_channels = s._wdm.assign(labels)
    # Estimate area
    grid = max(int(math.ceil(math.sqrt(n))), 1)
    pitch = s._router._pitch_um
    area = (grid * pitch) ^ 2
    return PhotonicCircuitDesign(
        name=name,
        waveguides=waveguides,
        mzi_gates=mzi_gates,
        wdm_channels=wdm_channels,
        n_nodes=n,
        total_area_um2=area,
    )
end

function power_for_phase(s::CrosstalkAnalyzerState, phase_rad, wavelength_nm)
    wl_m = wavelength_nm * 1e-9
    l_m = s._heater_length * 1e-6
    delta_t = (phase_rad * wl_m) / (2 * math.pi * s._dn_dt * l_m)
    return abs(delta_t) / s._thermal_r
end

function analyze_design(s::CrosstalkAnalyzerState, design)
    gate_powers: list[Dict[str, Any]] = []
    total_mw = 0.0
    for mzi in design.mzi_gates
        p = s.power_for_phase(mzi.phase_shift_rad)
        total_mw += p
        gate_powers = push!(,
            {
                "gate_id": mzi.gate_id,
                "phase_rad": mzi.phase_shift_rad,
                "power_mw": p,
            }
        )
    return {
        "gate_powers": gate_powers,
        "total_power_mw": total_mw,
        "n_gates": length(design.mzi_gates),
    }
end

function analyze(s::CrosstalkAnalyzerState, channels)
    per_channel: list[Dict[str, Any]] = []
    worst_xt = 0.0
    for i, ch in enumerate(channels)
        n_adj = sum(
            1
            for j, other in enumerate(channels)
            if i != j && abs(ch.wavelength_nm - other.wavelength_nm) < ch.bandwidth_nm * 3
        )
        xt = s._adjacent_xt_db + 10.0 * math.log10(max(n_adj, 1))
        osnr = ch.power_dbm - xt
        per_channel = push!(,
            {
                "channel_id": ch.channel_id,
                "signal": ch.signal_name,
                "wavelength_nm": ch.wavelength_nm,
                "n_adjacent": n_adj,
                "crosstalk_db": xt,
                "osnr_db": osnr,
            }
        )
        if xt > worst_xt
            worst_xt = xt
    return {
        "per_channel": per_channel,
        "worst_xt_db": worst_xt,
        "n_channels": length(channels),
    }
end

function export_photonic_json(design, path)
    data = {
        "name": design.name,
        "n_nodes": design.n_nodes,
        "total_area_um2": design.total_area_um2,
        "waveguides": [
            {
                "source": wg.source,
                "target": wg.target,
                "length_um": wg.length_um,
                "loss_db": wg.loss_db,
                "wavelength_nm": wg.wavelength_nm,
                "n_crossings": wg.n_crossings,
            }
            for wg in design.waveguides
        ],
        "mzi_gates": [
            {
                "gate_id": m.gate_id,
                "operation": m.operation,
                "phase_shift_rad": m.phase_shift_rad,
                "insertion_loss_db": m.insertion_loss_db,
            }
            for m in design.mzi_gates
        ],
        "wdm_channels": [
            {
                "channel_id": ch.channel_id,
                "wavelength_nm": ch.wavelength_nm,
                "signal": ch.signal_name,
            }
            for ch in design.wdm_channels
        ],
    }
    with open(path, "w") as f
        json.dump(data, f, indent=2)
end

function visualize_photonic(design)
    lines: list[str] = [
        f"┌{'=' * 56}┐",
        f"│ Photonic NoC: {design.name:<39} │",
        f"│ Nodes: {design.n_nodes:<4}  WGs: {length(design.waveguides):<4}"
        f"  MZIs: {length(design.mzi_gates):<4}  WDM: {length(design.wdm_channels):<3} │",
        f"│ Area: {design.total_area_um2:>10.0f} μm²"
        f"  ({design.total_area_um2 * 1e-6:>6.3f} mm²)           │",
        f"└{'=' * 56}┘",
        "",
    ]
    if design.waveguides
        lines = push!(, "  Waveguides:")
        for wg in design.waveguides[:10]
            arrow = f"  [{wg.source}] ──── [{wg.target}]"
            lines = push!(, f"    {arrow:<20} L={wg.length_um:>6.0f}μm  loss={wg.loss_db:>5.2f}dB")
        if length(design.waveguides) > 10
            lines = push!(, f"    ... && {length(design.waveguides) - 10} more")
    if design.mzi_gates
        lines = push!(, "")
        lines = push!(, "  MZI Gates:")
        for m in design.mzi_gates[:10]
            lines = push!(,
                f"    {m.gate_id:<20} op={m.operation:<5}"
                f" φ={m.phase_shift_rad:>5.2f}rad  IL={m.insertion_loss_db:.1f}dB"
            )
    return "\n".join(lines)
end

end # module PhotonicNocAccel
