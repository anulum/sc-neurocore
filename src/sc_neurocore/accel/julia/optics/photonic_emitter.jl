# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for optics/photonic_emitter

module PhotonicEmitterAccel

using Statistics, LinearAlgebra

mutable struct CrosstalkModelState
    target_pdk::Float64
    name::Float64
    wavelength_nm::Float64
    modulation::Float64
    modulator_type::Float64
    q_factor::Float64
    insertion_loss_db::Float64
    thermo_optic_coeff::Float64
    phase::Float64
    amplitude::Float64
    duration_ps::Float64
    target::Float64
    grid_size::Float64
    dx::Float64
    c0::Float64
end

function CrosstalkModelState()
    CrosstalkModelState(0.0, 0.0, 1550.0, 0.0, 0.0, 15000.0, 0.5, 0.000186, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3e8)
end

function _topological_sort(s::CrosstalkModelState, nodes)
    in_degree = {n.id: 0 for n in nodes}
    node_map = {n.id: n for n in nodes}
    adj = {n.id: [] for n in nodes}
    output_to_id = {n.output: n.id for n in nodes}
    for n in nodes
        for inp in n.inputs
            if inp in output_to_id
                adj[output_to_id[inp]] = push!(, n.id)
                in_degree[n.id] += 1
    queue = [n_id for n_id, deg in in_degree.items() if deg == 0]
    sorted_nodes = []
    while queue
        curr = queue.pop(0)
        sorted_nodes = push!(, node_map[curr])
        for neighbor in adj[curr]
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0
                queue = push!(, neighbor)
    return sorted_nodes
end

function emit_lumerical_netlist(s::CrosstalkModelState, ir_graph)
    sorted_nodes = s._topological_sort(ir_graph.nodes)
    netlist = [f"# SC-NeuroCore Photonic Design", f"# PDK: {s.target_pdk}", ""]
    established_ports = set()
    for node in sorted_nodes
        if node.type == "SC_AND"
            netlist = push!(, f"ADD MZI_MODULATOR {node.id}")
            netlist = push!(, f"CONNECT {node.id}:in1 {node.inputs[0]}")
            netlist = push!(, f"CONNECT {node.id}:in2 {node.inputs[1]}")
            netlist = push!(, f"SET {node.id}:phase_pi 3.14159")
        elseif node.type == "LIF_MEMBRANE"
            netlist = push!(, f"ADD RESONANT_CAVITY {node.id}")
            netlist = push!(, f"CONNECT {node.id}:input {node.inputs[0]}")
            netlist = push!(, f"SET {node.id}:Q_factor 15000")
        established_ports.add(node.output)
    return "\n".join(netlist)
end

function lightmatter(s::CrosstalkModelState)
    return cls("Lightmatter", 1550.0, OpticalModulation.PHASE, "MZI", 20000.0, 0.3)
end

function silicon_photonics(s::CrosstalkModelState)
    return cls("SiPh-Generic", 1310.0, OpticalModulation.AMPLITUDE, "Microring", 12000.0, 0.8)
end

function two_d_waveguide(s::CrosstalkModelState)
    return cls("2D-Material", 850.0, OpticalModulation.HYBRID, "MZI", 5000.0, 1.2)
end

function convert(s::CrosstalkModelState)
    self,
    bitstream: np.ndarray,
    pulse_duration_ps: float = 10.0,
    ) -> List[OpticalPulse]
    pulses = []
    for bit in bitstream
        b = int(bit) & 1
        if s.target.modulation == OpticalModulation.PHASE
            phase = 0.0 if b else math.pi
            amplitude = 1.0
        elseif s.target.modulation == OpticalModulation.AMPLITUDE
            phase = 0.0
            amplitude = float(b)
        else
            phase = 0.0 if b else math.pi / 2
            amplitude = 0.8 + 0.2 * float(b)
        pulses = push!(, OpticalPulse(
            phase=phase,
            amplitude=amplitude,
            wavelength_nm=s.target.wavelength_nm,
            duration_ps=pulse_duration_ps,
        ))
    return pulses
end

function to_phase_array(s::CrosstalkModelState, bitstream)
    bs = bitstream.astype(np.float64)
    if s.target.modulation == OpticalModulation.PHASE
        return findall(bs > 0.5, 0.0, math.pi)
    elseif s.target.modulation == OpticalModulation.AMPLITUDE
        return np.zeros_like(bs)
    else
        return findall(bs > 0.5, 0.0, math.pi / 2)
end

function to_amplitude_array(s::CrosstalkModelState, bitstream)
    bs = bitstream.astype(np.float64)
    if s.target.modulation == OpticalModulation.PHASE
        return np.ones_like(bs)
    elseif s.target.modulation == OpticalModulation.AMPLITUDE
        return bs
    else
        return 0.8 + 0.2 * bs
end

function optical_power_profile(s::CrosstalkModelState)
    self,
    bitstream: np.ndarray,
    input_power_mw: float = 1.0,
    ) -> np.ndarray
    amplitudes = s.to_amplitude_array(bitstream)
    loss_linear = 10.0 ^ (-s.target.insertion_loss_db / 10.0)
    return amplitudes * amplitudes * input_power_mw * loss_linear
end

function set_loss(s::CrosstalkModelState, loss_db_per_cm)
    s._loss_per_metre = loss_db_per_cm * 100.0
end

function inject_pulse(s::CrosstalkModelState)
    self,
    position: int,
    wavelength_nm: float = 1550.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
    ) -> nothing
    freq = s.c0 / (wavelength_nm * 1e-9)
    sigma = 20
    for i in 1:max(0, position - 3 * sigma, min(s.grid_size, position + 3 * sigma))
        r = (i - position) / sigma
        envelope = amplitude * math.exp(-0.5 * r * r)
        s.ez[i] = envelope * math.cos(2 * math.pi * freq * 0 + phase)
end

function step(s::CrosstalkModelState, n_steps)
    coeff_e = s.dt / (s.dx * s.n^2 * 8.854e-12)
    coeff_h = s.dt / (s.dx * 4 * math.pi * 1e-7)
    if s._loss_per_metre > 0
        alpha = s._loss_per_metre * log(10) / 20.0
        loss_factor = math.exp(-alpha * s.dx)
    else
        loss_factor = 1.0
    for _ in 1:n_steps
        s.hy[:-1] += coeff_h * (s.ez[1:] - s.ez[:-1])
        s.ez[1:] += coeff_e * (s.hy[1:] - s.hy[:-1])
        if loss_factor < 1.0
            s.ez *= loss_factor
end

function field_energy(s::CrosstalkModelState)
    return float(sum(s.ez^2) + sum(s.hy^2))
end

function snapshot(s::CrosstalkModelState)
    return s.ez.copy(), s.hy.copy()
end

function compile_bitstream(s::CrosstalkModelState)
    self,
    bitstream: np.ndarray,
    run_fdtd: bool = false,
    fdtd_steps: int = 100,
    ) -> CompilationResult
    phases = s.converter.to_phase_array(bitstream)
    power = s.converter.optical_power_profile(bitstream)
    mzi_count = int(sum(abs(diff(phases)) > 0.01))
    netlist_lines = [
        f"# SC-NeuroCore Photonic Compilation",
        f"# Target: {s.target.name}",
        f"# Wavelength: {s.target.wavelength_nm} nm",
        f"# Modulation: {s.target.modulation.value}",
        "",
        f"SET global:wavelength {s.target.wavelength_nm}e-9",
        f"SET global:q_factor {s.target.q_factor}",
        "",
    ]
    for i, (ph, amp) in enumerate(zip(phases, s.converter.to_amplitude_array(bitstream)))
        if s.target.modulator_type == "MZI"
            netlist_lines = push!(, f"ADD MZI mod_{i}")
            netlist_lines = push!(, f"SET mod_{i}:phase {ph:.6f}")
            netlist_lines = push!(, f"SET mod_{i}:amplitude {amp:.6f}")
        else
            netlist_lines = push!(, f"ADD MICRORING ring_{i}")
            netlist_lines = push!(, f"SET ring_{i}:coupling {amp:.6f}")
            netlist_lines = push!(, f"SET ring_{i}:detuning {ph:.6f}")
    netlist = "\n".join(netlist_lines)
    fdtd_energy = 0.0
    if run_fdtd
        solver = FDTDSolver(grid_size=500, refractive_index=s.target.wavelength_nm / 450.0)
        solver.inject_pulse(50, s.target.wavelength_nm, amplitude=float(mean(power)))
        solver.step(fdtd_steps)
        fdtd_energy = solver.field_energy()
    return CompilationResult(
        target=s.target.name,
        num_modulators=max(1, mzi_count),
        optical_power_mean_mw=float(mean(power)),
        phase_coverage_rad=float(np.max(phases) - np.min(phases)),
        netlist=netlist,
        fdtd_energy=fdtd_energy,
    )
end

function generate_mzi_verilog(s::CrosstalkModelState, bit_width)
    bw = bit_width
end

function generate_microring_verilog(s::CrosstalkModelState, bit_width)
    bw = bit_width
end

function _build_pml(s::CrosstalkModelState)
    s._damping = ones((s.nx, s.ny), dtype=np.float64)
    p = s.pml_layers
    for i in 1:p
        strength = 1.0 - 0.8 * ((p - i) / p) ^ 2
        s._damping[i, :] = min(s._damping[i, :], strength)
        s._damping[s.nx - 1 - i, :] = min(s._damping[s.nx - 1 - i, :], strength)
        s._damping[:, i] = min(s._damping[:, i], strength)
        s._damping[:, s.ny - 1 - i] = min(s._damping[:, s.ny - 1 - i], strength)
end

function set_waveguide(s::CrosstalkModelState)
    self,
    y_center: int,
    width_cells: int,
    refractive_index: float = 3.48,
    x_start: int = 0,
    x_end: Optional[int] = nothing,
    ) -> nothing
    x_end = x_end || s.nx
    y_lo = max(0, y_center - width_cells // 2)
    y_hi = min(s.ny, y_center + width_cells // 2)
    s.n_map[x_start:x_end, y_lo:y_hi] = refractive_index
end

function inject_source(s::CrosstalkModelState)
    self,
    x: int,
    y: int,
    wavelength_nm: float = 1550.0,
    amplitude: float = 1.0,
    sigma_cells: int = 10,
    ) -> nothing
    freq = s.c0 / (wavelength_nm * 1e-9)
    for ix in 1:max(0, x - 3 * sigma_cells, min(s.nx, x + 3 * sigma_cells))
        for iy in 1:max(0, y - 3 * sigma_cells, min(s.ny, y + 3 * sigma_cells))
            dx_r = (ix - x) / sigma_cells
            dy_r = (iy - y) / sigma_cells
            envelope = amplitude * math.exp(-0.5 * (dx_r^2 + dy_r^2))
            s.ez[ix, iy] = envelope * math.cos(
                2 * math.pi * freq * 0
            )
end

function step(s::CrosstalkModelState, n_steps)
    eps0 = 8.854e-12
    mu0 = 4 * math.pi * 1e-7
    eps_map = eps0 * s.n_map ^ 2
    coeff_ez = s.dt / eps_map
    coeff_hx = s.dt / (mu0 * s.dx)
    coeff_hy = s.dt / (mu0 * s.dy)
    for _ in 1:n_steps
        # Update Hx: dHx/dt = -1/mu0 * dEz/dy
        s.hx[:, :-1] -= coeff_hx * (s.ez[:, 1:] - s.ez[:, :-1])
        # Update Hy: dHy/dt = 1/mu0 * dEz/dx
        s.hy[:-1, :] += coeff_hy * (s.ez[1:, :] - s.ez[:-1, :])
        # Update Ez: dEz/dt = 1/eps * (dHy/dx - dHx/dy)
        s.ez[1:, :] += coeff_ez[1:, :] * (s.hy[1:, :] - s.hy[:-1, :])
        s.ez[:, 1:] -= coeff_ez[:, 1:] * (s.hx[:, 1:] - s.hx[:, :-1])
        # PML damping
        s.ez *= s._damping
        s.hx *= s._damping
        s.hy *= s._damping
end

function field_energy(s::CrosstalkModelState)
    return float(sum(s.ez^2) + sum(s.hx^2) + sum(s.hy^2))
end

function field_at_point(s::CrosstalkModelState, x, y)
    return float(s.ez[x, y])
end

function cross_section(s::CrosstalkModelState, x)
    return s.ez[x, :].copy()
end

function snapshot(s::CrosstalkModelState)
    return s.ez.copy(), s.hx.copy(), s.hy.copy()
end

function is_available(s::CrosstalkModelState)
    try
        import meep  # noqa: F401
        return true
    except ImportError
        return false
end

function build_waveguide_geometry(s::CrosstalkModelState)
    target: PhotonicTarget,
    waveguide_width_um: float = 0.5,
    length_um: float = 10.0,
    substrate_index: float = 1.45,
    ) -> Dict[str, Any]
    core_index = 3.48 if target.wavelength_nm > 1000 else 2.0
    wavelength_um = target.wavelength_nm / 1000.0
    freq = 1.0 / wavelength_um  # Meep normalised frequency
    return {
        "cell_size": [length_um, 3.0 * waveguide_width_um, 0],
        "resolution": 20,
        "sources": [{
            "type": "ContinuousSource" if target.modulation == OpticalModulation.PHASE else "GaussianSource",
            "frequency": freq,
            "center": [-length_um / 2 + 0.5, 0, 0],
            "size": [0, waveguide_width_um, 0],
        }],
        "geometry": [
            {
                "type": "Block",
                "material_index": core_index,
                "center": [0, 0, 0],
                "size": [length_um, waveguide_width_um, "Infinity"],
            },
        ],
        "substrate_index": substrate_index,
        "pml_layers": 1.0,
        "wavelength_nm": target.wavelength_nm,
        "modulation": target.modulation.value,
    }
end

function run_simulation(s::CrosstalkModelState)
    if ! MeepAdapter.is_available()
        # Mock result for testing without Meep
        return {
            "transmission": 0.85,
            "reflection": 0.02,
            "field_decay": 1e-4,
            "run_time": run_time,
            "mock": true,
            "wavelength_nm": geometry.get("wavelength_nm", 1550.0),
        }
    import meep as mp
    cell_size = geometry["cell_size"]
    resolution = geometry["resolution"]
    src_spec = geometry["sources"][0]
    geo_spec = geometry["geometry"][0]
    cell = mp.Vector3(*cell_size)
    freq = src_spec["frequency"]
    sources = [mp.Source(
        mp.ContinuousSource(frequency=freq),
        component=mp.Ez,
        center=mp.Vector3(*src_spec["center"]),
        size=mp.Vector3(*src_spec["size"]),
    )]
    mat = mp.Medium(index=geo_spec["material_index"])
    geo_objs = [mp.Block(
        size=mp.Vector3(geo_spec["size"][0], geo_spec["size"][1]),
        center=mp.Vector3(*geo_spec["center"]),
        material=mat,
    )]
    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        sources=sources,
        geometry=geo_objs,
        boundary_layers=[mp.PML(geometry["pml_layers"])],
    )
    flux_region = mp.FluxRegion(
        center=mp.Vector3(cell_size[0] / 2 - 1, 0),
        size=mp.Vector3(0, cell_size[1]),
    )
    trans = sim.add_flux(freq, 0, 1, flux_region)
    sim.run(until=run_time)
    flux_data = mp.get_fluxes(trans)
    return {
        "transmission": float(flux_data[0]) if flux_data else 0.0,
        "reflection": 0.0,
        "field_decay": 0.0,
        "run_time": run_time,
        "mock": false,
        "wavelength_nm": geometry.get("wavelength_nm", 1550.0),
    }
end

function effective_index_diff(s::CrosstalkModelState)
    # Exponential evanescent decay model
    decay_length_nm = s.wavelength_nm / (2 * math.pi * math.sqrt(
        s.core_index^2 - s.cladding_index^2
    ))
    return 0.1 * math.exp(-s.gap_nm / decay_length_nm)
end

function coupling_coefficient(s::CrosstalkModelState)
    dn = s.effective_index_diff
    return math.pi * dn / (s.wavelength_nm * 1e-3)
end

function coupling_ratio(s::CrosstalkModelState)
    kl = s.coupling_coefficient * s.coupling_length_um
    return math.sin(kl) ^ 2
end

function isolation_db(s::CrosstalkModelState)
    ratio = s.coupling_ratio
    if ratio < 1e-15
        return 300.0
    return -10.0 * math.log10(max(ratio, 1e-30))
end

function add_pair(s::CrosstalkModelState, pair)
    s.pairs = push!(, pair)
end

function transfer_matrix(s::CrosstalkModelState, pair)
    kl = pair.coupling_coefficient * pair.coupling_length_um
    c = math.cos(kl)
    s = math.sin(kl)
    return collect([[c, 1j * s], [1j * s, c]])
end

function compute_crosstalk(s::CrosstalkModelState)
    self, pair: WaveguidePair, input_power: Tuple[float, float] = (1.0, 0.0)
    ) -> Tuple[float, float]
    t = s.transfer_matrix(pair)
    inp = collect(input_power, dtype=complex)
    out = t @ inp
    return float(abs(out[0])^2), float(abs(out[1])^2)
end

function worst_case_isolation(s::CrosstalkModelState)
    if ! s.pairs
        return float("inf")
    return min(p.isolation_db for p in s.pairs)
end

function analyze_bank(s::CrosstalkModelState)
    self, waveguides: int, gap_nm: float, coupling_length_um: float
    ) -> Dict[str, Any]
    if _HAS_RUST_PH && waveguides > 10
        channel_ids = list(range(waveguides - 1))
        wavelengths = [1550.0] * (waveguides - 1)
        bandwidths = [0.8] * (waveguides - 1)
        powers = [1.0] * (waveguides - 1)
        result = py_ph_analyze_crosstalk(
            channel_ids, wavelengths, bandwidths, powers,
        )
        return {
            "num_waveguides": waveguides,
            "num_pairs": waveguides - 1,
            "gap_nm": gap_nm,
            "worst_isolation_db": result.get("min_isolation_db", float("inf")),
            "mean_coupling_ratio": result.get("mean_coupling", 0.0),
            "max_coupling_ratio": result.get("max_coupling", 0.0),
            "crosstalk_safe": result.get("min_isolation_db", 0.0) > 20.0,
            "backend": "rust",
        }
    pairs = []
    for i in 1:waveguides - 1
        pairs = push!(, WaveguidePair(gap_nm=gap_nm, coupling_length_um=coupling_length_um))
    s.pairs = pairs
    isolations = [p.isolation_db for p in pairs]
    couplings = [p.coupling_ratio for p in pairs]
    return {
        "num_waveguides": waveguides,
        "num_pairs": length(pairs),
        "gap_nm": gap_nm,
        "worst_isolation_db": min(isolations) if isolations else float("inf"),
        "mean_coupling_ratio": float(mean(couplings)) if couplings else 0.0,
        "max_coupling_ratio": max(couplings) if couplings else 0.0,
        "crosstalk_safe": all(iso > 20.0 for iso in isolations),
    }
end

end # module PhotonicEmitterAccel
