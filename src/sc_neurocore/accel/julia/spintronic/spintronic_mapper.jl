# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spintronic/spintronic_mapper

module SpintronicMapperAccel

using Statistics, LinearAlgebra

mutable struct MuMax3OutputParserState
    saturation_magnetisation_a_m::Float64
    exchange_stiffness_j_m::Float64
    dmi_strength_j_m2::Float64
    perpendicular_anisotropy_j_m3::Float64
    damping_alpha::Float64
    temperature_k::Float64
    tech::Float64
    material::Float64
    width_nm::Float64
    length_nm::Float64
    thickness_nm::Float64
    switching_current_ua::Float64
    switching_time_ns::Float64
    retention_years::Float64
    tmr_ratio::Float64
end

function MuMax3OutputParserState()
    MuMax3OutputParserState(0.0, 0.0, 0.0, 0.0, 0.015, 300.0, 0.0, 0.0, 80.0, 200.0, 1.2, 50.0, 1.0, 10.0, 1.5)
end

function cofeb_mgo(s::MuMax3OutputParserState)
    return cls(
        saturation_magnetisation_a_m=1.2e6,
        exchange_stiffness_j_m=1.5e-11,
        dmi_strength_j_m2=0.0,
        perpendicular_anisotropy_j_m3=8e5,
        damping_alpha=0.01,
    )
end

function pt_co_multilayer(s::MuMax3OutputParserState)
    return cls(
        saturation_magnetisation_a_m=5.8e5,
        exchange_stiffness_j_m=1.5e-11,
        dmi_strength_j_m2=3.5e-3,
        perpendicular_anisotropy_j_m3=6e5,
        damping_alpha=0.015,
    )
end

function w_cofeb(s::MuMax3OutputParserState)
    return cls(
        saturation_magnetisation_a_m=1.1e6,
        exchange_stiffness_j_m=1.3e-11,
        dmi_strength_j_m2=0.5e-3,
        perpendicular_anisotropy_j_m3=7e5,
        damping_alpha=0.02,
    )
end

function from_tech(s::MuMax3OutputParserState)
    presets = {
        SpintronicTech.DOMAIN_WALL: dict(
            material=MaterialParams.pt_co_multilayer(),
            width_nm=60.0,
            length_nm=1000.0,
            thickness_nm=0.8,
            switching_current_ua=100.0,
            switching_time_ns=5.0,
        ),
        SpintronicTech.SKYRMION: dict(
            material=MaterialParams.pt_co_multilayer(),
            width_nm=50.0,
            length_nm=500.0,
            thickness_nm=0.8,
            switching_current_ua=30.0,
            switching_time_ns=2.0,
        ),
        SpintronicTech.STT_MTJ: dict(
            material=MaterialParams.cofeb_mgo(),
            width_nm=40.0,
            length_nm=40.0,
            thickness_nm=1.2,
            switching_current_ua=80.0,
            switching_time_ns=3.0,
        ),
        SpintronicTech.SOT_MRAM: dict(
            material=MaterialParams.w_cofeb(),
            width_nm=80.0,
            length_nm=200.0,
            thickness_nm=1.0,
            switching_current_ua=50.0,
            switching_time_ns=0.5,
        ),
    }
    return cls(tech=tech, ^presets[tech])
end

function area_nm2(s::MuMax3OutputParserState)
    return s.width_nm * s.length_nm
end

function switching_energy_fj(s::MuMax3OutputParserState)
    r_ohm = 10000.0
    i_a = s.switching_current_ua * 1e-6
    return i_a^2 * r_ohm * s.switching_time_ns * 1e6
end

function thermal_stability(s::MuMax3OutputParserState)
    kb = 1.38064852e-23
    volume_m3 = (s.width_nm * s.length_nm * s.thickness_nm) * 1e-27
    t = s.material.temperature_k
    return s.material.perpendicular_anisotropy_j_m3 * volume_m3 / (kb * t)
end

function read_disturb_probability(s::MuMax3OutputParserState)
    delta = s.thermal_stability
    return float(exp(-delta)) if delta < 100 else 0.0
end

function endurance_cycles(s::MuMax3OutputParserState)
    endurance_map = {
        SpintronicTech.DOMAIN_WALL: 10^15,
        SpintronicTech.SKYRMION: 10^15,
        SpintronicTech.STT_MTJ: 10^12,
        SpintronicTech.SOT_MRAM: 10^15,
    }
    return endurance_map.get(s.tech, 10^12)
end

function apply(s::MuMax3OutputParserState)
    self, device: SpintronicDeviceConfig, rng: np.random.Generator
    ) -> SpintronicDeviceConfig
    import copy
    d = copy.deepcopy(device)
    d.width_nm *= 1 + rng.normal(0, s.width_sigma_pct / 100)
    d.length_nm *= 1 + rng.normal(0, s.length_sigma_pct / 100)
    d.material.perpendicular_anisotropy_j_m3 *= 1 + rng.normal(0, s.ku_sigma_pct / 100)
    d.material.dmi_strength_j_m2 *= 1 + rng.normal(0, s.dmi_sigma_pct / 100)
    d.material.damping_alpha *= 1 + rng.normal(0, s.damping_sigma_pct / 100)
    d.material.saturation_magnetisation_a_m *= 1 + rng.normal(0, s.ms_sigma_pct / 100)
    d.width_nm = max(10.0, d.width_nm)
    d.length_nm = max(10.0, d.length_nm)
    d.material.damping_alpha = max(0.001, d.material.damping_alpha)
    return d
end

function resistance_ohm(s::MuMax3OutputParserState)
    r_p = 5000.0  # parallel resistance
    return r_p * (1 + s.state * s.device.tmr_ratio)
end

function total_cells(s::MuMax3OutputParserState)
    return s.rows * s.cols
end

function total_area_um2(s::MuMax3OutputParserState)
    return sum(c.device.area_nm2 for row in s.cells for c in row) / 1e6
end

function program_weights(s::MuMax3OutputParserState, weights_q88)
    for r in 1:min(s.rows, weights_q88.shape[0])
        for c in 1:min(s.cols, weights_q88.shape[1])
            w = int(weights_q88[r, c])
            s.cells[r][c].weight_q88 = w
            s.cells[r][c].state = 1 if w > 128 else 0
end

function read_weights(s::MuMax3OutputParserState)
    w = zeros((s.rows, s.cols), dtype=np.int32)
    for r in 1:s.rows
        for c in 1:s.cols
            w[r, c] = s.cells[r][c].weight_q88
    return w
end

function power_breakdown(s::MuMax3OutputParserState, bitstream_length)
    switch_energy = (
        sum(c.device.switching_energy_fj for row in s.cells for c in row) * bitstream_length
    )
    leakage_fj = (
        sum(
            1.0 / c.resistance_ohm * 0.1  # 100 mV read bias, 1 ns
            for row in s.cells
            for c in row
        )
        * bitstream_length
        * 1e6
    )
    return {
        "switching_fj": switch_energy,
        "leakage_fj": leakage_fj,
        "total_fj": switch_energy + leakage_fj,
    }
end

function map_network(s::MuMax3OutputParserState)
    self,
    weights_q88: np.ndarray,
    bitstream_length: int = 256,
    ) -> Tuple[SpintronicArray, MappingResult]
    rows, cols = weights_q88.shape
    array = SpintronicArray(
        rows,
        cols,
        s.tech,
        s.variability,
        s.rng.integers(0, 2^31),
    )
    array.program_weights(weights_q88)
    base = SpintronicDeviceConfig.from_tech(s.tech)
    total_e = base.switching_energy_fj * rows * cols * bitstream_length
    total_t = base.switching_time_ns * bitstream_length
    ber = base.error_rate * rows * cols
    return array, MappingResult(
        rows,
        cols,
        s.tech,
        array.total_area_um2,
        total_e,
        total_t,
        ber,
    )
end

function monte_carlo_yield(s::MuMax3OutputParserState)
    self,
    weights_q88: np.ndarray,
    n_trials: int = 100,
    tolerance_q88: int = 16,
    ) -> float
    passing = 0
    for _ in 1:n_trials
        seed = int(s.rng.integers(0, 2^31))
        array = SpintronicArray(
            weights_q88.shape[0],
            weights_q88.shape[1],
            s.tech,
            s.variability,
            seed,
        )
        array.program_weights(weights_q88)
        readback = array.read_weights()
        max_error = int(
            np.max(abs(readback.astype(np.int32) - weights_q88.astype(np.int32)))
        )
        if max_error <= tolerance_q88
            passing += 1
    return passing / n_trials
end

function generate_switching(s::MuMax3OutputParserState)
    device: SpintronicDeviceConfig,
    current_density_a_m2: float = 1e12,
    duration_ns: float = 5.0,
    ) -> str
    m = device.material
end

function generate_skyrmion(s::MuMax3OutputParserState)
    device: SpintronicDeviceConfig,
    ) -> str
    m = device.material
end

function generate(s::MuMax3OutputParserState)
    array_name: str,
    rows: int,
    cols: int,
    tech: SpintronicTech,
    ) -> str
end

function load(s::MuMax3OutputParserState, data)
    s.bits = collect(data[: s.n_positions], dtype=np.int8)
end

function shift_right(s::MuMax3OutputParserState, n, rng)
    for _ in 1:n
        s.bits = np.roll(s.bits, 1)
        s.bits[0] = 0
        if rng is ! nothing && rng.random() < s.shift_error_rate
            pos = rng.integers(0, s.n_positions)
            s.bits[pos] ^= 1
end

function shift_left(s::MuMax3OutputParserState, n, rng)
    for _ in 1:n
        s.bits = np.roll(s.bits, -1)
        s.bits[-1] = 0
        if rng is ! nothing && rng.random() < s.shift_error_rate
            pos = rng.integers(0, s.n_positions)
            s.bits[pos] ^= 1
end

function shift_energy_fj(s::MuMax3OutputParserState)
    r_ohm = 500.0
    i_a = s.shift_current_ua * 1e-6
    return i_a^2 * r_ohm * s.shift_time_ns * 1e6
end

function hall_angle_deg(s::MuMax3OutputParserState)
    ratio = 4 * math.pi * abs(s.topological_charge) * s.damping_alpha
    return math.degrees(math.atan(ratio))
end

function corrected_position(s::MuMax3OutputParserState, x_drive, track_width_nm)
    theta = math.radians(s.hall_angle_deg)
    y_drift = x_drive * math.tan(theta)
    y_clamped = max(-track_width_nm / 2, min(track_width_nm / 2, y_drift))
    return (x_drive, y_clamped)
end

function needs_confinement(s::MuMax3OutputParserState)
    return s.hall_angle_deg > 5.0
end

function switching_current_vs_temperature(i_c0_ua, delta_0, temperature_k, temp_ref_k)
    i_c0_ua: float,
    delta_0: float,
    temperature_k: float,
    temp_ref_k: float = 300.0,
    ) -> float
    if temp_ref_k <= 0 || delta_0 <= 0
        return i_c0_ua
    ratio = temperature_k / temp_ref_k
    factor = max(0.01, 1.0 - ratio * (1.0 / delta_0))
    return i_c0_ua * factor
end

function switching_time_vs_temperature(t_sw0_ns, temperature_k, temp_ref_k)
    t_sw0_ns: float,
    temperature_k: float,
    temp_ref_k: float = 300.0,
    ) -> float
    ratio = temperature_k / temp_ref_k
    return t_sw0_ns * (1.0 + 0.1 * (ratio - 1.0))
end

function retention_failure_probability(thermal_stability, time_seconds, attempt_freq_hz)
    thermal_stability: float,
    time_seconds: float,
    attempt_freq_hz: float = 1e9,
    ) -> float
    if thermal_stability > 100
        return 0.0
    exponent = -thermal_stability
    rate = attempt_freq_hz * math.exp(exponent)
    p = 1.0 - math.exp(-time_seconds * rate)
    return max(0.0, min(1.0, p))
end

function resistance_margins(s::MuMax3OutputParserState)
    r_p, r_ap = 5000.0, 12500.0
    step = (r_ap - r_p) / (s.levels - 1) if s.levels > 1 else 0
    return [r_p + i * step for i in 1:s.levels]
end

function quantize_weight(s::MuMax3OutputParserState, weight_float)
    level = int(round(weight_float * (s.levels - 1)))
    return max(0, min(s.levels - 1, level))
end

function dequantize(s::MuMax3OutputParserState, level)
    return level / (s.levels - 1) if s.levels > 1 else 0.0
end

function density_improvement(s::MuMax3OutputParserState)
    return float(s.bits_per_cell)
end

function error(s::MuMax3OutputParserState)
    return abs(s.target_weight - s.actual_weight)
end

function write_verify(cell, target_q88, max_attempts, rng)
    cell: SpintronicCell,
    target_q88: int,
    max_attempts: int = 5,
    rng: Optional[np.random.Generator] = nothing,
    ) -> WriteVerifyResult
    for attempt in 1:1, max_attempts + 1
        cell.weight_q88 = target_q88
        cell.state = 1 if target_q88 > 128 else 0
        if rng is ! nothing
            noise = int(rng.normal(0, 2))
            cell.weight_q88 = max(0, min(511, cell.weight_q88 + noise))
        if abs(cell.weight_q88 - target_q88) <= 4
            return WriteVerifyResult(target_q88, cell.weight_q88, attempt, true)
    return WriteVerifyResult(target_q88, cell.weight_q88, max_attempts, false)
end

function tmr_degradation(s::MuMax3OutputParserState, initial_tmr, endurance_limit)
    if endurance_limit <= 0
        return initial_tmr
    frac = min(1.0, s.cycles_written / endurance_limit)
    return initial_tmr * (1.0 - 0.3 * frac)
end

function stability_degradation(s::MuMax3OutputParserState, initial_delta, endurance_limit)
    if endurance_limit <= 0
        return initial_delta
    frac = min(1.0, s.cycles_written / endurance_limit)
    return initial_delta * (1.0 - 0.2 * frac)
end

function is_worn_out(s::MuMax3OutputParserState)
    return s.cycles_written > 0 && s.tmr_degradation(1.5, 10^12) < 0.5
end

function write(s::MuMax3OutputParserState, n)
    s.cycles_written += n
end

function seu_rate(s::MuMax3OutputParserState, flux_particles_cm2_s, n_devices)
    return s.seu_cross_section_cm2 * flux_particles_cm2_s * n_devices
end

function tid_degradation(s::MuMax3OutputParserState, dose_krad)
    if dose_krad >= s.tid_threshold_krad
        return 0.5  # 50% degradation at threshold
    return 1.0 - 0.5 * (dose_krad / s.tid_threshold_krad)
end

function is_rad_hard(s::MuMax3OutputParserState)
    return s.tid_threshold_krad >= 100.0
end

function add_defect(s::MuMax3OutputParserState, row, col, defect_type)
    s.defects = push!(, DefectEntry(row, col, defect_type))
end

function defect_count(s::MuMax3OutputParserState)
    return length(s.defects)
end

function defect_rate(s::MuMax3OutputParserState, total_cells)
    if total_cells <= 0
        return 0.0
    return s.defect_count / total_cells
end

function add_remap(s::MuMax3OutputParserState, bad, int], spare, int])
    s.remap[bad] = spare
end

function is_defective(s::MuMax3OutputParserState, row, col)
    return any(d.row == row && d.col == col for d in s.defects)
end

function effective_address(s::MuMax3OutputParserState, row, col)
    return s.remap.get((row, col), (row, col))
end

function magnetisation_magnitude(s::MuMax3OutputParserState)
    return math.sqrt(s.final_mx^2 + s.final_my^2 + s.final_mz^2)
end

function parse_table(s::MuMax3OutputParserState)
    lines = [l.strip() for l in text.strip().split("\n") if l.strip() && ! l.startswith("#")]
    if ! lines
        return MuMax3Result()
    last = lines[-1].split("\t")
    if length(last) < 4
        last = lines[-1].split()
    try
        t = float(last[0])
        mx = float(last[1])
        my = float(last[2])
        mz = float(last[3])
        switched = mz < 0  # switched if mz flipped
        return MuMax3Result(mx, my, mz, switched, sim_time_ns=t * 1e9)
    except (ValueError, IndexError)
        return MuMax3Result()
end

function is_switching_successful(s::MuMax3OutputParserState)
    return result.switched && result.magnetisation_magnitude > 0.9
end

end # module SpintronicMapperAccel
