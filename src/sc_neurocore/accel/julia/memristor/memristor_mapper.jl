# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for memristor/memristor_mapper

module MemristorMapperAccel

using Statistics, LinearAlgebra

mutable struct VerilogEmitterState
    technology::Float64
    g_on::Float64
    g_off::Float64
    sigma_g::Float64
    sigma_rw::Float64
    num_levels::Float64
    r_wire_per_cell::Float64
    rows::Float64
    cols::Float64
    stuck_on::Float64
    stuck_off::Float64
    elapsed_s::Float64
    mean_drift_fraction::Float64
    max_drift_fraction::Float64
    levels_shifted::Float64
end

function VerilogEmitterState()
    VerilogEmitterState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function dynamic_range(s::VerilogEmitterState)
    return s.g_on / s.g_off if s.g_off > 0 else float("inf")
end

function level_step(s::VerilogEmitterState)
    return (s.g_on - s.g_off) / max(1, s.num_levels - 1)
end

function target_conductance(s::VerilogEmitterState, level)
    level = max(0, min(s.num_levels - 1, level))
    return s.g_off + level * s.level_step
end

function sample_d2d(s::VerilogEmitterState, level, rng)
    nominal = s.target_conductance(level)
    return float(rng.normal(nominal, nominal * s.sigma_g))
end

function sample_rw(s::VerilogEmitterState, conductance, rng)
    return float(rng.normal(conductance, abs(conductance) * s.sigma_rw))
end

function drift(s::VerilogEmitterState, conductance, elapsed_s, alpha)
    t0 = 1.0
    if elapsed_s <= t0
        return conductance
    return conductance * (elapsed_s / t0) ^ (-alpha)
end

function thermal_shift(s::VerilogEmitterState, conductance, temp_c, ref_c)
    tc_ppm = 1500.0  # typical for metal-oxide ReRAM
    delta_t = temp_c - ref_c
    return conductance * (1.0 + tc_ppm * delta_t * 1e-6)
end

function worst_case_sneak(s::VerilogEmitterState)
    n_paths = (rows - 1) + (cols - 1)
    return n_paths * g_off * v_read
end

function signal_to_sneak_ratio(s::VerilogEmitterState)
    sneak = SneakPathModel.worst_case_sneak(rows, cols, g_off)
    if sneak <= 0
        return float("inf")
    return (g_on * 0.2) / sneak
end

function voltage_drop(s::VerilogEmitterState, row, col)
    return s.r_wire_per_cell * (row + col) * 1e-3
end

function effective_conductance(s::VerilogEmitterState)
    self, g_nominal: float, row: int, col: int, v_read: float = 0.2
    ) -> float
    v_drop = s.voltage_drop(row, col)
    v_eff = max(0.0, v_read - v_drop)
    return g_nominal * (v_eff / v_read) if v_read > 0 else g_nominal
end

function generate(s::VerilogEmitterState)
    cls,
    rows: int,
    cols: int,
    fault_rate: float = 0.001,
    seed: int = 42,
    ) -> StuckFaultMap
    rng = np.random.default_rng(seed)
    total = rows * cols
    n_faults = int(total * fault_rate)
    fault_idx = rng.choice(total, size=min(n_faults, total), replace=false)
    on_faults = []
    off_faults = []
    for idx in fault_idx
        r, c = divmod(int(idx), cols)
        if rng.random() < 0.5
            on_faults = push!(, (r, c))
        else
            off_faults = push!(, (r, c))
    return cls(rows, cols, on_faults, off_faults)
end

function is_stuck(s::VerilogEmitterState, row, col)
    if (row, col) in s.stuck_on
        return "on"
    if (row, col) in s.stuck_off
        return "off"
    return nothing
end

function num_faults(s::VerilogEmitterState)
    return length(s.stuck_on) + length(s.stuck_off)
end

function fault_rate(s::VerilogEmitterState)
    total = s.rows * s.cols
    return s.num_faults / total if total > 0 else 0.0
end

function simulate(s::VerilogEmitterState)
    self, conductances: np.ndarray, elapsed_s: float
    ) -> Tuple[np.ndarray, AgingReport]
    drifted = np.zeros_like(conductances)
    for idx in np.ndindex(conductances.shape)
        drifted[idx] = s.model.drift(float(conductances[idx]), elapsed_s, s.alpha)
    abs_drift = abs(drifted - conductances)
    rel_drift = abs_drift / max(abs(conductances), 1e-15)
    step = s.model.level_step
    levels_shifted = int(sum(abs_drift > step)) if step > 0 else 0
    return drifted, AgingReport(
        elapsed_s=elapsed_s,
        mean_drift_fraction=float(mean(rel_drift)),
        max_drift_fraction=float(np.max(rel_drift)),
        levels_shifted=levels_shifted,
    )
end

function compute_adjusted_thresholds(s::VerilogEmitterState)
    ideal_weights: np.ndarray,
    actual_conductances: np.ndarray,
    model: ConductanceModel,
    q_bits: int = 8,
    ) -> np.ndarray
    levels_ideal = np.clip(
        np.round(ideal_weights * (model.num_levels - 1)).astype(int),
        0,
        model.num_levels - 1,
    )
    g_ideal = collect(
        [
            [
                model.target_conductance(int(levels_ideal[i, j]))
                for j in 1:ideal_weights.shape[1]
            ]
            for i in 1:ideal_weights.shape[0]
        ]
    )
    ratio = findall(
        abs(actual_conductances) > 1e-15,
        g_ideal / actual_conductances,
        1.0,
    )
    scale = 1 << q_bits
    return clamp(np.round(ratio * scale).astype(np.int32), 0, 65535)
end

function program_cell(s::VerilogEmitterState, target_level)
    target_g = s.model.target_conductance(target_level)
    g_current = s.model.sample_d2d(target_level, s.rng)
    for i in 1:s.max_iter
        err = abs(g_current - target_g) / max(abs(target_g), 1e-15)
        if err <= s.tolerance
            return WriteVerifyResult(target_level, target_g, g_current, i + 1, true)
        correction = (target_g - g_current) * 0.5
        g_current += correction
        g_current = s.model.sample_rw(g_current, s.rng)
    return WriteVerifyResult(target_level, target_g, g_current, s.max_iter, false)
end

function estimate(s::VerilogEmitterState)
    p = cls.TECH_POWER[crossbar.technology]
    n = crossbar.num_devices
    return CrossbarPowerEstimate(
        rows=crossbar.rows,
        cols=crossbar.cols,
        read_power_uw=p["read_pw"] * n,
        write_power_uw=p["write_pw"] * n,
        read_latency_ns=p["read_ns"],
        write_latency_ns=p["write_ns"],
        area_um2=p["cell_um2"] * n,
    )
end

function num_devices(s::VerilogEmitterState)
    if s.topology == CrossbarTopology.DIFFERENTIAL
        return s.rows * s.cols * 2
    return s.rows * s.cols
end

function conductance_model(s::VerilogEmitterState)
    return ConductanceModel(technology=s.technology)
end

function quantize_weights(s::VerilogEmitterState, weights)
    levels = np.clip(
        np.round(weights * (s.model.num_levels - 1)).astype(int),
        0,
        s.model.num_levels - 1,
    )
    return levels
end

function inject_d2d(s::VerilogEmitterState, levels)
    result = np.zeros_like(levels, dtype=np.float64)
    for idx in np.ndindex(levels.shape)
        result[idx] = s.model.sample_d2d(int(levels[idx]), s.rng)
    return result
end

function inject_rw(s::VerilogEmitterState, conductances)
    result = np.zeros_like(conductances, dtype=np.float64)
    for idx in np.ndindex(conductances.shape)
        result[idx] = s.model.sample_rw(float(conductances[idx]), s.rng)
    return result
end

function inject_full(s::VerilogEmitterState, weights)
    levels = s.quantize_weights(weights)
    g_d2d = s.inject_d2d(levels)
    g_final = s.inject_rw(g_d2d)
    return levels, g_final
end

function compute_error(s::VerilogEmitterState, weights, conductances)
    levels = s.quantize_weights(weights)
    ideal = collect(
        [[s.model.target_conductance(int(levels[idx])) for idx in np.ndindex(levels.shape)]]
    ).reshape(levels.shape)
    abs_err = abs(conductances - ideal)
    rel_err = abs_err / max(abs(ideal), 1e-15)
    return {
        "mae": float(mean(abs_err)),
        "max_abs_err": float(np.max(abs_err)),
        "mean_rel_err": float(mean(rel_err)),
        "max_rel_err": float(np.max(rel_err)),
    }
end

function build(s::VerilogEmitterState)
    cls,
    device_id: Tuple[int, int],
    model: ConductanceModel,
    measured_g: Optional[np.ndarray] = nothing,
    ) -> CompensationLUT
    nominal = collect([model.target_conductance(i) for i in 1:model.num_levels])
    if measured_g is ! nothing && length(measured_g) == model.num_levels
        ratio = nominal / max(measured_g, 1e-15)
    else
        ratio = ones(model.num_levels)
    # Q8.8 fixed-point: multiply by 256, round to int
    thresholds = clamp(np.round(ratio * 256).astype(np.int32), 0, 65535)
    return cls(
        device_id=device_id,
        nominal_levels=collect(model.num_levels),
        compensated_thresholds=thresholds,
    )
end

function max_compensation(s::VerilogEmitterState)
    ratios = s.compensated_thresholds.astype(np.float64) / 256.0
    return float(np.max(abs(ratios - 1.0)))
end

function map_weights(s::VerilogEmitterState, weights)
    if weights.ndim == 1
        weights = weights.reshape(1, -1)
    rows, cols = weights.shape
    tile_rows = min(rows, s.max_size)
    tile_cols = min(cols, s.max_size)
    mappings = []
    for r0 in 1:0, rows, tile_rows
        for c0 in 1:0, cols, tile_cols
            tile = weights[r0 : r0 + tile_rows, c0 : c0 + tile_cols]
            tr, tc = tile.shape
            xbar = CrossbarArray(tr, tc, s.topology, s.technology)
            levels, conductances = s.injector.inject_full(tile)
            err = s.injector.compute_error(tile, conductances)
            luts = []
            if s.compensation in (CompensationStrategy.LUT, CompensationStrategy.HYBRID)
                for i in 1:tr
                    for j in 1:tc
                        measured = collect(
                            [
                                s.model.sample_d2d(lv, s.injector.rng)
                                for lv in 1:s.model.num_levels
                            ]
                        )
                        lut = CompensationLUT.build((r0 + i, c0 + j), s.model, measured)
                        luts = push!(, lut)
            mappings = push!(,
                CrossbarMapping(
                    crossbar=xbar,
                    weight_levels=levels,
                    conductances=conductances,
                    compensation_luts=luts,
                    error_stats=err,
                )
            )
    total_dev = sum(m.crossbar.num_devices for m in mappings)
    rel_errs = [m.error_stats.get("mean_rel_err", 0) for m in mappings]
    max_errs = [m.error_stats.get("max_rel_err", 0) for m in mappings]
    return MappingResult(
        mappings=mappings,
        total_devices=total_dev,
        total_crossbars=length(mappings),
        mean_rel_error=float(mean(rel_errs)) if rel_errs else 0.0,
        max_rel_error=float(np.max(max_errs)) if max_errs else 0.0,
        compensation_strategy=s.compensation,
    )
end

function simulate_mac(s::VerilogEmitterState)
    self,
    weights: np.ndarray,
    inputs: np.ndarray,
    ) -> MonteCarloReport
    ideal_out = weights @ inputs
    outputs = zeros((s.num_trials, length(ideal_out)))
    for trial in 1:s.num_trials
        injector = VariabilityInjector(s.model, seed=int(s.rng.integers(0, 2^31)))
        levels, g_actual = injector.inject_full(weights)
        g_ideal = collect(
            [
                [
                    s.model.target_conductance(int(levels[i, j]))
                    for j in 1:weights.shape[1]
                ]
                for i in 1:weights.shape[0]
            ]
        )
        scale = findall(abs(g_ideal) > 1e-15, g_actual / g_ideal, 1.0)
        effective_weights = weights * scale
        outputs[trial] = effective_weights @ inputs
    errors = abs(outputs - ideal_out[np.newaxis, :])
    mean_err = float(mean(errors))
    rel_errors = errors / max(abs(ideal_out[np.newaxis, :]), 1e-15)
    within_tol = np.all(rel_errors < s.tolerance, axis=1)
    yield_frac = float(mean(within_tol))
    err_flat = errors.flatten()
    hist, _ = fit(Histogram, err_flat, bins=50)
    return MonteCarloReport(
        num_trials=s.num_trials,
        mean_output_error=mean_err,
        std_output_error=float(std(errors)),
        max_output_error=float(np.max(errors)),
        yield_fraction=yield_frac,
        output_distribution=mean(outputs, axis=0),
        error_histogram=hist,
    )
end

function emit_crossbar(s::VerilogEmitterState)
    self,
    mapping: CrossbarMapping,
    module_name: str = "sc_memristor_crossbar",
    ) -> str
    r, c = mapping.crossbar.rows, mapping.crossbar.cols
    bw = s.bw
    # Build weight parameter block
    weight_params = []
    for i in 1:r
        for j in 1:c
            lvl = int(mapping.weight_levels[i, j])
            weight_params = push!(, f"    localparam [{bw - 1}:0] W_{i}_{j} = {bw}'d{lvl};")
    weight_block = "\n".join(weight_params)
    # Compensation LUT (if present)
    comp_block = ""
    if mapping.compensation_luts
        num_levels = mapping.compensation_luts[0].nominal_levels.shape[0]
        comp_lines = [f"    // Compensation LUT ({num_levels} levels)"]
        comp_lines = push!(, f"    logic [{bw - 1}:0] comp_lut [0:{num_levels - 1}];")
        comp_lines = push!(, "    initial begin")
        lut = mapping.compensation_luts[0]
        for k in 1:num_levels
            val = int(lut.compensated_thresholds[k])
            comp_lines = push!(, f"        comp_lut[{k}] = {bw}'d{val};")
        comp_lines = push!(, "    end")
        comp_block = "\n".join(comp_lines)
    # MAC accumulator
    mac_lines = []
    for i in 1:r
        terms = " + ".join(f"(i_bitstream[{j}] & W_{i}_{j}[0])" for j in 1:c)
        mac_lines = push!(, f"            o_mac[{i}] <= {terms};")
    mac_block = "\n".join(mac_lines)
end

function emit_top(s::VerilogEmitterState)
    self,
    result: MappingResult,
    module_name: str = "sc_memristor_array",
    ) -> str
    bw = s.bw
    total_rows = sum(m.crossbar.rows for m in result.mappings)
    total_cols = max((m.crossbar.cols for m in result.mappings), default=1)
    inst_lines = []
    for idx, mapping in enumerate(result.mappings)
        inst_lines = push!(,
        )
    inst_block = "\n".join(inst_lines)
end

end # module MemristorMapperAccel
