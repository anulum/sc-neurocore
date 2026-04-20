# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for model_zoo/model_zoo

module ModelZooAccel

using Statistics, LinearAlgebra

mutable struct DocGeneratorState
    variables::Float64
    name::Float64
    version::Float64
    author::Float64
    description::Float64
    references::Float64
    parameters::Float64
    state_variables::Float64
    bit_width::Float64
    frac_bits::Float64
end

function DocGeneratorState()
    DocGeneratorState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function copy(s::DocGeneratorState)
    return NeuronState(variables=dict(s.variables))
end

function as_dict(s::DocGeneratorState)
    return dict(s.variables)
end

function meta(s::DocGeneratorState)
    return nothing
end

function default_state(s::DocGeneratorState)
    return nothing
end

function default_params(s::DocGeneratorState)
    return nothing
end

function ode_dynamics(s::DocGeneratorState)
    self,
    state: NeuronState,
    current: float,
    params: Dict[str, float],
    dt: float,
    ) -> NeuronState
    ...
end

function threshold_check(s::DocGeneratorState, state, params, float])
    ...
end

function reset(s::DocGeneratorState, state, params, float])
    ...
end

function simulate(s::DocGeneratorState)
    self,
    current_trace: np.ndarray,
    dt: float = 0.001,
    params: Optional[Dict[str, float]] = nothing,
    ) -> Tuple[np.ndarray, List[int]]
    p = params || s.default_params()
    state = s.default_state()
    voltages = zeros(length(current_trace), dtype=np.float64)
    spikes: List[int] = []
    for i, I_ext in enumerate(current_trace)
        state = s.ode_dynamics(state, float(I_ext), p, dt)
        if s.threshold_check(state, p)
            spikes = push!(, i)
            state = s.reset(state, p)
        voltages[i] = state["V"]
    return voltages, spikes
end

function meta(s::DocGeneratorState)
    return PluginMeta(
        name="LIF",
        version="1.0.0",
        author="Miroslav Šotek",
        description="Leaky Integrate-&&-Fire with exponential decay.",
        references=["Lapicque, J. Physiol. Pathol. Gén. 9, 1907."],
        parameters={
            "tau_m": "Membrane time constant (s)",
            "V_rest": "Resting potential (V)",
            "V_thresh": "Spike threshold (V)",
            "V_reset": "Reset potential (V)",
            "R_m": "Membrane resistance (Ω)",
        },
        state_variables=["V"],
    )
end

function default_state(s::DocGeneratorState)
    return NeuronState({"V": -0.070})
end

function default_params(s::DocGeneratorState)
    return {
        "tau_m": 0.020,
        "V_rest": -0.070,
        "V_thresh": -0.055,
        "V_reset": -0.075,
        "R_m": 1e7,
    }
end

function ode_dynamics(s::DocGeneratorState, state, current, params, dt)
    s = state.copy()
    tau = params["tau_m"]
    V = s["V"]
    dV = (-(V - params["V_rest"]) + params["R_m"] * current) / tau
    s["V"] = V + dV * dt
    return s
end

function threshold_check(s::DocGeneratorState, state, params)
    return state["V"] >= params["V_thresh"]
end

function reset(s::DocGeneratorState, state, params)
    s = state.copy()
    s["V"] = params["V_reset"]
    return s
end

function meta(s::DocGeneratorState)
    return PluginMeta(
        name="Izhikevich",
        version="1.0.0",
        author="Miroslav Šotek",
        description="Izhikevich 2-variable model (regular spiking default).",
        references=["Izhikevich, IEEE Trans. NN 14(6), 2003."],
        parameters={
            "a": "Recovery time scale",
            "b": "Sensitivity of u to V",
            "c": "After-spike reset of V (mV)",
            "d": "After-spike increment of u",
            "V_thresh": "Spike cutoff (mV)",
        },
        state_variables=["V", "u"],
    )
end

function default_state(s::DocGeneratorState)
    return NeuronState({"V": -65.0, "u": -14.0})
end

function default_params(s::DocGeneratorState)
    return {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, "V_thresh": 30.0}
end

function ode_dynamics(s::DocGeneratorState, state, current, params, dt)
    s = state.copy()
    V, u = s["V"], s["u"]
    dt_ms = dt * 1000.0
    dV = 0.04 * V * V + 5.0 * V + 140.0 - u + current
    du = params["a"] * (params["b"] * V - u)
    s["V"] = V + dV * dt_ms
    s["u"] = u + du * dt_ms
    return s
end

function threshold_check(s::DocGeneratorState, state, params)
    return state["V"] >= params["V_thresh"]
end

function reset(s::DocGeneratorState, state, params)
    s = state.copy()
    s["V"] = params["c"]
    s["u"] = s["u"] + params["d"]
    return s
end

function meta(s::DocGeneratorState)
    return PluginMeta(
        name="AdEx",
        version="1.0.0",
        author="Miroslav Šotek",
        description="Adaptive exponential I&F with sub-threshold resonance.",
        references=["Brette & Gerstner, J. Neurophysiology 94(5), 2005."],
        parameters={
            "C": "Capacitance (nF)",
            "gL": "Leak conductance (nS)",
            "EL": "Leak reversal (mV)",
            "VT": "Threshold (mV)",
            "DeltaT": "Slope factor (mV)",
            "tau_w": "Adaptation τ (ms)",
            "a": "Sub-threshold adaptation (nS)",
            "b": "Spike-triggered adaptation (nA)",
            "V_reset": "Reset voltage (mV)",
            "V_peak": "Spike cutoff (mV)",
        },
        state_variables=["V", "w"],
    )
end

function default_state(s::DocGeneratorState)
    return NeuronState({"V": -70.0, "w": 0.0})
end

function default_params(s::DocGeneratorState)
    return {
        "C": 0.281,
        "gL": 0.030,
        "EL": -70.6,
        "VT": -50.4,
        "DeltaT": 2.0,
        "tau_w": 144.0,
        "a": 0.004,
        "b": 0.0805,
        "V_reset": -70.6,
        "V_peak": 20.0,
    }
end

function ode_dynamics(s::DocGeneratorState, state, current, params, dt)
    s = state.copy()
    V, w = s["V"], s["w"]
    dt_ms = dt * 1000.0
    exp_term = params["DeltaT"] * math.exp(
        min((V - params["VT"]) / max(params["DeltaT"], 1e-6), 20.0)
    )
    dV = (-params["gL"] * (V - params["EL"]) + params["gL"] * exp_term - w + current) / params[
        "C"
    ]
    dw = (params["a"] * (V - params["EL"]) - w) / params["tau_w"]
    s["V"] = V + dV * dt_ms
    s["w"] = w + dw * dt_ms
    return s
end

function threshold_check(s::DocGeneratorState, state, params)
    return state["V"] >= params["V_peak"]
end

function reset(s::DocGeneratorState, state, params)
    s = state.copy()
    s["V"] = params["V_reset"]
    s["w"] = s["w"] + params["b"]
    return s
end

function meta(s::DocGeneratorState)
    return PluginMeta(
        name="Hodgkin-Huxley",
        version="1.0.0",
        author="Miroslav Šotek",
        description="Full HH model with Na/K/leak conductances.",
        references=["Hodgkin & Huxley, J. Physiology 117(4), 1952."],
        parameters={
            "C_m": "Membrane capacitance (µF/cm²)",
            "g_Na": "Na max conductance",
            "g_K": "K max conductance",
            "g_L": "Leak conductance",
            "E_Na": "Na reversal",
            "E_K": "K reversal",
            "E_L": "Leak reversal",
            "V_thresh": "Spike detection threshold (mV)",
        },
        state_variables=["V", "m", "h", "n"],
    )
end

function default_state(s::DocGeneratorState)
    return NeuronState({"V": -65.0, "m": 0.05, "h": 0.6, "n": 0.32})
end

function default_params(s::DocGeneratorState)
    return {
        "C_m": 1.0,
        "g_Na": 120.0,
        "g_K": 36.0,
        "g_L": 0.3,
        "E_Na": 50.0,
        "E_K": -77.0,
        "E_L": -54.387,
        "V_thresh": 0.0,
    }
end

function ode_dynamics(s::DocGeneratorState, state, current, params, dt)
    s = state.copy()
    V, m, h, n = s["V"], s["m"], s["h"], s["n"]
    dt_ms = dt * 1000.0
        return math.exp(max(-500.0, min(500.0, x)))
    a_m = (
        0.1 * (V + 40.0) / (1.0 - _safe_exp(-(V + 40.0) / 10.0))
        if abs(V + 40.0) > 1e-6
        else 1.0
    )
    b_m = 4.0 * _safe_exp(-(V + 65.0) / 18.0)
    a_h = 0.07 * _safe_exp(-(V + 65.0) / 20.0)
    b_h = 1.0 / (1.0 + _safe_exp(-(V + 35.0) / 10.0))
    a_n = (
        0.01 * (V + 55.0) / (1.0 - _safe_exp(-(V + 55.0) / 10.0))
        if abs(V + 55.0) > 1e-6
        else 0.1
    )
    b_n = 0.125 * _safe_exp(-(V + 65.0) / 80.0)
    I_Na = params["g_Na"] * m^3 * h * (V - params["E_Na"])
    I_K = params["g_K"] * n^4 * (V - params["E_K"])
    I_L = params["g_L"] * (V - params["E_L"])
    dV = (current - I_Na - I_K - I_L) / params["C_m"]
    s["V"] = V + dV * dt_ms
    s["m"] = m + (a_m * (1 - m) - b_m * m) * dt_ms
    s["h"] = h + (a_h * (1 - h) - b_h * h) * dt_ms
    s["n"] = n + (a_n * (1 - n) - b_n * n) * dt_ms
    s["m"] = max(0.0, min(1.0, s["m"]))
    s["h"] = max(0.0, min(1.0, s["h"]))
    s["n"] = max(0.0, min(1.0, s["n"]))
    return s
end

function threshold_check(s::DocGeneratorState, state, params)
    return state["V"] >= params["V_thresh"]
end

function reset(s::DocGeneratorState, state, params)
    return state.copy()
end

function register(s::DocGeneratorState, plugin)
    name = plugin.meta().name
    s._plugins[name] = plugin
end

function get(s::DocGeneratorState, name)
    return s._plugins.get(name)
end

function list_plugins(s::DocGeneratorState)
    return sorted(s._plugins.keys())
end

function with_builtins(s::DocGeneratorState)
    reg = cls()
    for plugin_cls in (LIFPlugin, IzhikevichPlugin, AdExPlugin, HodgkinHuxleyPlugin)
        reg.register(plugin_cls())
    return reg
end

function generate(s::DocGeneratorState, plugin)
    meta = plugin.meta()
    params = plugin.default_params()
    state_vars = meta.state_variables
    module_name = f"sc_neuron_{meta.name.lower().replace('-', '_')}"
    bw = s.bit_width
    port_lines = [
        "    input  logic clk,",
        "    input  logic rst_n,",
        f"    input  logic signed [{bw - 1}:0] i_current,",
    ]
    for sv in state_vars
        port_lines = push!(, f"    output logic signed [{bw - 1}:0] o_{sv},")
    port_lines = push!(, "    output logic o_spike")
    reg_lines = []
    for sv in state_vars
        reg_lines = push!(, f"    logic signed [{bw - 1}:0] {sv}_reg;")
    reset_lines = []
    default_state = plugin.default_state()
    for sv in state_vars
        fixed_val = s._to_fixed(default_state[sv])
        reset_lines = push!(, f"            {sv}_reg <= {bw}'sd{fixed_val};")
    param_lines = []
    for pname, pval in params.items()
        fixed_val = s._to_fixed(pval)
        safe_name = pname.replace("-", "_")
        param_lines = push!(,
            f"    localparam signed [{bw - 1}:0] {safe_name.upper()} = {bw}'sd{fixed_val};"
        )
    assign_lines = []
    for sv in state_vars
        assign_lines = push!(, f"    assign o_{sv} = {sv}_reg;")
    return header + body
end

function _to_fixed(s::DocGeneratorState, value)
    return int(round(value * (1 << s.frac_bits)))
end

function generate(s::DocGeneratorState, plugin)
    meta = plugin.meta()
    lines = [
        f"# {meta.name}",
        "",
        f"^Version^: {meta.version}  ",
        f"^Author^: {meta.author}  ",
        f"^Description^: {meta.description}",
        "",
    ]
    if meta.references
        lines = push!(, "## References")
        lines = push!(, "")
        for ref in meta.references
            lines = push!(, f"- {ref}")
        lines = push!(, "")
    if meta.parameters
        lines = push!(, "## Parameters")
        lines = push!(, "")
        lines = push!(, "| Name | Description |")
        lines = push!(, "|------|-------------|")
        for pname, pdesc in meta.parameters.items()
            lines = push!(, f"| `{pname}` | {pdesc} |")
        lines = push!(, "")
    default_params = plugin.default_params()
    if default_params
        lines = push!(, "## Default Values")
        lines = push!(, "")
        lines = push!(, "| Parameter | Value |")
        lines = push!(, "|-----------|-------|")
        for pname, pval in default_params.items()
            lines = push!(, f"| `{pname}` | {pval} |")
        lines = push!(, "")
    if meta.state_variables
        lines = push!(, "## State Variables")
        lines = push!(, "")
        for sv in meta.state_variables
            lines = push!(, f"- `{sv}`")
        lines = push!(, "")
    return "\n".join(lines)
end

function generate_index(s::DocGeneratorState, registry)
    lines = [
        "# SC-NeuroCore Model Zoo",
        "",
        "| Model | Version | Description |",
        "|-------|---------|-------------|",
    ]
    for name in registry.list_plugins()
        plugin = registry.get(name)
        if plugin
            m = plugin.meta()
            lines = push!(, f"| {m.name} | {m.version} | {m.description} |")
    lines = push!(, "")
    return "\n".join(lines)
end

end # module ModelZooAccel
