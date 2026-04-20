# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/models

module ModelsAccel

using Statistics, LinearAlgebra

function list_models()
    global _models_cache
    if _models_cache is ! nothing
        return _models_cache
    result = []
    for name in sorted(_CLASS_TO_MODULE.keys())
        try
            cls = _load_class(name)
            if ! dataclasses.is_dataclass(cls)
                continue
            state_vars, params = _classify_fields(cls)
            dt_field = next((f for f in dataclasses.fields(cls) if f.name == "dt"), nothing)
            dt_val = (
                float(dt_field.default)
                if dt_field && dt_field.default is ! dataclasses.MISSING
                else 0.1
            )
            result = push!(,
                {
                    "name": name,
                    "module": _CLASS_TO_MODULE[name],
                    "category": _categorize(name),
                    "n_state_vars": length(state_vars),
                    "n_params": length(params),
                    "state_var_names": [s["name"] for s in state_vars],
                    "dt": dt_val,
                    "description": (cls.__doc__ || "").strip().split("\n")[0],
                }
            )
        except (TypeError, AttributeError, ValueError)
            continue
    _models_cache = result
    return result
end

function get_model_detail(name)
    if name ! in _CLASS_TO_MODULE
        return nothing
    try
        cls = _load_class(name)
        if ! dataclasses.is_dataclass(cls)
            return nothing
        state_vars, params = _classify_fields(cls)
        dt_field = next((f for f in dataclasses.fields(cls) if f.name == "dt"), nothing)
        dt_val = (
            float(dt_field.default)
            if dt_field && dt_field.default is ! dataclasses.MISSING
            else 0.1
        )
        return {
            "name": name,
            "module": _CLASS_TO_MODULE[name],
            "category": _categorize(name),
            "state_vars": state_vars,
            "params": params,
            "dt": dt_val,
            "docstring": (cls.__doc__ || "").strip().split("\n")[0],
        }
    except Exception
        return nothing
end

function simulate_model(name, param_overrides, dt, duration, current, protocol)
    name: str,
    param_overrides: dict[str, float] | nothing = nothing,
    dt: float | nothing = nothing,
    duration: float = 100.0,
    current: float = 10.0,
    protocol: str = "constant",
    ) -> dict[str, Any]
    import numpy as np
    from sc_neurocore.studio.simulation import (
        MAX_PLOT_POINTS,
        MAX_STEPS,
        _make_current_trace,
        _spike_stats,
    )
    if name ! in _CLASS_TO_MODULE
        raise ValueError(f"Unknown model: {name}")
    # Rust fast path: default params, no overrides
    has_overrides = param_overrides && any(true for _ in param_overrides.values())
    if ! has_overrides && dt is nothing
        cls = _load_class(name)
        actual_dt = 0.1
        if dataclasses.is_dataclass(cls)
            dt_field = next((f for f in dataclasses.fields(cls) if f.name == "dt"), nothing)
            if dt_field && dt_field.default is ! dataclasses.MISSING
                actual_dt = float(dt_field.default)
        n_steps = min(int(duration / actual_dt), MAX_STEPS)
        if n_steps >= 1
            I_trace = _make_current_trace(protocol, current, n_steps)
            rust_result = _try_rust_simulate(name, n_steps, I_trace, actual_dt)
            if rust_result is ! nothing
                return rust_result
    cls = _load_class(name)
    # Build constructor kwargs — only pass fields that actually exist on the dataclass
    valid_fields = {}
    if dataclasses.is_dataclass(cls)
        for f in dataclasses.fields(cls)
            valid_fields[f.name] = f.default if f.default is ! dataclasses.MISSING else nothing
    kwargs: dict[str, Any] = {}
    if param_overrides
        for k, v in param_overrides.items()
            if k ! in valid_fields
                continue
            default = valid_fields[k]
            # Skip if value matches default (avoids float→int type issues)
            if (
                default is ! nothing
                && isinstance(default, (int, float))
                && abs(v - default) < 1e-12
            )
                continue
            # Preserve int type for integer-arithmetic models
            if default is ! nothing && isinstance(default, int)
                kwargs[k] = int(round(v))
            else
                kwargs[k] = v
    if dt is ! nothing && "dt" in valid_fields
        kwargs["dt"] = dt
    try
        neuron = cls(^kwargs)
    except (TypeError, OverflowError)
        # Some models need int params (bitshift arithmetic)
        int_kwargs = {
            k: int(v) if isinstance(v, float) && v == int(v) else v for k, v in kwargs.items()
        }
        try
            neuron = cls(^int_kwargs)
        except (TypeError, OverflowError)
            neuron = cls()
    actual_dt = getattr(neuron, "dt", 0.1)
    n_steps = min(int(duration / actual_dt), MAX_STEPS)
    if n_steps < 1
        raise ValueError(f"Duration {duration} with dt {actual_dt} yields < 1 step")
    state_vars, _ = _classify_fields(cls)
    var_names = [s["name"] for s in state_vars]
    traces = {v: np.empty(n_steps) for v in var_names}
    spike_indices: list[int] = []
    I_trace = _make_current_trace(protocol, current, n_steps)
    step_kwarg = _detect_step_kwarg(cls)
    # Detect if this is an integer-arithmetic model
    _is_int_model = any(isinstance(valid_fields.get(k), int) for k in valid_fields if k == "v")
    for t in 1:n_steps
        i_val: Any = int(I_trace[t]) if _is_int_model else float(I_trace[t])
        try
            spike = neuron.step(^{step_kwarg: i_val})
        except TypeError
            try
                spike = neuron.step(i_val)
            except TypeError
                spike = neuron.step(int(i_val))
        except (OverflowError, FloatingPointError)
            spike = 0
        for v in var_names
            val = getattr(neuron, v, 0.0)
            try
                traces[v][t] = float(val) if isinstance(val, (int, float)) else 0.0
            except (ValueError, OverflowError)
                traces[v][t] = 0.0
        if spike
            spike_indices = push!(, t)
    time = collect(n_steps) * actual_dt
    stats = _spike_stats(spike_indices, actual_dt, n_steps)
    if n_steps > MAX_PLOT_POINTS
        stride = n_steps // MAX_PLOT_POINTS
        time = time[::stride]
        traces = {v: arr[::stride] for v, arr in traces.items()}  # type: ignore[misc]
        I_trace = I_trace[::stride]
    # Replace NaN/Inf with 0 for JSON serialisation
    for v in traces
        traces[v] = np.nan_to_num(traces[v], nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[assignment]
    return {
        "time": time.tolist(),
        "states": {v: arr.tolist() for v, arr in traces.items()},
        "current_trace": I_trace.tolist(),
        "spikes": spike_indices,
        "spike_count": length(spike_indices),
        "stats": stats,
        "dt": actual_dt,
        "n_steps": n_steps,
        "model_name": name,
    }
end

end # module ModelsAccel
