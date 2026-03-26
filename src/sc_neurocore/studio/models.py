# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model browser for Studio (all 118 neuron models)

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

from sc_neurocore.neurons.models import _CLASS_TO_MODULE

# State variable names that change during .step() — common across models
_KNOWN_STATE_VARS = {"v", "m", "h", "n", "w", "u", "g_e", "g_i", "s", "r", "q",
                     "ca", "ca_i", "ca_concentration", "a", "b", "z", "x", "y",
                     "phase", "amplitude", "trace", "s_trace", "refractory_timer",
                     "n_k", "h_na", "m_na", "m_t", "h_t", "m_a", "h_a",
                     "m_kd", "m_h", "m_ca", "h_ca", "m_nap", "h_nap"}


def _load_class(name: str) -> type:
    module_name = _CLASS_TO_MODULE[name]
    mod = importlib.import_module(f"sc_neurocore.neurons.models.{module_name}")
    return getattr(mod, name)


def _classify_fields(cls: type) -> tuple[list[dict], list[dict]]:
    """Split dataclass fields into state variables and parameters."""
    state_vars: list[dict] = []
    params: list[dict] = []
    for f in dataclasses.fields(cls):
        if f.name == "dt":
            continue
        default = f.default if f.default is not dataclasses.MISSING else 0.0
        entry = {"name": f.name, "default": float(default) if isinstance(default, (int, float)) else 0.0}
        if f.name in _KNOWN_STATE_VARS or f.name.startswith("v") and len(f.name) <= 2:
            state_vars.append(entry)
        elif f.name.startswith(("v_", "e_", "g_", "tau_", "c_", "sigma", "alpha", "beta")):
            params.append(entry)
        elif f.name.endswith(("_threshold", "_reset", "_rest", "_rev", "_max", "_min")):
            params.append(entry)
        elif f.name in _KNOWN_STATE_VARS:
            state_vars.append(entry)
        else:
            params.append(entry)
    if not state_vars:
        state_vars.append({"name": "v", "default": -65.0})
    return state_vars, params


def list_models() -> list[dict]:
    """Return metadata for all 118 neuron models."""
    result = []
    for name in sorted(_CLASS_TO_MODULE.keys()):
        try:
            cls = _load_class(name)
            if not dataclasses.is_dataclass(cls):
                continue
            state_vars, params = _classify_fields(cls)
            dt_field = next((f for f in dataclasses.fields(cls) if f.name == "dt"), None)
            dt_val = float(dt_field.default) if dt_field and dt_field.default is not dataclasses.MISSING else 0.1
            result.append({
                "name": name,
                "module": _CLASS_TO_MODULE[name],
                "n_state_vars": len(state_vars),
                "n_params": len(params),
                "state_var_names": [s["name"] for s in state_vars],
                "dt": dt_val,
            })
        except Exception:
            continue
    return result


def get_model_detail(name: str) -> dict | None:
    """Return full metadata for a single model."""
    if name not in _CLASS_TO_MODULE:
        return None
    try:
        cls = _load_class(name)
        if not dataclasses.is_dataclass(cls):
            return None
        state_vars, params = _classify_fields(cls)
        dt_field = next((f for f in dataclasses.fields(cls) if f.name == "dt"), None)
        dt_val = float(dt_field.default) if dt_field and dt_field.default is not dataclasses.MISSING else 0.1
        return {
            "name": name,
            "module": _CLASS_TO_MODULE[name],
            "state_vars": state_vars,
            "params": params,
            "dt": dt_val,
            "docstring": (cls.__doc__ or "").strip().split("\n")[0],
        }
    except Exception:
        return None


def simulate_model(
    name: str,
    param_overrides: dict[str, float] | None = None,
    dt: float | None = None,
    duration: float = 100.0,
    current: float = 10.0,
    protocol: str = "constant",
) -> dict[str, Any]:
    """Simulate a named model and return traces."""
    import numpy as np
    from sc_neurocore.studio.simulation import MAX_PLOT_POINTS, MAX_STEPS, _make_current_trace, _spike_stats

    if name not in _CLASS_TO_MODULE:
        raise ValueError(f"Unknown model: {name}")

    cls = _load_class(name)
    kwargs: dict[str, Any] = {}
    if param_overrides:
        kwargs.update(param_overrides)
    if dt is not None:
        kwargs["dt"] = dt

    neuron = cls(**kwargs)
    actual_dt = getattr(neuron, "dt", 0.1)
    n_steps = min(int(duration / actual_dt), MAX_STEPS)
    if n_steps < 1:
        raise ValueError(f"Duration {duration} with dt {actual_dt} yields < 1 step")

    state_vars, _ = _classify_fields(cls)
    var_names = [s["name"] for s in state_vars]
    traces = {v: np.empty(n_steps) for v in var_names}
    spike_indices: list[int] = []

    I_trace = _make_current_trace(protocol, current, n_steps)

    for t in range(n_steps):
        spike = neuron.step(current=float(I_trace[t]))
        for v in var_names:
            val = getattr(neuron, v, 0.0)
            traces[v][t] = float(val) if isinstance(val, (int, float)) else 0.0
        if spike:
            spike_indices.append(t)

    time = np.arange(n_steps) * actual_dt
    stats = _spike_stats(spike_indices, actual_dt, n_steps)

    if n_steps > MAX_PLOT_POINTS:
        stride = n_steps // MAX_PLOT_POINTS
        time = time[::stride]
        traces = {v: arr[::stride] for v, arr in traces.items()}
        I_trace = I_trace[::stride]

    return {
        "time": time.tolist(),
        "states": {v: arr.tolist() for v, arr in traces.items()},
        "current_trace": I_trace.tolist(),
        "spikes": spike_indices,
        "spike_count": len(spike_indices),
        "stats": stats,
        "dt": actual_dt,
        "n_steps": n_steps,
        "model_name": name,
    }
