# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio model simulation entrypoints

"""Python and optional Rust batch simulation for Studio model runs."""

from __future__ import annotations

import dataclasses
from typing import Any

try:
    from sc_neurocore_engine.studio import get_batch_simulate
except ImportError:

    def get_batch_simulate() -> object:
        """Return the optional Rust batch simulator or raise when unavailable."""
        raise ImportError("Studio Rust batch simulator unavailable")


from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.model_introspection import (
    _classify_fields,
    _load_class,
)


class RustStudioBackendUnavailable(ImportError):
    """Raised when the Studio Rust batch-simulation path is unavailable."""


class RustStudioBackendError(RuntimeError):
    """Raised when the Studio Rust batch-simulation path fails at runtime."""


def _load_rust_batch_simulate() -> Any:
    """Load the Rust batch-simulation bridge entrypoint.

    Import failure means the backend is unavailable; it must not be conflated
    with runtime failure inside an otherwise available backend.
    """
    try:
        return get_batch_simulate()
    except ImportError as exc:
        raise RustStudioBackendUnavailable("Studio Rust batch simulator unavailable") from exc


def _is_rust_unsupported_model_error(exc: Exception) -> bool:
    """Return whether the Rust backend rejected a model as unsupported."""
    return isinstance(exc, ValueError) and "Unsupported model:" in str(exc)


def _detect_step_kwarg(cls: Any) -> str:
    """Figure out what keyword the .step() method uses for current injection."""
    import inspect

    sig = inspect.signature(cls.step)
    params = list(sig.parameters.keys())
    # Skip 'self'
    for candidate in ["current", "I", "input_current", "i_ext", "ext_input"]:
        if candidate in params:
            return candidate
    # Fallback: second param after self (positional)
    if len(params) >= 2:
        return params[1]
    return "current"


def _try_rust_simulate(
    name: str,
    n_steps: int,
    current_trace: Any,
    actual_dt: float,
) -> dict[str, Any] | None:
    """Attempt Rust batch simulation.

    Returns ``None`` only when the backend is unavailable or the model is not
    implemented in Rust. Runtime failures in an available backend are raised so
    the caller does not silently degrade to Python.
    """
    import numpy as np
    from sc_neurocore.studio.simulation import MAX_PLOT_POINTS, _spike_stats

    try:
        py_batch_simulate = _load_rust_batch_simulate()
    except RustStudioBackendUnavailable:
        return None

    current_arr = np.asarray(current_trace, dtype=np.float64)
    try:
        result = py_batch_simulate(name, n_steps, current_arr)
    except Exception as exc:
        if _is_rust_unsupported_model_error(exc):
            return None
        raise RustStudioBackendError(
            f"Studio Rust batch simulation failed for model '{name}'"
        ) from exc

    voltages = np.asarray(result["voltages"])
    spikes = result["spikes"].tolist()
    stats = _spike_stats(spikes, actual_dt, n_steps)

    time = np.arange(n_steps) * actual_dt
    if n_steps > MAX_PLOT_POINTS:
        stride = n_steps // MAX_PLOT_POINTS
        time = time[::stride]
        voltages = voltages[::stride]
        current_trace = current_trace[::stride]

    voltages = np.nan_to_num(voltages, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "time": time.tolist(),
        "states": {"v": voltages.tolist()},
        "current_trace": current_trace.tolist()
        if hasattr(current_trace, "tolist")
        else list(current_trace),
        "spikes": spikes,
        "spike_count": len(spikes),
        "stats": stats,
        "dt": actual_dt,
        "n_steps": n_steps,
        "model_name": name,
    }


def simulate_model(
    name: str,
    param_overrides: dict[str, float] | None = None,
    dt: float | None = None,
    duration: float = 100.0,
    current: float = 10.0,
    protocol: str = "constant",
    frequency_hz: float = 10.0,
    use_fast_path: bool = True,
) -> dict[str, Any]:
    """Simulate a named model. Uses Rust engine when model has default params.

    Set ``use_fast_path=False`` to force the Python reference model and bypass the
    Rust accelerator. The behaviour probe relies on this so its characterisation
    is the canonical model's, independent of whether the Rust extension happens to
    be loaded (the two backends can differ for models with an internal RNG).
    """
    import numpy as np
    from sc_neurocore.studio.simulation import (
        MAX_PLOT_POINTS,
        MAX_STEPS,
        _make_current_trace,
        _spike_stats,
    )

    if name not in _CLASS_TO_MODULE:
        raise ValueError(f"Unknown model: {name}")

    # Rust fast path: default params, no overrides
    has_overrides = param_overrides and any(True for _ in param_overrides.values())
    if use_fast_path and not has_overrides and dt is None:
        cls = _load_class(name)
        actual_dt = 0.1
        if dataclasses.is_dataclass(cls):
            dt_field = next((f for f in dataclasses.fields(cls) if f.name == "dt"), None)
            if dt_field and dt_field.default is not dataclasses.MISSING:
                actual_dt = float(dt_field.default)
        n_steps = min(int(duration / actual_dt), MAX_STEPS)
        if n_steps >= 1:
            I_trace = _make_current_trace(
                protocol, current, n_steps, dt=actual_dt, frequency_hz=frequency_hz
            )
            rust_result = _try_rust_simulate(name, n_steps, I_trace, actual_dt)
            if rust_result is not None:
                return rust_result

    cls = _load_class(name)

    # Build constructor kwargs — only pass fields that actually exist on the dataclass
    valid_fields = {}
    if dataclasses.is_dataclass(cls):
        for f in dataclasses.fields(cls):
            valid_fields[f.name] = f.default if f.default is not dataclasses.MISSING else None
    kwargs: dict[str, Any] = {}
    if param_overrides:
        for k, v in param_overrides.items():
            if k not in valid_fields:
                continue
            default = valid_fields[k]
            # Skip if value matches default (avoids float→int type issues)
            if (
                default is not None
                and isinstance(default, (int, float))
                and abs(v - default) < 1e-12
            ):
                continue
            # Preserve int type for integer-arithmetic models
            if default is not None and isinstance(default, int):
                kwargs[k] = int(round(v))
            else:
                kwargs[k] = v
    if dt is not None and "dt" in valid_fields:
        kwargs["dt"] = dt

    try:
        neuron = cls(**kwargs)
    except (TypeError, OverflowError):
        # Some models need int params (bitshift arithmetic)
        int_kwargs = {
            k: int(v) if isinstance(v, float) and v == int(v) else v for k, v in kwargs.items()
        }
        try:
            neuron = cls(**int_kwargs)
        except (TypeError, OverflowError):
            neuron = cls()

    actual_dt = getattr(neuron, "dt", 0.1)
    n_steps = min(int(duration / actual_dt), MAX_STEPS)
    if n_steps < 1:
        raise ValueError(f"Duration {duration} with dt {actual_dt} yields < 1 step")

    state_vars, _ = _classify_fields(cls)
    var_names = [s["name"] for s in state_vars]
    traces = {v: np.empty(n_steps) for v in var_names}
    spike_indices: list[int] = []

    I_trace = _make_current_trace(
        protocol, current, n_steps, dt=actual_dt, frequency_hz=frequency_hz
    )
    step_kwarg = _detect_step_kwarg(cls)

    # Detect if this is an integer-arithmetic model
    _is_int_model = any(isinstance(valid_fields.get(k), int) for k in valid_fields if k == "v")

    for t in range(n_steps):
        i_val: Any = int(I_trace[t]) if _is_int_model else float(I_trace[t])
        try:
            spike = neuron.step(**{step_kwarg: i_val})
        except TypeError:
            try:
                spike = neuron.step(i_val)
            except TypeError:
                spike = neuron.step(int(i_val))
        except (OverflowError, FloatingPointError):
            spike = 0
        for v in var_names:
            val = getattr(neuron, v, 0.0)
            try:
                traces[v][t] = float(val) if isinstance(val, (int, float)) else 0.0
            except (ValueError, OverflowError):
                traces[v][t] = 0.0
        if spike:
            spike_indices.append(t)

    time = np.arange(n_steps) * actual_dt
    stats = _spike_stats(spike_indices, actual_dt, n_steps)

    if n_steps > MAX_PLOT_POINTS:
        stride = n_steps // MAX_PLOT_POINTS
        time = time[::stride]
        traces = {v: arr[::stride] for v, arr in traces.items()}  # type: ignore[misc]
        I_trace = I_trace[::stride]

    # Replace NaN/Inf with 0 for JSON serialisation
    for v in traces:
        traces[v] = np.nan_to_num(traces[v], nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[assignment]

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
