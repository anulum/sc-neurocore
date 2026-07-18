# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model browser for Studio runtime catalogue entries

"""Model catalogue, descriptor, and simulation helpers for Studio."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from pathlib import Path
from typing import Any

try:
    from sc_neurocore_engine.studio import get_batch_simulate
except ImportError:

    def get_batch_simulate() -> object:
        """Return the optional Rust batch simulator or raise when unavailable."""
        raise ImportError("Studio Rust batch simulator unavailable")


from sc_neurocore.neurons.descriptor_tiers import completeness_tiers, is_perfect
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_descriptor import (
    ModelDescriptor,
    descriptor_completeness_tier,
)


def _evidence_kind(tier: int) -> str:
    """Map a completeness tier to the SCPN-Studio evidence modality."""
    if tier >= 3:
        return "measured"
    if tier == 2:
        return "curated"
    return ""


from sc_neurocore.neurons.models import _CLASS_TO_MODULE

# State variable names that change during .step() — common across models
_KNOWN_STATE_VARS = {
    "v",
    "m",
    "h",
    "n",
    "w",
    "u",
    "g_e",
    "g_i",
    "s",
    "r",
    "q",
    "ca",
    "ca_i",
    "ca_concentration",
    "a",
    "b",
    "z",
    "x",
    "y",
    "phase",
    "amplitude",
    "trace",
    "s_trace",
    "refractory_timer",
    "n_k",
    "h_na",
    "m_na",
    "m_t",
    "h_t",
    "m_a",
    "h_a",
    "m_kd",
    "m_h",
    "m_ca",
    "h_ca",
    "m_nap",
    "h_nap",
}


_class_cache: dict[str, type] = {}


def _load_class(name: str) -> type:
    if name in _class_cache:
        return _class_cache[name]
    module_name = _CLASS_TO_MODULE[name]
    mod = importlib.import_module(f"sc_neurocore.neurons.models.{module_name}")
    cls: type = getattr(mod, name)
    _class_cache[name] = cls
    return cls


def _model_field_specs(cls: type) -> list[tuple[str, float]]:
    """Return ``(name, numeric-default)`` specs for a model class.

    Works for dataclass models (declared fields) and plain classes (the numeric
    keyword parameters of ``__init__``) so the catalogue can browse any
    registered model, not only dataclasses. Non-numeric parameters (identifiers,
    pools, flags) are skipped for plain classes; missing or non-numeric dataclass
    defaults are reported as ``0.0`` to preserve the historical contract.
    """
    if dataclasses.is_dataclass(cls):
        specs: list[tuple[str, float]] = []
        for f in dataclasses.fields(cls):
            default = f.default if f.default is not dataclasses.MISSING else 0.0
            value = (
                float(default)
                if isinstance(default, (int, float)) and not isinstance(default, bool)
                else 0.0
            )
            specs.append((f.name, value))
        return specs
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return []
    plain: list[tuple[str, float]] = []
    for pname, param in signature.parameters.items():
        if pname == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        default = param.default
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            continue
        plain.append((pname, float(default)))
    return plain


def _extract_dt(cls: type) -> float:
    """Return the model's default timestep, or ``0.1`` when undeclared."""
    for name, default in _model_field_specs(cls):
        if name == "dt":
            return default
    return 0.1


def _classify_fields(cls: type) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split model fields into state variables and parameters."""
    state_vars: list[dict[str, Any]] = []
    params: list[dict[str, Any]] = []
    for name, default in _model_field_specs(cls):
        if name == "dt":
            continue
        entry = {"name": name, "default": default}
        if name in _KNOWN_STATE_VARS or name.startswith("v") and len(name) <= 2:
            state_vars.append(entry)
        elif name.startswith(
            ("v_", "e_", "g_", "tau_", "c_", "sigma", "alpha", "beta")
        ) or name.endswith(("_threshold", "_reset", "_rest", "_rev", "_max", "_min")):
            params.append(entry)
        elif name in _KNOWN_STATE_VARS:
            state_vars.append(entry)
        else:
            params.append(entry)
    if not state_vars:
        state_vars.append({"name": "v", "default": -65.0})
    return state_vars, params


_CATEGORY_RULES = [
    (
        "Conductance",
        [
            "HodgkinHuxley",
            "ConnorStevens",
            "WangBuzsaki",
            "TraubMiles",
            "PinskyRinzel",
            "MainenSejnowski",
            "BoothRinzel",
            "HayL5",
            "COBA",
            "TwoCompartment",
            "ReducedTraub",
        ],
    ),
    (
        "Integrate-and-Fire",
        [
            "LIF",
            "IF",
            "QIF",
            "EIF",
            "AdEx",
            "CLIF",
            "Adaptive",
            "GIF",
            "GLIF",
            "Mihalas",
            "Brette",
            "Integer",
        ],
    ),
    (
        "Oscillator",
        [
            "FitzHugh",
            "MorrisLecar",
            "Hindmarsh",
            "VanDerPol",
            "Theta",
            "Selkov",
            "Oregonator",
            "Lotka",
        ],
    ),
    ("Bursting", ["Chay", "Izhikevich", "Bertram", "Butera", "Rulkov", "Map"]),
    ("Hardware", ["Loihi", "SpiNNaker", "Akida", "BrainScale", "TrueNorth", "DPI", "Xylo"]),
    (
        "Network/Population",
        [
            "WilsonCowan",
            "WongWang",
            "JansenRit",
            "Wendling",
            "Ermentrout",
            "Amari",
            "Compte",
            "Larter",
        ],
    ),
    (
        "Statistical",
        [
            "Poisson",
            "Gamma",
            "GLM",
            "SpikeResponse",
            "GalvesLocherbach",
            "McCullochPitts",
            "Renewal",
        ],
    ),
    ("AI-Optimized", ["Attention", "Compositional", "CFC", "Arcane"]),
]


def _categorize(name: str) -> str:
    for category, keywords in _CATEGORY_RULES:
        if any(kw in name for kw in keywords):
            return category
    return "Other"


_models_cache: list[dict[str, Any]] | None = None


class RustStudioBackendUnavailable(ImportError):
    """Raised when the Studio Rust batch-simulation path is unavailable."""


class RustStudioBackendError(RuntimeError):
    """Raised when the Studio Rust batch-simulation path fails at runtime."""


class ModelMetadataError(RuntimeError):
    """Raised when Studio model metadata loading fails for a known model."""


def _provenance_summary(descriptor: ModelDescriptor) -> dict[str, Any] | None:
    """Return a path-free provenance summary, or ``None`` when uncited."""
    prov = descriptor.provenance
    if not (prov.authors or prov.year or prov.doi):
        return None
    return {
        "authors": list(prov.authors),
        "year": prov.year,
        "doi": prov.doi,
        "paper_title": prov.paper_title,
        "url": prov.url,
        "citeable": prov.is_citeable,
    }


def _descriptor_summary(descriptor: ModelDescriptor) -> dict[str, Any]:
    """Build a catalogue list entry from a declared descriptor."""
    tier = descriptor_completeness_tier(descriptor)
    tiers = completeness_tiers(descriptor)
    return {
        "name": descriptor.class_name,
        "module": descriptor.module,
        "tier": tier,
        "evidence_kind": _evidence_kind(tier),
        "science_tier": tiers.science,
        "science_label": tiers.science_label,
        "silicon_tier": tiers.silicon,
        "silicon_label": tiers.silicon_label,
        # ``category`` carries the family display name so existing clients group
        # by the curated family; the fine slug is exposed separately.
        "category": descriptor.family,
        "category_slug": descriptor.category,
        "category_source": "declared",
        "family": descriptor.family,
        "maturity": descriptor.maturity,
        "biophysical_detail": descriptor.biophysical_detail,
        "n_state_vars": len(descriptor.state),
        "n_params": len(descriptor.parameters),
        "state_var_names": [s.name for s in descriptor.state],
        "dt": descriptor.dt,
        "description": descriptor.summary,
        "intended_use": list(descriptor.intended_use),
        "hardware_fit": list(descriptor.hardware_fit),
        "behavior_tags": list(descriptor.behavior_tags),
        "provenance": _provenance_summary(descriptor),
    }


def _descriptor_detail(descriptor: ModelDescriptor) -> dict[str, Any]:
    """Build a full catalogue detail view from a declared descriptor."""
    detail = _descriptor_summary(descriptor)
    detail.update(
        {
            "docstring": descriptor.summary,
            "display_name": descriptor.display_name,
            "state_vars": [
                {"name": s.name, "default": s.init, "unit": s.unit, "meaning": s.meaning}
                for s in descriptor.state
            ],
            "params": [
                {
                    "name": p.name,
                    "default": p.default,
                    "unit": p.unit,
                    "range": list(p.value_range) if p.value_range else None,
                    "biological_range": (list(p.biological_range) if p.biological_range else None),
                    "meaning": p.meaning,
                }
                for p in descriptor.parameters
            ],
            "dynamics": dict(descriptor.dynamics),
            "integration_method": descriptor.integration_method,
            "backends": [
                {"name": b.name, "status": b.status, "parity": b.parity}
                for b in descriptor.backends
            ],
            "reproducibility": {
                "reference_config": descriptor.reproducibility.reference_config,
                "golden_trace_sha256": descriptor.reproducibility.golden_trace_sha256,
                "golden_trace_sha256_variants": list(
                    descriptor.reproducibility.golden_trace_sha256_variants
                ),
                "reproducible": descriptor.reproducibility.is_reproducible,
            },
            "readiness": _readiness_detail(descriptor),
            "documentation_slug": descriptor.documentation_slug,
        }
    )
    return detail


def _readiness_detail(descriptor: ModelDescriptor) -> dict[str, Any]:
    """Build the auditable dual-axis readiness view for a declared descriptor.

    Surfaces the science (S0-S5) and silicon (H0-H5) tiers together with the raw
    evidence facets that justify them, so a reviewer can see exactly why a model
    sits where it does — and whether it meets its declared deployability class.
    """
    tiers = completeness_tiers(descriptor)
    return {
        "science_tier": tiers.science,
        "science_label": tiers.science_label,
        "silicon_tier": tiers.silicon,
        "silicon_label": tiers.silicon_label,
        "is_perfect": is_perfect(descriptor),
        "terminal_silicon_tier": descriptor.silicon.target_tier,
        "terminal_reason": descriptor.silicon.terminal_reason,
        "validation": {
            "dynamics_faithful": descriptor.validation.dynamics_faithful,
            "metric": descriptor.validation.metric,
            "operating_point": descriptor.validation.operating_point,
            "tolerance": descriptor.validation.tolerance,
            "evidence": descriptor.validation.evidence,
        },
        "silicon": {
            "compiles": descriptor.silicon.compiles,
            "cosim_validated": descriptor.silicon.cosim_validated,
            "synthesised": descriptor.silicon.synthesised,
            "timing_closed": descriptor.silicon.timing_closed,
            "formally_equivalent": descriptor.silicon.formally_equivalent,
            "ppa_signed": descriptor.silicon.ppa_signed,
            "target_device": descriptor.silicon.target_device,
            "clock_mhz": descriptor.silicon.clock_mhz,
        },
    }


def _introspected_summary(name: str) -> dict[str, Any]:
    """Fallback catalogue entry for a model with no committed descriptor."""
    cls = _load_class(name)
    state_vars, params = _classify_fields(cls)
    return {
        "name": name,
        "module": _CLASS_TO_MODULE[name],
        "tier": 0,
        "evidence_kind": "",
        "science_tier": 0,
        "science_label": "S0",
        "silicon_tier": None,
        "silicon_label": "none",
        "category": _categorize(name),
        "category_slug": "",
        "category_source": "inferred",
        "family": _categorize(name),
        "maturity": "experimental",
        "biophysical_detail": "point",
        "n_state_vars": len(state_vars),
        "n_params": len(params),
        "state_var_names": [s["name"] for s in state_vars],
        "dt": _extract_dt(cls),
        "description": (cls.__doc__ or "").strip().split("\n")[0],
        "intended_use": [],
        "hardware_fit": [],
        "behavior_tags": [],
        "provenance": None,
    }


def list_models() -> list[dict[str, Any]]:
    """Return declared metadata for every registered neuron model.

    Each entry is built from the model's committed descriptor (family, category,
    maturity, provenance, parameter and state counts). Models without a descriptor
    fall back to code introspection with an ``inferred`` category. Results are
    cached after the first call.
    """
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    result = []
    for name in sorted(_CLASS_TO_MODULE.keys()):
        try:
            descriptor = load_descriptor(name)
            if descriptor is not None:
                result.append(_descriptor_summary(descriptor))
            else:
                result.append(_introspected_summary(name))
        except (TypeError, AttributeError, ValueError):
            continue
    _models_cache = result
    return result


def get_model_detail(name: str) -> dict[str, Any] | None:
    """Return the full declared metadata view for a single model."""
    if name not in _CLASS_TO_MODULE:
        return None
    try:
        descriptor = load_descriptor(name)
    except Exception as exc:
        raise ModelMetadataError(f"Failed to load Studio model descriptor for '{name}'") from exc
    if descriptor is not None:
        return _descriptor_detail(descriptor)
    try:
        cls = _load_class(name)
        state_vars, params = _classify_fields(cls)
        dt_val = _extract_dt(cls)
    except Exception as exc:
        raise ModelMetadataError(f"Failed to classify Studio model metadata for '{name}'") from exc
    return {
        **_introspected_summary(name),
        "docstring": (cls.__doc__ or "").strip().split("\n")[0],
        "state_vars": state_vars,
        "params": params,
        "dt": dt_val,
    }


def model_facets() -> dict[str, Any]:
    """Return the catalogue facet taxonomy and counts for discovery UX."""
    from collections import Counter

    models = list_models()
    family_counts: Counter[tuple[str, str]] = Counter()
    maturity_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    science_tier_counts: Counter[str] = Counter()
    silicon_tier_counts: Counter[str] = Counter()
    for model in models:
        family_counts[(str(model["family"]), str(model["category_slug"]))] += 1
        maturity_counts[str(model["maturity"])] += 1
        science_tier_counts[str(model.get("science_label", "S0"))] += 1
        silicon_tier_counts[str(model.get("silicon_label", "none"))] += 1
        for tag in model.get("behavior_tags", []):
            behavior_counts[str(tag)] += 1
    families = [
        {"family": family, "category_slug": slug, "count": count}
        for (family, slug), count in sorted(family_counts.items())
    ]
    # Most-common behaviour first so the discovery UX leads with the richest filters.
    behaviors = [
        {"tag": tag, "count": count}
        for tag, count in sorted(behavior_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "total": len(models),
        "families": families,
        "maturities": dict(sorted(maturity_counts.items())),
        "behaviors": behaviors,
        "science_tiers": dict(sorted(science_tier_counts.items())),
        "silicon_tiers": dict(sorted(silicon_tier_counts.items())),
    }


_DOCS_DIR = Path(__file__).resolve().parents[3] / "docs" / "api" / "models"


def model_documentation(name: str) -> dict[str, Any] | None:
    """Return the rendered reference documentation for a model, or ``None``.

    The per-model reference page lives at ``docs/api/models/<module>.md``; the
    Studio serves its Markdown so the documentation is browsable inline next to
    the live model rather than only in the built docs site.
    """
    if name not in _CLASS_TO_MODULE:
        return None
    module = _CLASS_TO_MODULE[name]
    path = _DOCS_DIR / f"{module}.md"
    if not path.is_file():
        return None
    return {"name": name, "slug": f"models/{module}", "markdown": path.read_text(encoding="utf-8")}


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
