# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference-trace neuron validation contracts

"""Production contracts for the neuron reference-trace validation harness."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from sc_neurocore.neurons.reference_traces import (
    ReferenceTraceSpec,
    list_reference_trace_specs,
    load_reference_trace_spec,
    simulate_reference_trace,
    validate_all_reference_traces,
    validate_reference_trace,
    validate_reference_trace_spec,
)
from sc_neurocore.neurons.universal_dsl import list_bundled_schemas

_STOCHASTIC_SCHEMA_NAMES = frozenset({"escape_rate", "poisson"})
_DETERMINISTIC_SCHEMA_TRACES = {
    "adex": "adex_resting_adaptation_doi",
    "connor_stevens": "connor_stevens_resting_gate_doi",
    "exp_if": "exp_if_resting_exponential_doi",
    "fitzhugh_nagumo": "fitzhugh_nagumo_driven_oscillation_doi",
    "glif": "glif_constant_current_threshold_adaptation",
    "hindmarsh_rose": "hindmarsh_rose_short_bursting_prefix",
    "hodgkin_huxley": "hodgkin_huxley_resting_gate_doi",
    "izhikevich": "izhikevich_regular_spiking_doi",
    "lapicque": "lapicque_constant_current_closed_form",
    "lif": "lif_constant_current_closed_form",
    "morris_lecar": "morris_lecar_depolarizing_current_doi",
    "perfect_integrator": "perfect_integrator_constant_current_sawtooth",
    "quadratic_if": "quadratic_if_zero_current_analytic",
    "resonate_fire": "resonate_fire_subthreshold_resonance_doi",
    "rulkov_map": "rulkov_map_short_window_boundary",
    "theta": "theta_constant_current_phase_analytic",
    "wang_buzsaki": "wang_buzsaki_resting_interneuron_doi",
}


def _closed_form_features(
    *,
    initial: float,
    steady: float,
    tau: float,
    dt: float,
    steps: int,
) -> dict[str, float]:
    values = [
        steady + (initial - steady) * math.exp(-(step * dt) / tau) for step in range(1, steps + 1)
    ]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.v": values[-1],
        "min.v": min(values),
        "max.v": max(values),
        "mean.v": math.fsum(values) / len(values),
    }


def _quadratic_if_zero_current_features(*, dt: float, steps: int) -> dict[str, float]:
    values = [-1.0 / (1.0 + step * dt) for step in range(1, steps + 1)]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.v": values[-1],
        "min.v": min(values),
        "max.v": max(values),
        "mean.v": math.fsum(values) / len(values),
    }


def _perfect_integrator_sawtooth_features(
    *,
    current: float,
    dt: float,
    steps: int,
    c_m: float = 1.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
) -> dict[str, float]:
    """Return exact post-reset features for constant-current perfect integration."""
    values: list[float] = []
    spikes: list[int] = []
    voltage = v_reset
    increment = current * dt / c_m
    for _ in range(steps):
        voltage += increment
        if voltage >= v_threshold:
            spikes.append(1)
            voltage = v_reset
        else:
            spikes.append(0)
        values.append(voltage)

    return {
        "spike_count": float(math.fsum(spikes)),
        "first_spike_step": float(
            next((index for index, spike in enumerate(spikes, start=1) if spike), -1)
        ),
        "final.v": values[-1],
        "min.v": min(values),
        "max.v": max(values),
        "mean.v": math.fsum(values) / len(values),
    }


def _theta_constant_current_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return continuous theta-neuron phase features for constant positive current."""
    if current <= 0.0:
        msg = "theta analytic helper requires positive current"
        raise ValueError(msg)
    root_current = math.sqrt(current)
    values = [
        2.0 * math.atan(root_current * math.tan(root_current * step * dt))
        for step in range(1, steps + 1)
    ]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.theta": values[-1],
        "min.theta": min(values),
        "max.theta": max(values),
        "mean.theta": math.fsum(values) / len(values),
    }


def _resonate_fire_linear_euler_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return exact Euler features for the linear resonate-and-fire schema."""
    omega = 0.5
    damping = -0.1
    threshold = 1.0
    x = 0.0
    y = 0.0
    x_values: list[float] = []
    y_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dx = damping * x - omega * y + current
        dy = omega * x + damping * y
        x_next = x + dt * dx
        y_next = y + dt * dy
        if x_next > threshold:
            spikes.append(1)
            x = 0.0
            y = 0.0
        else:
            spikes.append(0)
            x = x_next
            y = y_next
        x_values.append(x)
        y_values.append(y)

    return {
        "spike_count": float(math.fsum(spikes)),
        "first_spike_step": float(
            next((index for index, spike in enumerate(spikes, start=1) if spike), -1)
        ),
        "final.x": x_values[-1],
        "min.x": min(x_values),
        "max.x": max(x_values),
        "mean.x": math.fsum(x_values) / len(x_values),
        "final.y": y_values[-1],
        "min.y": min(y_values),
        "max.y": max(y_values),
        "mean.y": math.fsum(y_values) / len(y_values),
    }


def test_seeded_corpus_has_analytic_schema_entries() -> None:
    """The seed corpus must expose deterministic analytic schema references."""
    names = list_reference_trace_specs()

    assert names == tuple(sorted(names))
    assert set(_DETERMINISTIC_SCHEMA_TRACES.values()) <= set(names)

    spec = load_reference_trace_spec("lif_constant_current_closed_form")
    assert isinstance(spec, ReferenceTraceSpec)
    assert spec.schema_name == "lif"
    assert spec.provenance.kind == "analytic_closed_form"
    assert spec.protocol.state_variables == ("v",)
    assert spec.protocol.inputs["I"] == 1.0


def test_reference_trace_corpus_covers_every_deterministic_bundled_schema() -> None:
    """Every deterministic bundled schema must have one committed trace."""
    deterministic_schemas = set(list_bundled_schemas()) - _STOCHASTIC_SCHEMA_NAMES

    assert set(_DETERMINISTIC_SCHEMA_TRACES) == deterministic_schemas
    for schema_name, trace_name in _DETERMINISTIC_SCHEMA_TRACES.items():
        spec = load_reference_trace_spec(trace_name)
        assert spec.schema_name == schema_name
        assert spec.runner == "universal_dsl"
        assert spec.provenance.source.endswith(f"/{schema_name}.toml")
        assert spec.provenance.citation is not None
        assert spec.provenance.citation
        if "doi" in trace_name:
            assert spec.provenance.citation.startswith("doi:")


def test_lif_seed_features_match_independent_closed_form_solution() -> None:
    """Committed LIF features must match the closed-form RC solution, not the runner."""
    spec = load_reference_trace_spec("lif_constant_current_closed_form")

    expected = _closed_form_features(
        initial=-65.0,
        steady=-55.0,
        tau=10.0,
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_quadratic_if_trace_features_match_independent_analytic_solution() -> None:
    """Committed QIF features must match the analytic zero-current Riccati flow."""
    spec = load_reference_trace_spec("quadratic_if_zero_current_analytic")

    expected = _quadratic_if_zero_current_features(
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "quadratic_if"
    assert spec.provenance.citation == "doi:10.1152/jn.2000.83.2.808"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_perfect_integrator_trace_features_match_independent_sawtooth_solution() -> None:
    """Committed perfect-integrator features must match the exact reset sawtooth."""
    spec = load_reference_trace_spec("perfect_integrator_constant_current_sawtooth")

    expected = _perfect_integrator_sawtooth_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "perfect_integrator"
    assert spec.provenance.kind == "analytic_sawtooth"
    assert spec.provenance.citation == "doi:10.1017/CBO9781107447615"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_theta_trace_features_match_independent_phase_solution() -> None:
    """Committed theta features must match the tangent half-angle phase solution."""
    spec = load_reference_trace_spec("theta_constant_current_phase_analytic")

    expected = _theta_constant_current_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "theta"
    assert spec.provenance.kind == "analytic_closed_form"
    assert spec.provenance.citation == "doi:10.1137/0146017"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_resonate_fire_trace_features_match_independent_linear_euler_solution() -> None:
    """Committed resonate-fire features must match its linear Euler recurrence."""
    spec = load_reference_trace_spec("resonate_fire_subthreshold_resonance_doi")

    expected = _resonate_fire_linear_euler_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "resonate_fire"
    assert spec.provenance.kind == "analytic_linear_euler_reference"
    assert spec.provenance.citation == "doi:10.1162/089976601300014538"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_simulation_exercises_universal_schema_runner() -> None:
    """The harness must execute the committed schema through UniversalNeuron."""
    spec = load_reference_trace_spec("lapicque_constant_current_closed_form")

    simulation = simulate_reference_trace(spec)

    assert simulation.name == spec.name
    assert simulation.steps == spec.protocol.steps
    assert tuple(simulation.trace) == ("v",)
    assert len(simulation.trace["v"]) == spec.protocol.steps
    assert simulation.trace["v"][0] > 0.0
    assert simulation.spikes == tuple(0 for _ in range(spec.protocol.steps))
    assert simulation.features["spike_count"] == 0.0
    assert simulation.features["first_spike_step"] == -1.0
    assert simulation.features["max.v"] == max(simulation.trace["v"])


def test_reference_trace_validation_accepts_seeded_corpus() -> None:
    """All committed seed references must pass their own tolerance contracts."""
    reports = validate_all_reference_traces()

    assert {report.name for report in reports} == set(list_reference_trace_specs())
    assert all(report.passed for report in reports)
    assert all(report.mismatches == () for report in reports)


def test_name_based_simulation_and_validation_paths_are_public() -> None:
    """Name-based public helpers must route through the committed corpus."""
    simulation = simulate_reference_trace("lif_constant_current_closed_form")
    report = validate_reference_trace("lif_constant_current_closed_form")

    assert simulation.name == "lif_constant_current_closed_form"
    assert report.name == simulation.name
    assert report.passed


def test_unknown_reference_trace_name_fails_closed() -> None:
    """Unknown corpus identifiers must not silently fall back to another trace."""
    with pytest.raises(ValueError, match="unknown reference trace"):
        load_reference_trace_spec("not_a_committed_reference")


def test_validation_reports_feature_drift() -> None:
    """A drifted expected feature must fail closed with the feature name."""
    spec = load_reference_trace_spec("lif_constant_current_closed_form")
    drifted_features = dict(spec.expected_features)
    drifted_features["final.v"] += 0.25
    drifted = replace(spec, expected_features=drifted_features)

    report = validate_reference_trace_spec(drifted)

    assert not report.passed
    assert [mismatch.feature for mismatch in report.mismatches] == ["final.v"]


def test_simulation_rejects_unsupported_in_memory_runner() -> None:
    """In-memory specs cannot select runners outside the v1 production surface."""
    spec = load_reference_trace_spec("lif_constant_current_closed_form")
    unsupported = replace(spec, runner="python_loop")

    with pytest.raises(ValueError, match="unsupported reference-trace runner"):
        simulate_reference_trace(unsupported)
