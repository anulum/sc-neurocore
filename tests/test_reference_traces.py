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


def _glif_subthreshold_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the subthreshold GLIF5 recurrence.

    The Allen Institute GLIF5 membrane, adaptive threshold, and two after-spike
    currents are linear, so the schema runner's simultaneous explicit-Euler update
    has an exact independent re-derivation. For a subthreshold constant current the
    threshold is never crossed and both after-spike currents stay quiescent at zero.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v``, ``theta``, ``i_asc1``, and ``i_asc2``
        state variables plus spike-count and first-spike-step features.
    """
    v_rest = -70.0
    v_reset = -70.0
    resistance = 1.0
    tau_m = 10.0
    theta_inf = -50.0
    a_theta = 0.01
    tau_theta = 100.0
    tau_asc1 = 10.0
    tau_asc2 = 200.0
    delta_theta = 2.0
    r_asc1 = 1.0
    r_asc2 = 0.5

    v = v_rest
    theta = theta_inf
    i_asc1 = 0.0
    i_asc2 = 0.0
    recorded: dict[str, list[float]] = {"v": [], "theta": [], "i_asc1": [], "i_asc2": []}
    spikes: list[int] = []
    for _ in range(steps):
        dv = (-(v - v_rest) + resistance * current + i_asc1 + i_asc2) / tau_m
        dtheta = (theta_inf - theta + a_theta * (v - v_rest)) / tau_theta
        di_asc1 = -i_asc1 / tau_asc1
        di_asc2 = -i_asc2 / tau_asc2
        v_next = v + dv * dt
        theta_next = theta + dtheta * dt
        i_asc1_next = i_asc1 + di_asc1 * dt
        i_asc2_next = i_asc2 + di_asc2 * dt
        if v_next > theta_next:
            spikes.append(1)
            v_next = v_reset
            theta_next = theta_next + delta_theta
            i_asc1_next = i_asc1_next + r_asc1
            i_asc2_next = i_asc2_next + r_asc2
        else:
            spikes.append(0)
        v, theta, i_asc1, i_asc2 = v_next, theta_next, i_asc1_next, i_asc2_next
        recorded["v"].append(v)
        recorded["theta"].append(theta)
        recorded["i_asc1"].append(i_asc1)
        recorded["i_asc2"].append(i_asc2)

    features: dict[str, float] = {
        "spike_count": float(math.fsum(spikes)),
        "first_spike_step": float(
            next((index for index, spike in enumerate(spikes, start=1) if spike), -1)
        ),
    }
    for name, values in recorded.items():
        features[f"final.{name}"] = values[-1]
        features[f"min.{name}"] = min(values)
        features[f"max.{name}"] = max(values)
        features[f"mean.{name}"] = math.fsum(values) / len(values)
    return features


def _izhikevich_rs_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the regular-spiking Izhikevich recurrence.

    The Izhikevich (2003) quadratic membrane and linear recovery equations are
    advanced with the same simultaneous explicit-Euler update the schema runner
    applies, and the ``v = c``, ``u = u + d`` reset fires whenever the post-update
    membrane crosses the ``v > 30`` peak. The reference is therefore an independent
    re-derivation of the committed spike-bearing trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``u`` state variables plus
        spike-count and first-spike-step features.
    """
    a = 0.02
    b = 0.2
    c = -65.0
    d = 8.0
    v = -65.0
    u = -14.0
    v_values: list[float] = []
    u_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = 0.04 * v**2 + 5 * v + 140 - u + current
        du = a * (b * v - u)
        v_next = v + dv * dt
        u_next = u + du * dt
        if v_next > 30:
            spikes.append(1)
            v_next = c
            u_next = u_next + d
        else:
            spikes.append(0)
        v, u = v_next, u_next
        v_values.append(v)
        u_values.append(u)

    features: dict[str, float] = {
        "spike_count": float(math.fsum(spikes)),
        "first_spike_step": float(
            next((index for index, spike in enumerate(spikes, start=1) if spike), -1)
        ),
    }
    for name, values in (("v", v_values), ("u", u_values)):
        features[f"final.{name}"] = values[-1]
        features[f"min.{name}"] = min(values)
        features[f"max.{name}"] = max(values)
        features[f"mean.{name}"] = math.fsum(values) / len(values)
    return features


def _fitzhugh_nagumo_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the driven FitzHugh-Nagumo recurrence.

    The FitzHugh (1961) cubic membrane and linear recovery equations are advanced
    with the same simultaneous explicit-Euler update the schema runner applies, and
    the ``v = -1`` reset (recovery ``w`` left unchanged) fires whenever the
    post-update membrane crosses the ``v > 1`` threshold. The reference is an
    independent re-derivation of the committed relaxation-oscillation trace, not a
    copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    a = 0.7
    b = 0.8
    epsilon = 0.08
    v = -1.0
    w = -0.5
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = v - v**3 / 3 - w + current
        dw = epsilon * (v + a - b * w)
        v_next = v + dv * dt
        w_next = w + dw * dt
        if v_next > 1.0:
            spikes.append(1)
            v_next = -1.0
        else:
            spikes.append(0)
        v, w = v_next, w_next
        v_values.append(v)
        w_values.append(w)

    features: dict[str, float] = {
        "spike_count": float(math.fsum(spikes)),
        "first_spike_step": float(
            next((index for index, spike in enumerate(spikes, start=1) if spike), -1)
        ),
    }
    for name, values in (("v", v_values), ("w", w_values)):
        features[f"final.{name}"] = values[-1]
        features[f"min.{name}"] = min(values)
        features[f"max.{name}"] = max(values)
        features[f"mean.{name}"] = math.fsum(values) / len(values)
    return features


def _adex_subthreshold_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the subthreshold AdEx recurrence.

    The Brette-Gerstner (2005) exponential membrane and linear adaptation equations
    are advanced with the same simultaneous explicit-Euler update the schema runner
    applies. For the resting zero-current protocol the ``v > -50`` threshold is never
    reached, so the ``v = v_reset``, ``w = w + b`` reset stays inactive and the
    reference is an independent re-derivation of the committed quiet trajectory.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    v_rest = -65.0
    v_reset = -68.0
    v_rh = -55.0
    delta_t = 2.0
    tau = 20.0
    tau_w = 100.0
    a = 0.5
    b_adapt = 7.0
    capacitance = 200.0
    v = -65.0
    w = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = (-(v - v_rest) + delta_t * math.exp((v - v_rh) / delta_t)) / tau + (
            -w + current
        ) / capacitance
        dw = (a * (v - v_rest) - w) / tau_w
        v_next = v + dv * dt
        w_next = w + dw * dt
        if v_next > -50:
            spikes.append(1)
            v_next = v_reset
            w_next = w_next + b_adapt
        else:
            spikes.append(0)
        v, w = v_next, w_next
        v_values.append(v)
        w_values.append(w)

    features: dict[str, float] = {
        "spike_count": float(math.fsum(spikes)),
        "first_spike_step": float(
            next((index for index, spike in enumerate(spikes, start=1) if spike), -1)
        ),
    }
    for name, values in (("v", v_values), ("w", w_values)):
        features[f"final.{name}"] = values[-1]
        features[f"min.{name}"] = min(values)
        features[f"max.{name}"] = max(values)
        features[f"mean.{name}"] = math.fsum(values) / len(values)
    return features


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


def test_glif_trace_features_match_independent_linear_euler_solution() -> None:
    """Committed GLIF5 features must match an independent subthreshold Euler recurrence."""
    spec = load_reference_trace_spec("glif_constant_current_threshold_adaptation")

    expected = _glif_subthreshold_euler_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "glif"
    assert spec.provenance.kind == "analytic_linear_euler_reference"
    assert spec.provenance.citation == "doi:10.1038/s41467-017-02717-4"
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_izhikevich_trace_features_match_independent_euler_solution() -> None:
    """Committed Izhikevich RS features must match an independent explicit-Euler recurrence."""
    spec = load_reference_trace_spec("izhikevich_regular_spiking_doi")

    expected = _izhikevich_rs_euler_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "izhikevich"
    assert spec.provenance.kind == "independent_euler_reference"
    assert spec.provenance.citation == "doi:10.1109/TNN.2003.820440"
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_fitzhugh_nagumo_trace_features_match_independent_euler_solution() -> None:
    """Committed FitzHugh-Nagumo features must match an independent explicit-Euler recurrence."""
    spec = load_reference_trace_spec("fitzhugh_nagumo_driven_oscillation_doi")

    expected = _fitzhugh_nagumo_euler_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "fitzhugh_nagumo"
    assert spec.provenance.kind == "independent_euler_reference"
    assert spec.provenance.citation == "doi:10.1016/S0006-3495(61)86902-6"
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_adex_trace_features_match_independent_euler_solution() -> None:
    """Committed AdEx features must match an independent subthreshold Euler recurrence."""
    spec = load_reference_trace_spec("adex_resting_adaptation_doi")

    expected = _adex_subthreshold_euler_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "adex"
    assert spec.provenance.kind == "independent_euler_reference"
    assert spec.provenance.citation == "doi:10.1152/jn.00686.2005"
    assert set(expected) == set(spec.expected_features)
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
