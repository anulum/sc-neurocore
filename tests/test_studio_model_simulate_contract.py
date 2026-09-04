# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio model-run contract (public Python surface)

"""Fail-closed contract of ``simulate_model`` and the model-run input resolver.

Every case here fails on the former behaviour, which filtered unknown overrides,
rounded integer fields, retried constructors with defaults, retried ``step``
with casts, replaced numerical failures by ``spike = 0`` and non-scalar or
non-finite state by ``0.0``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron
from sc_neurocore.studio import model_simulate
from sc_neurocore.studio.model_run_contract import (
    RECEIPT_SCHEMA_VERSION,
    STUDIO_DEFAULT_DT_MS,
    ModelInputError,
    ModelSimulationFailure,
    model_drive_contract,
    model_parameter_contracts,
    resolve_model_run_inputs,
)
from sc_neurocore.studio.models import simulate_model
from sc_neurocore.studio.simulation import MAX_STEPS

ATIF = "AdaptiveThresholdIFNeuron"
ATIF_OVERRIDES = {"delta_theta": 8.0, "tau_theta": 30.0, "theta_rest": -48.0}


def _rejects(**kwargs: object) -> ModelInputError:
    with pytest.raises(ModelInputError) as info:
        simulate_model(**kwargs)  # type: ignore[arg-type]  # invalid inputs are the subject
    return info.value


class TestParameterContracts:
    def test_adaptive_threshold_if_exposes_every_float_field(self) -> None:
        contracts = model_parameter_contracts(AdaptiveThresholdIFNeuron)
        names = {field.name for field in dataclasses.fields(AdaptiveThresholdIFNeuron)}
        assert set(contracts.overridable) == names
        assert contracts.unsupported == {}
        assert all(contract.kind == "float" for contract in contracts.overridable.values())
        assert contracts.overridable["tau_theta"].default == 50.0

    def test_non_numeric_private_and_derived_fields_are_named_unsupported(self) -> None:
        from sc_neurocore.studio.model_introspection import _load_class

        adex = model_parameter_contracts(_load_class("AdExNeuron"))
        assert "integrator" not in adex.overridable
        assert adex.unsupported["integrator"].startswith("non-numeric field")

        rall = model_parameter_contracts(_load_class("RallCableNeuron"))
        assert "v" not in rall.overridable
        assert "init=False" in rall.unsupported["v"]

        arcane = model_parameter_contracts(_load_class("ArcaneNeuron"))
        assert "_hist_idx" not in arcane.overridable
        assert arcane.unsupported["_hist_idx"] == "private model state is not an input"

        integer_qif = model_parameter_contracts(_load_class("IntegerQIFNeuron"))
        assert integer_qif.overridable["v_threshold"].kind == "int"

    def test_drive_contract_follows_the_step_signature(self) -> None:
        from sc_neurocore.studio.model_introspection import _load_class

        integer_qif = model_drive_contract("IntegerQIFNeuron", _load_class("IntegerQIFNeuron"))
        assert (integer_qif.parameter, integer_qif.kind) == ("current", "int")
        loihi = model_drive_contract("Loihi2Neuron", _load_class("Loihi2Neuron"))
        assert (loihi.parameter, loihi.kind) == ("weighted_input", "int")
        atif = model_drive_contract(ATIF, AdaptiveThresholdIFNeuron)
        assert (atif.parameter, atif.kind, atif.positional_only) == ("current", "float", False)
        with pytest.raises(ModelInputError) as info:
            model_drive_contract("DendriticNMDANeuron", _load_class("DendriticNMDANeuron"))
        assert info.value.field == "step"
        assert "glutamate" in info.value.reason


class TestRejectedRequests:
    def test_unknown_model_name(self) -> None:
        error = _rejects(name="NoSuchNeuron")
        assert (error.model, error.field, error.reason) == ("NoSuchNeuron", "name", "unknown model")
        assert error.to_public_detail() == {
            "error": "invalid_model_input",
            "model": "NoSuchNeuron",
            "field": "name",
            "reason": "unknown model",
        }

    def test_unknown_parameter_is_rejected_not_filtered(self) -> None:
        error = _rejects(name=ATIF, param_overrides={"no_such_parameter": 1.0})
        assert (error.model, error.field, error.reason) == (
            ATIF,
            "params.no_such_parameter",
            "unknown parameter",
        )

    def test_non_numeric_field_is_rejected_not_defaulted(self) -> None:
        error = _rejects(name="AdExNeuron", param_overrides={"integrator": 1.0})
        assert error.field == "params.integrator"
        assert error.reason.startswith("not overridable: non-numeric field")

    def test_fractional_value_for_integer_field(self) -> None:
        error = _rejects(name="IntegerQIFNeuron", param_overrides={"v_threshold": 30.5})
        assert error.field == "params.v_threshold"
        assert "fractional" in error.reason

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "10"])
    def test_non_finite_or_non_numeric_value(self, value: object) -> None:
        error = _rejects(name=ATIF, param_overrides={"tau_m": value})
        assert (error.field, error.reason) == ("params.tau_m", "must be a finite number")

    def test_dt_inside_params_is_rejected(self) -> None:
        error = _rejects(name=ATIF, param_overrides={"dt": 0.05})
        assert error.field == "params.dt"

    def test_invalid_constructor_is_reported_not_replaced_by_defaults(self) -> None:
        error = _rejects(name=ATIF, param_overrides={"theta_rest": -70.0}, use_fast_path=False)
        assert (error.model, error.field) == (ATIF, "constructor")
        assert "theta_rest must be greater than v_rest" in error.reason

    def test_explicit_dt_on_a_model_without_timestep(self) -> None:
        error = _rejects(name="ChialvoMapNeuron", dt=0.5, use_fast_path=False)
        assert error.field == "dt"
        assert "no integration timestep" in error.reason
        accepted = simulate_model("ChialvoMapNeuron", dt=STUDIO_DEFAULT_DT_MS, use_fast_path=False)
        assert accepted["effective_inputs"]["dt_source"] == "studio_default"
        assert accepted["dt"] == STUDIO_DEFAULT_DT_MS

    def test_fixed_step_model_reports_its_own_step_and_rejects_other_dt(self) -> None:
        error = _rejects(name="IntegerQIFNeuron", dt=0.1, use_fast_path=False)
        assert error.field == "dt"
        assert "fixed step of 1.0 ms" in error.reason
        accepted = simulate_model("IntegerQIFNeuron", duration=10.0, use_fast_path=False)
        assert accepted["dt"] == 1.0
        assert accepted["n_steps"] == 10
        assert accepted["effective_inputs"]["dt_source"] == "model_attribute"
        explicit = simulate_model("IntegerQIFNeuron", dt=1.0, duration=10.0, use_fast_path=False)
        assert explicit["states"] == accepted["states"]

    def test_unsupported_protocol_is_rejected_not_constant(self) -> None:
        error = _rejects(name=ATIF, protocol="sawtooth", use_fast_path=False)
        assert error.field == "protocol"
        assert "sawtooth" in error.reason

    def test_integer_drive_model_rejects_fractional_samples(self) -> None:
        error = _rejects(
            name="IntegerQIFNeuron", protocol="sine", current=10.0, use_fast_path=False
        )
        assert error.field == "current"
        assert "integral current samples" in error.reason

    def test_step_requiring_extra_inputs_is_rejected_before_running(self) -> None:
        error = _rejects(name="DendriticNMDANeuron", use_fast_path=False)
        assert error.field == "step"
        assert "glutamate" in error.reason

    def test_duration_shorter_than_one_step(self) -> None:
        error = _rejects(name=ATIF, duration=0.01, use_fast_path=False)
        assert error.field == "duration"

    def test_rejection_leaves_class_defaults_and_later_runs_untouched(self) -> None:
        before = simulate_model(ATIF, param_overrides=ATIF_OVERRIDES, use_fast_path=False)
        defaults = {f.name: f.default for f in dataclasses.fields(AdaptiveThresholdIFNeuron)}
        _rejects(name=ATIF, param_overrides={"tau_m": float("nan")})
        _rejects(name=ATIF, param_overrides={"theta_rest": -70.0}, use_fast_path=False)
        assert {
            f.name: f.default for f in dataclasses.fields(AdaptiveThresholdIFNeuron)
        } == defaults
        after = simulate_model(ATIF, param_overrides=ATIF_OVERRIDES, use_fast_path=False)
        assert after == before


class TestNumericalFailure:
    def test_intermediate_overflow_raises_with_step_and_time(self) -> None:
        with pytest.raises(ModelSimulationFailure) as info:
            simulate_model(
                "HodgkinHuxleyNeuron",
                protocol="ramp",
                current=1e300,
                duration=5.0,
                use_fast_path=False,
            )
        failure = info.value
        assert failure.model == "HodgkinHuxleyNeuron"
        assert failure.backend == "python"
        assert failure.step > 0
        assert failure.time_ms == pytest.approx(failure.step * 0.01)
        assert failure.diagnostic.startswith("OverflowError")
        detail = failure.to_public_detail()
        assert detail["error"] == "model_simulation_failed"
        assert detail["step"] == failure.step

    def test_rust_backend_non_finite_voltage_is_a_failure_not_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _batch(
            _name: str, n_steps: int, _current: np.ndarray[Any, Any]
        ) -> dict[str, np.ndarray[Any, Any]]:
            voltages = np.full(n_steps, -65.0)
            voltages[7] = np.nan
            return {"voltages": voltages, "spikes": np.array([], dtype=np.int64)}

        monkeypatch.setattr(model_simulate, "_load_rust_batch_simulate", lambda: _batch)
        with pytest.raises(ModelSimulationFailure) as info:
            simulate_model("AdExNeuron", duration=2.0)
        assert (info.value.backend, info.value.step) == ("rust", 7)
        assert info.value.time_ms == pytest.approx(0.7)
        assert "'v'" in info.value.diagnostic


class TestSuccessfulRuns:
    def test_valid_nondefault_adaptive_threshold_run_matches_reference_loop(self) -> None:
        result = simulate_model(ATIF, param_overrides=ATIF_OVERRIDES, current=20.0, duration=100.0)
        receipt = result["effective_inputs"]
        assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION
        assert receipt["backend"] == "python"
        assert receipt["overrides_applied"] == sorted(ATIF_OVERRIDES)
        assert (receipt["dt"], receipt["dt_source"]) == (0.1, "model_default")
        expected = {f.name: f.default for f in dataclasses.fields(AdaptiveThresholdIFNeuron)}
        expected.update(ATIF_OVERRIDES)
        assert receipt["parameters"] == expected
        assert receipt["drive"] == {"step_parameter": "current", "kind": "float"}
        assert receipt["state_recording"] == {"recorded": ["v"], "excluded": []}
        assert receipt["steps_truncated"] is False

        reference = AdaptiveThresholdIFNeuron(**ATIF_OVERRIDES)
        spikes: list[int] = []
        expected_v: list[float] = []
        for t in range(1000):
            if reference.step(20.0):
                spikes.append(t)
            expected_v.append(reference.v)
        assert result["spikes"] == spikes
        assert result["spike_count"] == len(spikes) > 0
        assert result["n_steps"] == 1000
        assert all(math.isfinite(v) for v in result["states"]["v"])
        assert result["states"]["v"] == expected_v

    def test_non_scalar_state_is_declared_excluded_not_zeroed(self) -> None:
        result = simulate_model("RallCableNeuron", duration=1.0, use_fast_path=False)
        assert "v" not in result["states"]
        excluded = result["effective_inputs"]["state_recording"]["excluded"]
        assert excluded == [{"name": "v", "reason": "non-scalar state (ndarray)"}]

    def test_rust_fast_path_receipt_declares_unexported_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[str, int]] = []

        def _batch(
            name: str, n_steps: int, current: np.ndarray[Any, Any]
        ) -> dict[str, np.ndarray[Any, Any]]:
            seen.append((name, n_steps))
            return {
                "voltages": np.full(n_steps, -65.0) + current * 0.0,
                "spikes": np.array([3], dtype=np.int64),
            }

        monkeypatch.setattr(model_simulate, "_load_rust_batch_simulate", lambda: _batch)
        result = simulate_model("AdExNeuron", duration=2.0)
        assert seen == [("AdExNeuron", 20)]
        receipt = result["effective_inputs"]
        assert receipt["backend"] == "rust"
        assert receipt["overrides_applied"] == []
        assert receipt["state_recording"]["recorded"] == ["v"]
        assert {"name": "w", "reason": "not exported by the Rust batch backend"} in receipt[
            "state_recording"
        ]["excluded"]
        assert result["spikes"] == [3]

    def test_overrides_bypass_the_rust_fast_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _never(*_args: object) -> None:
            raise AssertionError("Rust path must not run with overrides")

        monkeypatch.setattr(model_simulate, "_load_rust_batch_simulate", lambda: _never)
        result = simulate_model("AdExNeuron", param_overrides={"tau_w": 80.0}, duration=2.0)
        assert result["effective_inputs"]["backend"] == "python"
        assert result["effective_inputs"]["parameters"]["tau_w"] == 80.0

    def test_step_cap_is_declared_as_truncation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _batch(
            _name: str, n_steps: int, _current: np.ndarray[Any, Any]
        ) -> dict[str, np.ndarray[Any, Any]]:
            return {"voltages": np.full(n_steps, -65.0), "spikes": np.array([], dtype=np.int64)}

        monkeypatch.setattr(model_simulate, "_load_rust_batch_simulate", lambda: _batch)
        result = simulate_model("AdExNeuron", duration=1e9)
        assert result["n_steps"] == MAX_STEPS
        assert result["effective_inputs"]["steps_truncated"] is True
        assert result["effective_inputs"]["plot_stride"] == MAX_STEPS // 5_000

    def test_resolver_applies_integer_override_exactly(self) -> None:
        inputs = resolve_model_run_inputs("IntegerQIFNeuron", {"v_threshold": 40.0}, None)
        assert inputs.constructor_kwargs == {"v_threshold": 40}
        assert isinstance(inputs.constructor_kwargs["v_threshold"], int)
        assert (inputs.dt, inputs.dt_source) == (1.0, "model_attribute")
        assert inputs.effective_parameters()["v_threshold"] == 40
