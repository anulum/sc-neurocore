# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEscapeRateValidation from former test_model_escape_rate.py

"""Focused suite: TestEscapeRateValidation from former test_model_escape_rate.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.escape_rate import (
    EscapeRateNeuron,
)


class TestEscapeRateValidation:
    """Input and state validation of the EscapeRateNeuron."""

    @pytest.mark.parametrize("field", ["v", "v_rest", "v_reset", "v_threshold"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float) -> None:
        """The model rejects non-finite voltage parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            EscapeRateNeuron(**kwargs)

    @pytest.mark.parametrize("field", ["tau_m", "rho_0", "delta_u", "resistance", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(
        self, field: str, value: float
    ) -> None:
        """The model rejects non-positive or non-finite scale parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            EscapeRateNeuron(**kwargs)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_voltage_mutation(self, current: float) -> None:
        """The model rejects non-finite current before voltage mutation."""
        n = EscapeRateNeuron(v=-65.0)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("tau_m", 0.0, "tau_m"),
            ("rho_0", -1.0, "rho_0"),
            ("delta_u", 0.0, "delta_u"),
            ("resistance", np.nan, "resistance"),
            ("dt", np.inf, "dt"),
            ("v_rest", np.nan, "v_rest"),
        ],
    )
    def test_rejects_corrupted_runtime_state_before_voltage_mutation(
        self, field: str, value: float, message: str
    ) -> None:
        """The model rejects corrupted runtime state before voltage mutation."""
        n = EscapeRateNeuron(v=-65.0)
        setattr(n, field, value)
        before = -65.0
        with pytest.raises(ValueError, match=message):
            n.step(1.0)
        assert n.v == before

    def test_rejects_non_finite_voltage_candidate_before_reset_mutation(self) -> None:
        """The model rejects non-finite voltage candidate before reset mutation."""
        n = EscapeRateNeuron(v=-65.0, v_threshold=1.0e308, resistance=1.0e308)
        before = n.v
        with pytest.raises(ValueError, match="voltage candidate"):
            n.step(1.0e308)
        assert n.v == before

    def test_rejects_non_finite_hazard_before_random_draw(self) -> None:
        """The model rejects non-finite hazard before random draw."""
        n = EscapeRateNeuron(v=-50.0, rho_0=1.0e308, dt=10.0, seed=42)
        before = (n.v, n.rng_state)
        with pytest.raises(ValueError, match="escape hazard"):
            n.step(20.0)
        assert (n.v, n.rng_state) == before

    def test_python_batch_failure_restores_voltage_and_rng(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Python batch failure restores voltage and RNG."""
        n = EscapeRateNeuron(v=-65.0, seed=42)
        before = (n.v, n.rng_state)
        original_step = EscapeRateNeuron.step
        calls = 0

        def fail_after_one_step(neuron: EscapeRateNeuron, current: float) -> int:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise FloatingPointError("injected batch failure")
            return original_step(neuron, current)

        monkeypatch.setattr(EscapeRateNeuron, "step", fail_after_one_step)
        with pytest.raises(FloatingPointError, match="injected batch failure"):
            n.simulate(10, current=20.0, backend="python")
        assert (n.v, n.rng_state) == before

    def test_malformed_native_result_is_rejected_before_state_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Malformed native result is rejected before state commit."""
        from sc_neurocore.accel import escape_rate as backends

        n = EscapeRateNeuron(v=-65.0, seed=42)
        before = (n.v, n.rng_state)
        monkeypatch.setattr(backends, "_HAS_RUST", True)
        monkeypatch.setattr(
            backends,
            "simulate_rust",
            lambda *_args: (np.array([123.0]), np.array([1.5]), 123.0, 1.5),
        )
        with pytest.raises(FloatingPointError, match="non-binary events"):
            n.simulate(1, current=20.0, backend="rust")
        assert (n.v, n.rng_state) == before
