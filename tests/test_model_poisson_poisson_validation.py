# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonValidation from former test_model_poisson.py

"""Focused suite: TestPoissonValidation from former test_model_poisson.py."""

from __future__ import annotations

from tests.model_poisson_support import *  # noqa: F403

class TestPoissonValidation:
    @pytest.mark.parametrize("rate_hz", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_baseline_rate(self, rate_hz: float) -> None:
        with pytest.raises(ValueError, match="rate_hz"):
            PoissonNeuron(rate_hz=rate_hz)

    @pytest.mark.parametrize("dt_ms", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_dt(self, dt_ms: float) -> None:
        with pytest.raises(ValueError, match="dt_ms"):
            PoissonNeuron(dt_ms=dt_ms)

    def test_probability_uses_bounded_poisson_interval_transform(self) -> None:
        neuron = PoissonNeuron(rate_hz=2000.0, dt_ms=1.0)
        assert neuron._probability(neuron.rate_hz) == pytest.approx(1.0 - math.exp(-2.0))

    def test_probability_rejects_an_invalid_direct_rate_contract(self) -> None:
        """The internal probability boundary rejects invalid caller input."""
        neuron = PoissonNeuron()
        with pytest.raises(ValueError, match="rate_hz"):
            neuron._probability(-1.0)

    def test_probability_rejects_a_non_finite_math_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed exponential evaluation cannot advance the event stream."""
        neuron = PoissonNeuron(seed=42)
        initial_state = neuron.rng_state
        monkeypatch.setattr(
            "sc_neurocore.neurons.models.poisson.math.expm1",
            lambda _value: math.nan,
        )

        with pytest.raises(ValueError, match="spike probability"):
            neuron.step()

        assert neuron.rng_state == initial_state

    @pytest.mark.parametrize("rate_override", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_rate_override(self, rate_override: float) -> None:
        n = PoissonNeuron(rate_hz=100.0)
        with pytest.raises(ValueError, match="rate_override"):
            n.step(rate_override=rate_override)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("rate_hz", -1.0, "rate_hz"),
            ("rate_hz", np.nan, "rate_hz"),
            ("dt_ms", 0.0, "dt_ms"),
            ("dt_ms", np.inf, "dt_ms"),
        ],
    )
    def test_rejects_corrupted_runtime_rate_state_before_probability(
        self, field: str, value: float, message: str
    ) -> None:
        n = PoissonNeuron(rate_hz=100.0, dt_ms=1.0)
        setattr(n, field, value)
        with pytest.raises(ValueError, match=message):
            n.step()

    def test_rejects_non_finite_interval_hazard_before_sampling(self) -> None:
        n = PoissonNeuron(rate_hz=100.0, dt_ms=1.0)
        n.rate_hz = 1.0e308
        n.dt_ms = 1.0e308
        with pytest.raises(ValueError, match="interval hazard"):
            n.step()

    def test_high_rate_override_saturates_without_invalid_probability(self) -> None:
        n = PoissonNeuron(rate_hz=100.0, dt_ms=1.0)
        spikes = sum(n.step(rate_override=1.0e9) for _ in range(100))
        assert spikes == 100

    @pytest.mark.parametrize(
        ("n_steps", "backend", "rate_override", "message"),
        [
            (True, "python", -1.0, "n_steps"),
            (1.5, "python", -1.0, "n_steps"),
            (-1, "python", -1.0, "n_steps"),
            (1, "unknown", -1.0, "backend"),
            (1, "python", math.nan, "rate_override"),
        ],
    )
    def test_batch_rejects_invalid_public_contracts(
        self,
        n_steps: object,
        backend: str,
        rate_override: float,
        message: str,
    ) -> None:
        """Reject malformed batch work before changing the replay state."""
        neuron = PoissonNeuron(seed=42)
        initial_state = neuron.rng_state
        with pytest.raises(ValueError, match=message):
            neuron.simulate(cast(int, n_steps), rate_override, backend)
        assert neuron.rng_state == initial_state

    @pytest.mark.parametrize("selected", ["mojo", "go", "julia", "python"])
    def test_auto_batch_follows_the_declared_backend_fallback_order(
        self, selected: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exercise every production fallback without changing its stable order."""
        from sc_neurocore.accel import poisson as backends

        monkeypatch.setattr(backends, "_HAS_RUST", False)
        order = ("mojo", "go", "julia")
        availability = {
            backend: order.index(backend) >= order.index(selected)
            if selected != "python"
            else False
            for backend in order
        }
        calls: list[str] = []

        def loader(backend: str) -> bool:
            calls.append(f"load:{backend}")
            return availability[backend]

        def runner(
            backend: str,
        ) -> Callable[
            [float, float, int, int, float],
            tuple[npt.NDArray[np.uint8], int],
        ]:
            def run(
                _rate_hz: float,
                _dt_ms: float,
                rng_state: int,
                n_steps: int,
                _rate_override: float,
            ) -> tuple[npt.NDArray[np.uint8], int]:
                calls.append(f"run:{backend}")
                return np.zeros(n_steps, dtype=np.uint8), rng_state

            return run

        for backend in order:
            monkeypatch.setattr(backends, f"ensure_{backend}_loaded", lambda b=backend: loader(b))
            monkeypatch.setattr(backends, f"simulate_{backend}", runner(backend))

        neuron = PoissonNeuron(rate_hz=0.0, seed=42)
        events, count = neuron.simulate(2, backend="auto")

        assert events.tolist() == [0, 0]
        assert count == 0
        if selected == "python":
            assert all(not call.startswith("run:") for call in calls)
        else:
            assert calls[-1] == f"run:{selected}"

    def test_unavailable_explicit_backend_preserves_rng(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit unavailable lane fails without consuming randomness."""
        from sc_neurocore.accel import poisson as backends

        monkeypatch.setattr(backends, "_HAS_RUST", False)
        neuron = PoissonNeuron(seed=42)
        initial_state = neuron.rng_state

        with pytest.raises(RuntimeError, match="Rust Poisson backend is unavailable"):
            neuron.simulate(1, backend="rust")

        assert neuron.rng_state == initial_state

    def test_native_failure_restores_the_entry_rng_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A native execution error cannot partially commit the batch state."""
        from sc_neurocore.accel import poisson as backends

        def fail(
            _rate_hz: float,
            _dt_ms: float,
            _rng_state: int,
            _n_steps: int,
            _rate_override: float,
        ) -> tuple[npt.NDArray[np.uint8], int]:
            raise FloatingPointError("native failure")

        monkeypatch.setattr(backends, "_HAS_RUST", True)
        monkeypatch.setattr(backends, "simulate_rust", fail)
        neuron = PoissonNeuron(seed=42)
        initial_state = neuron.rng_state

        with pytest.raises(FloatingPointError, match="native failure"):
            neuron.simulate(1, backend="rust")

        assert neuron.rng_state == initial_state
