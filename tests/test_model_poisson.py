# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: PoissonNeuron

"""Full pipeline test for PoissonNeuron (Poisson spike generator).

Stateless process: P(spike in dt) = 1 - exp(-λ·dt/1000).
No membrane dynamics — pure stochastic rate coding."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.poisson import PoissonNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _poisson_step_probability(rate_hz: float, dt_ms: float) -> float:
    return -math.expm1(-rate_hz * dt_ms / 1000.0)


# ---------------------------------------------------------------------------
# 1. Isolation — construction and basic API
# ---------------------------------------------------------------------------


class TestPoissonIsolation:
    def test_construction_defaults(self) -> None:
        n = PoissonNeuron()
        assert n.rate_hz == 100.0
        assert n.dt_ms == 1.0

    def test_step_returns_binary(self) -> None:
        n = PoissonNeuron()
        assert n.step() in (0, 1)

    def test_rng_initialised(self) -> None:
        """Internal RNG should be initialised after construction."""
        n = PoissonNeuron()
        assert hasattr(n, "_rng")

    def test_reset_replays_the_seeded_event_stream(self) -> None:
        """Reset restores execution state even though no membrane state exists."""
        n = PoissonNeuron(seed=0xACE1)
        first = [n.step() for _ in range(1000)]
        assert n.rng_state != n.initial_seed
        n.reset()
        assert n.rng_state == n.initial_seed == 0xACE1
        assert [n.step() for _ in range(1000)] == first

    def test_default_seed_is_reproducible_and_none_requests_entropy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitted seeds replay; explicit ``None`` constructs independent streams."""
        default_a = PoissonNeuron(rate_hz=200.0)
        default_b = PoissonNeuron(rate_hz=200.0)
        assert [default_a.step() for _ in range(1000)] == [default_b.step() for _ in range(1000)]

        entropy_words = iter((0, 1))
        monkeypatch.setattr(
            "sc_neurocore.neurons._stochastic_threshold.secrets.randbelow",
            lambda _upper: next(entropy_words),
        )
        entropy_a = PoissonNeuron(rate_hz=200.0, seed=None)
        entropy_b = PoissonNeuron(rate_hz=200.0, seed=None)
        assert (entropy_a.initial_seed, entropy_b.initial_seed) == (1, 2)


# ---------------------------------------------------------------------------
# 2. Spike rate — statistical tests
# ---------------------------------------------------------------------------


class TestPoissonRate:
    def test_mean_rate_matches_lambda(self) -> None:
        """Over many trials, spike rate ≈ 1 - exp(-λ·dt/1000).

        At rate=100Hz, dt=1ms: P(spike) ≈ 0.09516.
        """
        n = PoissonNeuron(rate_hz=100.0, dt_ms=1.0)
        N = 100000
        spikes = sum(n.step() for _ in range(N))
        p = _poisson_step_probability(100.0, 1.0)
        expected = N * p
        # 5σ tolerance for statistical test
        sigma = np.sqrt(N * p * (1.0 - p))
        assert abs(spikes - expected) < 5 * sigma, (
            f"spikes={spikes}, expected={expected:.0f}, 5σ={5 * sigma:.0f}"
        )

    @pytest.mark.parametrize("rate_hz", [50.0, 100.0, 200.0, 500.0])
    def test_rate_proportional_to_lambda(self, rate_hz: float) -> None:
        """Spike count ∝ rate_hz."""
        n = PoissonNeuron(rate_hz=rate_hz, dt_ms=1.0)
        N = 50000
        spikes = sum(n.step() for _ in range(N))
        p = _poisson_step_probability(rate_hz, 1.0)
        expected = N * p
        sigma = np.sqrt(N * p * (1.0 - p))
        assert abs(spikes - expected) < 5 * sigma

    def test_higher_rate_more_spikes(self) -> None:
        """Monotonicity: higher λ → more spikes."""
        n_low = PoissonNeuron(rate_hz=50.0)
        n_high = PoissonNeuron(rate_hz=500.0)
        N = 50000
        s_low = sum(n_low.step() for _ in range(N))
        s_high = sum(n_high.step() for _ in range(N))
        assert s_high > s_low

    def test_zero_rate_no_spikes(self) -> None:
        """λ=0 → P(spike) = 0 → no spikes ever."""
        n = PoissonNeuron(rate_hz=0.0)
        spikes = sum(n.step() for _ in range(100000))
        assert spikes == 0

    def test_rate_override(self) -> None:
        """rate_override parameter overrides stored rate."""
        n = PoissonNeuron(rate_hz=100.0)
        # Override to 0 — no spikes
        spikes = sum(n.step(rate_override=0.0) for _ in range(10000))
        assert spikes == 0

    def test_negative_rate_override_uses_stored(self) -> None:
        """Negative rate_override → use stored rate_hz (API convention)."""
        n = PoissonNeuron(rate_hz=500.0, dt_ms=1.0)
        spikes = sum(n.step(rate_override=-1.0) for _ in range(10000))
        expected = 10000 * 0.5
        assert spikes > expected * 0.5  # should be near 5000


# ---------------------------------------------------------------------------
# 3. Statistical properties — ISI distribution
# ---------------------------------------------------------------------------


class TestPoissonISI:
    def test_isi_exponentially_distributed(self) -> None:
        """For Poisson process, ISI follows geometric distribution.

        Mean ISI = 1/p where p = λ·dt/1000.
        For rate=200Hz, dt=1ms: p=1-exp(-0.2), mean ISI=1/p steps.
        """
        n = PoissonNeuron(rate_hz=200.0, dt_ms=1.0)
        spike_times = []
        for t in range(200000):
            if n.step() == 1:
                spike_times.append(t)
        isis = np.diff(spike_times).astype(float)
        assert len(isis) >= 1000
        mean_isi = np.mean(isis)
        expected_mean = 1.0 / _poisson_step_probability(200.0, 1.0)
        assert abs(mean_isi - expected_mean) < 0.5, (
            f"mean ISI={mean_isi:.2f}, expected ≈{expected_mean:.1f}"
        )

    def test_cv_isi_near_one(self) -> None:
        """CV(ISI) ≈ 1 for Poisson process (geometric distribution)."""
        n = PoissonNeuron(rate_hz=200.0, dt_ms=1.0)
        spike_times = []
        for t in range(200000):
            if n.step() == 1:
                spike_times.append(t)
        isis = np.diff(spike_times).astype(float)
        cv = np.std(isis) / np.mean(isis)
        # Geometric CV = sqrt(1-p)/p ≈ 1 for small p
        # For p=0.2: CV = sqrt(0.8)/0.2 / (1/0.2) ≈ 0.894
        assert 0.7 < cv < 1.3, f"CV(ISI) = {cv:.3f}, expected ≈1"

    def test_no_refractory_period(self) -> None:
        """Consecutive spikes are possible (ISI=1 allowed)."""
        n = PoissonNeuron(rate_hz=800.0, dt_ms=1.0)  # p=0.8
        spike_times = []
        for t in range(10000):
            if n.step() == 1:
                spike_times.append(t)
        isis = np.diff(spike_times)
        assert 1 in isis, "Expected consecutive spikes (ISI=1) at high rate"


# ---------------------------------------------------------------------------
# 4. dt_ms scaling
# ---------------------------------------------------------------------------


class TestPoissonDtScaling:
    def test_dt_scales_probability(self) -> None:
        """P(spike) = λ·dt/1000. Doubling dt doubles spike probability."""
        N = 100000
        n1 = PoissonNeuron(rate_hz=100.0, dt_ms=0.5)
        n2 = PoissonNeuron(rate_hz=100.0, dt_ms=1.0)
        s1 = sum(n1.step() for _ in range(N))
        s2 = sum(n2.step() for _ in range(N))
        ratio = s2 / s1 if s1 > 0 else 0
        assert 1.5 < ratio < 2.5, f"ratio = {ratio:.2f}, expected ≈2.0"

    def test_small_dt_rare_spikes(self) -> None:
        """Very small dt → very rare spikes."""
        n = PoissonNeuron(rate_hz=100.0, dt_ms=0.01)
        # P = 100 * 0.01 / 1000 = 0.001
        spikes = sum(n.step() for _ in range(100000))
        assert spikes < 500  # expected ~100


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


# ---------------------------------------------------------------------------
# 5. Stochasticity
# ---------------------------------------------------------------------------


class TestPoissonStochasticity:
    def test_different_runs_differ(self) -> None:
        """Two neurons with distinct seeds produce distinct spike trains."""
        n1 = PoissonNeuron(rate_hz=200.0, seed=1)
        n2 = PoissonNeuron(rate_hz=200.0, seed=2)
        t1 = [n1.step() for _ in range(1000)]
        t2 = [n2.step() for _ in range(1000)]
        # Extremely unlikely to be identical
        assert t1 != t2

    def test_stateless(self) -> None:
        """Spike probability doesn't depend on history (memoryless)."""
        n = PoissonNeuron(rate_hz=200.0, dt_ms=1.0)
        # Run 10k steps, then measure rate for next 10k
        for _ in range(10000):
            n.step()
        spikes_after = sum(n.step() for _ in range(50000))
        p = _poisson_step_probability(200.0, 1.0)
        expected = 50000 * p
        sigma = np.sqrt(50000 * p * (1.0 - p))
        assert abs(spikes_after - expected) < 5 * sigma


# ---------------------------------------------------------------------------
# 6. Network
# ---------------------------------------------------------------------------


class TestPoissonNetwork:
    def test_population(self) -> None:
        pop = Population(PoissonNeuron, n=20, label="poisson")
        assert pop.n == 20

    def test_network_spikes(self) -> None:
        pop = Population(PoissonNeuron, n=20, label="poisson")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        # PoissonNeuron ignores input (fires at its own rate)
        assert mon.count > 0


# ---------------------------------------------------------------------------
# 7. Analysis
# ---------------------------------------------------------------------------


class TestPoissonAnalysis:
    def test_spike_count(self) -> None:
        n = PoissonNeuron(rate_hz=200.0)
        train = np.array([float(n.step()) for _ in range(10000)])
        count = spike_count(train)
        assert 1000 < count < 3000  # expected ~2000

    def test_spike_count_consistency(self) -> None:
        n = PoissonNeuron(rate_hz=200.0)
        train = np.array([float(n.step()) for _ in range(10000)])
        assert spike_count(train) == int(train.sum())
