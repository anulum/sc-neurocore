# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: PinskyRinzelNeuron

"""Full pipeline test for PinskyRinzelNeuron (Pinsky & Rinzel 1994).

Two-compartment CA3 pyramidal cell integrated with fourth-order Runge-Kutta:
soma (fast Na/K-DR) coupled to dendrite (Ca, K-AHP, K-C). Eight states
``(v_s, v_d, h, n, s, c, q, ca)``; ``step(current_soma, current_dend)`` has a
dual-input signature. The model fires repetitively at low somatic drive and
enters depolarisation block (Na inactivation) at high drive, giving a
non-monotonic f-I relation. Reference: PR1994 / ModelDB 35358.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.pinsky_rinzel import PinskyRinzelNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(
    neuron: PinskyRinzelNeuron, current_soma: float, steps: int, current_dend: float = 0.0
) -> list[int]:
    """Return the indices of steps on which a somatic spike was registered."""
    return [t for t in range(steps) if neuron.step(current_soma, current_dend) == 1]


# ---------------------------------------------------------------------------
# 1. Isolation — construction, state evolution, reset
# ---------------------------------------------------------------------------


class TestPinskyRinzelIsolation:
    def test_construction_defaults(self):
        n = PinskyRinzelNeuron()
        assert n.v_s == -60.0
        assert n.v_d == -60.0
        assert (n.h, n.n, n.s, n.c, n.q, n.ca) == (0.999, 0.001, 0.009, 0.007, 0.01, 0.2)
        assert n.cm == 3.0
        assert n.gc == 2.1
        assert n.p == 0.5
        assert n.dt == 0.02

    def test_step_returns_binary(self):
        assert PinskyRinzelNeuron().step(0.0) in (0, 1)

    def test_dual_input_signature(self):
        assert PinskyRinzelNeuron().step(5.0, 3.0) in (0, 1)

    def test_eight_state_variables_evolve(self):
        n = PinskyRinzelNeuron()
        initial = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca)
        for _ in range(2000):
            n.step(20.0)
        final = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca)
        diffs = [abs(f - i) for f, i in zip(final, initial)]
        assert all(d > 1e-10 for d in diffs), f"Some variables did not evolve: {diffs}"

    def test_state_finite_long_run(self):
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(30.0)
        for var in (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca):
            assert np.isfinite(var)

    def test_reset_restores_initial(self):
        n = PinskyRinzelNeuron()
        for _ in range(1000):
            n.step(30.0)
        n.reset()
        assert (n.v_s, n.v_d) == (-60.0, -60.0)
        assert (n.h, n.n, n.s, n.c, n.q, n.ca) == (0.999, 0.001, 0.009, 0.007, 0.01, 0.2)


# ---------------------------------------------------------------------------
# 2. Compartmental coupling and calcium
# ---------------------------------------------------------------------------


class TestPinskyRinzelCompartments:
    def test_soma_dendrite_coupling(self):
        coupled = PinskyRinzelNeuron(gc=2.1)
        uncoupled = PinskyRinzelNeuron(gc=0.001)
        for _ in range(5000):
            coupled.step(20.0, 0.0)
            uncoupled.step(20.0, 0.0)
        assert abs(coupled.v_d - uncoupled.v_d) > 1.0

    def test_dendritic_drive_evokes_spiking(self):
        n = PinskyRinzelNeuron()
        assert len(_run(n, current_soma=0.0, steps=50000, current_dend=20.0)) > 0

    def test_calcium_accumulates_during_spiking(self):
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(20.0)
        assert n.ca > 1.0

    def test_calcium_non_negative_without_drive(self):
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(0.0)
        assert n.ca >= 0.0

    def test_gc_coupling_strength_reduces_gap(self):
        weak = PinskyRinzelNeuron(gc=0.5)
        strong = PinskyRinzelNeuron(gc=5.0)
        for _ in range(10000):
            weak.step(20.0)
            strong.step(20.0)
        assert abs(strong.v_s - strong.v_d) < abs(weak.v_s - weak.v_d)


# ---------------------------------------------------------------------------
# 3. f-I relation — repetitive firing then depolarisation block
# ---------------------------------------------------------------------------


class TestPinskyRinzelFI:
    def test_quiescent_near_rest(self):
        assert len(_run(PinskyRinzelNeuron(), current_soma=0.0, steps=50000)) <= 5

    @pytest.mark.parametrize("drive", [2.0, 5.0, 20.0])
    def test_low_drive_fires_repetitively(self, drive: float):
        assert len(_run(PinskyRinzelNeuron(), current_soma=drive, steps=50000)) >= 10

    def test_non_monotonic_depolarisation_block(self):
        low = len(_run(PinskyRinzelNeuron(), current_soma=5.0, steps=50000))
        high = len(_run(PinskyRinzelNeuron(), current_soma=200.0, steps=50000))
        assert low > high
        assert high <= 5


# ---------------------------------------------------------------------------
# 4. Spike-frequency adaptation
# ---------------------------------------------------------------------------


class TestPinskyRinzelAdaptation:
    def test_isis_lengthen_with_adaptation(self):
        spike_times = _run(PinskyRinzelNeuron(), current_soma=5.0, steps=50000)
        assert len(spike_times) >= 20
        isis = np.diff(spike_times)
        assert np.mean(isis[:5]) <= np.mean(isis[-5:])

    def test_isi_coefficient_of_variation_bounded(self):
        spike_times = _run(PinskyRinzelNeuron(), current_soma=5.0, steps=50000)
        isis = np.diff(spike_times[10:]).astype(float)
        assert np.std(isis) / np.mean(isis) < 0.2


# ---------------------------------------------------------------------------
# 5. Gating variables
# ---------------------------------------------------------------------------


class TestPinskyRinzelGating:
    def test_gating_variables_bounded(self):
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(50.0)
        for name, value in (("h", n.h), ("n", n.n), ("s", n.s), ("c", n.c), ("q", n.q)):
            assert 0.0 <= value <= 1.0, f"{name} = {value}"

    def test_sodium_inactivates_at_high_drive(self):
        n = PinskyRinzelNeuron()
        h_initial = n.h
        for _ in range(50000):
            n.step(100.0)
        assert n.h < h_initial


# ---------------------------------------------------------------------------
# 6. Rate-function limit branches (removable singularities) and αc regime
# ---------------------------------------------------------------------------


class TestPinskyRinzelRateBranches:
    @pytest.mark.parametrize("v_s", [-46.9, -19.9, -24.9])
    def test_somatic_rate_singularities_are_finite(self, v_s: float):
        """αm/βm/αn evaluate their removable limit at the singular voltage."""
        n = PinskyRinzelNeuron(v_s=v_s)
        n.step(0.0)
        assert np.isfinite(n.v_s)

    def test_dendritic_beta_s_singularity_is_finite(self):
        n = PinskyRinzelNeuron(v_d=-8.9)
        n.step(0.0)
        assert np.isfinite(n.v_d)

    def test_depolarised_dendrite_uses_alternate_c_branch(self):
        """Vd > −10 mV selects the βc = 0 branch of the K-C activation rate."""
        n = PinskyRinzelNeuron(v_d=0.0)
        n.step(0.0)
        assert np.isfinite(n.v_d)


# ---------------------------------------------------------------------------
# 7. Fail-closed safety contracts
# ---------------------------------------------------------------------------


class TestPinskyRinzelSafety:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dt", 0.0),
            ("cm", 0.0),
            ("gc", 0.0),
            ("p", 0.0),
            ("p", 1.0),
            ("g_na", 0.0),
            ("g_kdr", 0.0),
            ("g_ca", 0.0),
            ("g_kahp", 0.0),
            ("g_kc", 0.0),
            ("g_l", 0.0),
            ("h", -0.01),
            ("n", 1.01),
            ("s", float("nan")),
            ("c", 1.01),
            ("q", 1.01),
            ("ca", -0.01),
        ],
    )
    def test_rejects_invalid_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            PinskyRinzelNeuron(**{field: value})

    def test_rejects_runtime_parameter_corruption_before_mutation(self):
        n = PinskyRinzelNeuron()
        n.p = 1.0
        before = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca)
        with pytest.raises(ValueError):
            n.step(30.0)
        assert (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca) == before

    def test_rejects_non_finite_input_before_mutation(self):
        n = PinskyRinzelNeuron()
        before = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca)
        with pytest.raises(ValueError):
            n.step(float("nan"))
        assert (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca) == before
        with pytest.raises(ValueError):
            n.step(0.0, float("inf"))
        assert (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca) == before

    def test_extreme_timestep_fails_closed(self):
        with pytest.raises(FloatingPointError):
            PinskyRinzelNeuron(dt=10.0).step(30.0)

    def test_validate_candidate_rejects_non_finite_state(self):
        with pytest.raises(FloatingPointError):
            PinskyRinzelNeuron._validate_candidate(
                (float("nan"), -60.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2)
            )

    def test_validate_candidate_clamps_gates_and_calcium(self):
        v_s, v_d, h, n, s, c, q, ca = PinskyRinzelNeuron._validate_candidate(
            (-60.0, -60.0, 1.5, -0.2, 0.5, 2.0, -0.3, -4.0)
        )
        assert (h, n, s, c, q, ca) == (1.0, 0.0, 0.5, 1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 8. Determinism and time-step stability
# ---------------------------------------------------------------------------


class TestPinskyRinzelNumerics:
    def test_bit_exact_reproducibility(self):
        def trace() -> list[tuple[int, float]]:
            n = PinskyRinzelNeuron()
            return [(n.step(30.0), n.v_s) for _ in range(500)]

        assert trace() == trace()

    @pytest.mark.parametrize("dt", [0.01, 0.02, 0.05])
    def test_dt_stability(self, dt: float):
        n = PinskyRinzelNeuron(dt=dt)
        for _ in range(20000):
            n.step(30.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)


# ---------------------------------------------------------------------------
# 9. Network wiring and analysis
# ---------------------------------------------------------------------------


class TestPinskyRinzelNetwork:
    def test_population(self):
        pop = Population(PinskyRinzelNeuron, n=5, label="pr")
        assert pop.n == 5

    def test_network_spikes(self):
        pop = Population(PinskyRinzelNeuron, n=5, label="pr")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestPinskyRinzelAnalysis:
    def test_spike_count_matches_train_sum(self):
        n = PinskyRinzelNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(50000)])
        assert spike_count(train) >= 5
        assert spike_count(train) == int(train.sum())
