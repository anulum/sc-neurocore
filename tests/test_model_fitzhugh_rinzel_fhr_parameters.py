# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFHRParameters from former test_model_fitzhugh_rinzel.py

"""Focused suite: TestFHRParameters from former test_model_fitzhugh_rinzel.py."""

from __future__ import annotations

from tests.model_fitzhugh_rinzel_support import *  # noqa: F403


class TestFHRParameters:
    def test_mu_controls_y_speed(self):
        n_fast = FitzHughRinzelNeuron(mu=0.01)
        n_slow = FitzHughRinzelNeuron(mu=0.00001)
        for _ in range(5000):
            n_fast.step(0.5)
            n_slow.step(0.5)
        assert abs(n_fast.y) > abs(n_slow.y)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = FitzHughRinzelNeuron(dt=dt)
        for _ in range(10000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w) and np.isfinite(n.y)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = FitzHughRinzelNeuron()
            trace = [(n.step(0.5), n.v, n.w, n.y) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"v": math.nan}, "v.*finite"),
            ({"v": True}, "v.*finite"),
            ({"w": object()}, "w.*finite"),
            ({"b": 0.0}, "b.*positive"),
            ({"d": -1.0}, "d.*positive"),
            ({"delta": -0.1}, "delta.*positive"),
            ({"mu": 0.0}, "mu.*positive"),
            ({"dt": 0.0}, "dt.*positive"),
        ],
    )
    def test_rejects_invalid_numeric_configuration(self, kwargs: dict[str, float], match: str):
        with pytest.raises(ValueError, match=match):
            FitzHughRinzelNeuron(**kwargs)

    def test_rejects_nonfinite_current_without_mutation(self):
        neuron = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
        before = (neuron.v, neuron.w, neuron.y)

        with pytest.raises(ValueError, match="current"):
            neuron.step(float("nan"))

        assert (neuron.v, neuron.w, neuron.y) == before

    def test_rejects_corrupted_runtime_parameter_without_mutation(self):
        neuron = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
        before = (neuron.v, neuron.w, neuron.y)
        neuron.mu = float("nan")

        with pytest.raises(ValueError, match="mu.*finite"):
            neuron.step(0.5)

        assert (neuron.v, neuron.w, neuron.y) == before

    def test_rejects_nonpositive_runtime_parameter_without_mutation(self):
        neuron = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
        before = (neuron.v, neuron.w, neuron.y)
        neuron.d = 0.0

        with pytest.raises(ValueError, match="d.*positive"):
            neuron.step(0.5)

        assert (neuron.v, neuron.w, neuron.y) == before

    def test_rejects_overflow_candidate_without_mutation(self):
        # v = 1e155 makes the cube overflow to +inf; the exact `v*v*v` form
        # produces inf (rather than the libm-pow OverflowError) which the finite
        # guard rejects as a non-finite derivative — same contract, no mutation.
        neuron = FitzHughRinzelNeuron(v=1.0e155, w=0.2, y=0.1)
        before = (neuron.v, neuron.w, neuron.y)

        with pytest.raises(FloatingPointError, match="derivative"):
            neuron.step(0.5)

        assert (neuron.v, neuron.w, neuron.y) == before

    def test_rejects_nonfinite_derivative_without_mutation(self):
        neuron = FitzHughRinzelNeuron(mu=1.0e308)
        before = (neuron.v, neuron.w, neuron.y)

        with pytest.raises(FloatingPointError, match="derivative"):
            neuron.step(0.5)

        assert (neuron.v, neuron.w, neuron.y) == before

    def test_rejects_nonfinite_candidate_directly(self):
        with pytest.raises(FloatingPointError, match="candidate"):
            FitzHughRinzelNeuron._validate_candidate(math.nan, -0.5, 0.0)
