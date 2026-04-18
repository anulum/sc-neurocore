# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for reservoir computing module

import numpy as np

from sc_neurocore.reservoir import AutoCriticalReservoir, ReservoirMetrics


class TestReservoirConstruction:
    def test_default_construction(self):
        res = AutoCriticalReservoir(n_inputs=2, n_neurons=50, seed=0)
        assert res.n_neurons == 50
        assert res.n_inputs == 2
        assert res.W_res.shape == (50, 50)
        assert res.W_in.shape == (50, 2)
        assert res.W_out.shape == (10, 50)

    def test_no_self_connections(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=20, seed=0)
        np.testing.assert_array_equal(np.diag(res.W_res), 0.0)

    def test_sparsity(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=100, connectivity=0.1, seed=0)
        nonzero_frac = np.count_nonzero(res.W_res) / (100 * 100)
        assert 0.05 < nonzero_frac < 0.15

    def test_critical_weight_formula(self):
        res = AutoCriticalReservoir(
            n_inputs=1,
            n_neurons=100,
            threshold=1.0,
            leak=0.1,
            connectivity=0.1,
            seed=0,
        )
        expected = 1.0 / (2.0 * 0.1 * 100 * 0.1)
        assert abs(res.w_critical - expected) < 1e-10

    def test_spectral_radius_finite(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=30, seed=0)
        sr = res.spectral_radius
        assert np.isfinite(sr)
        assert sr > 0

    def test_deterministic_seed(self):
        a = AutoCriticalReservoir(n_inputs=2, n_neurons=30, seed=99)
        b = AutoCriticalReservoir(n_inputs=2, n_neurons=30, seed=99)
        np.testing.assert_array_equal(a.W_res, b.W_res)
        np.testing.assert_array_equal(a.W_in, b.W_in)


class TestReservoirDynamics:
    def test_step_output_shape(self):
        res = AutoCriticalReservoir(n_inputs=3, n_neurons=50, seed=0)
        out = res.step(np.array([1.0, 0.0, -1.0]))
        assert out.shape == (50,)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_run_output_shape(self):
        res = AutoCriticalReservoir(n_inputs=2, n_neurons=50, seed=0)
        inputs = np.random.randn(20, 2)
        states = res.run(inputs)
        assert states.shape == (20, 50)

    def test_reset_clears_state(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=20, seed=0)
        res.step(np.array([5.0]))
        res.reset()
        assert np.all(res._v == 0)
        assert np.all(res._spikes == 0)

    def test_run_produces_spikes(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=100, seed=0)
        inputs = np.ones((50, 1)) * 2.0
        states = res.run(inputs)
        assert states.sum() > 0

    def test_different_inputs_different_states(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=50, seed=0)
        s1 = res.run(np.ones((10, 1)) * 2.0)
        s2 = res.run(np.ones((10, 1)) * -2.0)
        assert not np.array_equal(s1, s2)


class TestReadout:
    def test_fit_and_predict_shape(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=50, n_outputs=1, seed=0)
        inputs = np.random.randn(100, 1)
        states = res.run(inputs)
        targets = np.sin(np.arange(100)).reshape(-1, 1)
        res.fit_readout(states, targets)
        preds = res.predict(states)
        assert preds.shape == (100, 1)

    def test_train_and_predict_pipeline(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=50, n_outputs=1, seed=0)
        train_in = np.random.randn(80, 1)
        train_tgt = np.sin(np.arange(80)).reshape(-1, 1)
        test_in = np.random.randn(20, 1)
        preds = res.train_and_predict(train_in, train_tgt, test_in)
        assert preds.shape == (20, 1)
        assert np.all(np.isfinite(preds))

    def test_readout_shapes_with_multiple_outputs(self):
        res = AutoCriticalReservoir(n_inputs=2, n_neurons=50, n_outputs=5, seed=0)
        states = np.random.randn(40, 50)
        targets = np.random.randn(40, 5)
        res.fit_readout(states, targets)
        assert res.W_out.shape == (5, 50)
        preds = res.predict(states)
        assert preds.shape == (40, 5)


class TestMetrics:
    def test_metrics_returns_all_fields(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=50, seed=0)
        inputs = np.random.randn(30, 1)
        m = res.metrics(inputs)
        assert isinstance(m, ReservoirMetrics)
        assert 0.0 <= m.firing_fraction <= 1.0
        assert m.criticality_error >= 0.0
        assert 0.0 <= m.kernel_quality <= 1.0
        assert m.spectral_radius > 0

    def test_metrics_summary_string(self):
        m = ReservoirMetrics(
            firing_fraction=0.48,
            criticality_error=0.02,
            kernel_quality=0.95,
            spectral_radius=1.1,
        )
        s = m.summary()
        assert "firing=0.480" in s
        assert "spectral_r=1.100" in s
