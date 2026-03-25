# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Tests for sc_neurocore.reservoir
from __future__ import annotations
import numpy as np
from sc_neurocore.reservoir import AutoCriticalReservoir, ReservoirMetrics


class TestAutoCriticalReservoir:
    def test_init(self):
        r = AutoCriticalReservoir(n_inputs=4, n_neurons=100, n_outputs=2)
        assert r.W_res.shape == (100, 100)
        assert r.W_in.shape == (100, 4)
        assert r.w_critical > 0

    def test_step(self):
        r = AutoCriticalReservoir(n_inputs=4, n_neurons=50)
        state = r.step(np.random.rand(4))
        assert state.shape == (50,)

    def test_run(self):
        r = AutoCriticalReservoir(n_inputs=4, n_neurons=50)
        inputs = np.random.rand(100, 4)
        states = r.run(inputs)
        assert states.shape == (100, 50)

    def test_fit_readout(self):
        r = AutoCriticalReservoir(n_inputs=4, n_neurons=50, n_outputs=2)
        states = np.random.rand(100, 50)
        targets = np.random.rand(100, 2)
        r.fit_readout(states, targets)
        assert r.W_out.shape == (2, 50)

    def test_predict(self):
        r = AutoCriticalReservoir(n_inputs=4, n_neurons=50, n_outputs=2)
        states = np.random.rand(10, 50)
        r.W_out = np.random.randn(2, 50)
        preds = r.predict(states)
        assert preds.shape == (10, 2)

    def test_train_and_predict(self):
        r = AutoCriticalReservoir(n_inputs=4, n_neurons=100, n_outputs=2)
        train_x = np.random.rand(50, 4)
        train_y = np.random.rand(50, 2)
        test_x = np.random.rand(20, 4)
        preds = r.train_and_predict(train_x, train_y, test_x)
        assert preds.shape == (20, 2)

    def test_metrics(self):
        r = AutoCriticalReservoir(n_inputs=4, n_neurons=50)
        inputs = np.random.rand(50, 4)
        m = r.metrics(inputs)
        assert isinstance(m, ReservoirMetrics)
        assert 0 <= m.firing_fraction <= 1
        assert m.criticality_error >= 0
        s = m.summary()
        assert "Reservoir" in s

    def test_reset(self):
        r = AutoCriticalReservoir(n_inputs=4, n_neurons=50)
        r.step(np.ones(4))
        r.reset()
        assert np.allclose(r._v, 0)

    def test_no_self_connections(self):
        r = AutoCriticalReservoir(n_inputs=4, n_neurons=50)
        assert np.allclose(np.diag(r.W_res), 0)

    def test_spectral_radius(self):
        r = AutoCriticalReservoir(n_inputs=4, n_neurons=50)
        assert r.spectral_radius >= 0
