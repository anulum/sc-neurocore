# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReadout from former test_reservoir.py

"""Focused suite: TestReadout from former test_reservoir.py."""

from __future__ import annotations

from tests.reservoir_support import *  # noqa: F403

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
