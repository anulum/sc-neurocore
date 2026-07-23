# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSimulate from former test_autofit.py

"""Focused suite: TestSimulate from former test_autofit.py."""

from __future__ import annotations

from tests.autofit_support import *  # noqa: F403

class TestSimulate:
    def test_with_mock_model(self):
        class MockNeuron:
            def __init__(self):
                self.v = 0.0
                self.dt = 1.0

            def step(self, current):
                self.v += current * 0.1

        v = _simulate(MockNeuron, {}, np.ones(10), dt=0.1)
        assert len(v) == 10
        assert v[-1] > 0

    def test_model_with_params(self):
        class MockNeuron:
            def __init__(self, gain=1.0):
                self.v = 0.0
                self.gain = gain

            def step(self, current):
                self.v = current * self.gain

        v = _simulate(MockNeuron, {"gain": 2.0}, np.ones(5), dt=0.1)
        assert v[-1] == pytest.approx(2.0)

    def test_model_exception_handling(self):
        class BrokenNeuron:
            def __init__(self):
                self.v = 0.0

            def step(self, current):
                raise RuntimeError("boom")

        v = _simulate(BrokenNeuron, {}, np.ones(5), dt=0.1)
        assert len(v) == 5

    def test_model_init_fallback(self):
        class FussyNeuron:
            def __init__(self, required_param=None):
                if required_param == "bad":
                    raise TypeError("bad param")
                self.v = 0.0

            def step(self, current):
                self.v = current

        v = _simulate(FussyNeuron, {"required_param": "bad"}, np.ones(3), dt=0.1)
        assert len(v) == 3
