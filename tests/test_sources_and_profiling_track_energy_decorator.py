# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrackEnergyDecorator from former test_sources_and_profiling.py

"""Focused suite: TestTrackEnergyDecorator from former test_sources_and_profiling.py."""

from __future__ import annotations

from tests.sources_and_profiling_support import *  # noqa: F403


class TestTrackEnergyDecorator:
    def test_decorator_accumulates_ops(self):
        """track_energy should add AND ops for a layer-like object."""
        profiler.reset()

        class MockLayer:
            n_neurons = 4
            n_inputs = 3
            length = 100

            @track_energy
            def forward(self):
                return "done"

        layer = MockLayer()
        layer.forward()
        # ops = n_inputs * n_neurons * length = 3 * 4 * 100 = 1200
        assert profiler.total_ops_and == 1200
        # mem = (n_neurons * n_inputs * length) + (n_inputs * length) = 1200 + 300
        assert profiler.total_bits_mem == 1500

    def test_decorator_returns_original_result(self):
        profiler.reset()

        class MockLayer:
            n_neurons = 2
            n_inputs = 2
            length = 10

            @track_energy
            def forward(self):
                return 42

        layer = MockLayer()
        assert layer.forward() == 42

    def test_decorator_no_layer_attributes(self):
        """Decorator should still work if the object lacks layer attributes."""
        profiler.reset()

        @track_energy
        def simple_func():
            return "ok"

        assert simple_func() == "ok"
        assert profiler.total_ops_and == 0  # nothing accumulated
