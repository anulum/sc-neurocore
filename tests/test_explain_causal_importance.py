# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCausalImportance from former test_explain.py

"""Focused suite: TestCausalImportance from former test_explain.py."""

from __future__ import annotations

from tests.explain_support import *  # noqa: F403


class TestCausalImportance:
    def _run_fn(self, spikes):
        return spikes.sum(axis=0).astype(np.float64)

    def test_basic(self):
        spikes = _make_spikes()
        ci = CausalImportance(run_fn=self._run_fn)
        result = ci.explain(spikes, output_neuron=0)
        assert result.importance_map.shape == (1, 8)
        assert result.method == "causal_importance"

    def test_active_neuron_important(self):
        spikes = np.zeros((10, 4), dtype=np.int8)
        spikes[:, 0] = 1  # neuron 0 fires every step

        ci = CausalImportance(run_fn=self._run_fn)
        result = ci.explain(spikes, output_neuron=0)
        # Silencing neuron 0 causes biggest change
        assert result.importance_map[0, 0] >= result.importance_map[0, 1]

    def test_silent_neuron_unimportant(self):
        spikes = np.zeros((10, 4), dtype=np.int8)
        spikes[:, 0] = 1

        ci = CausalImportance(run_fn=self._run_fn)
        result = ci.explain(spikes, output_neuron=0)
        # Neuron 2 never fires, silencing it changes nothing
        assert result.importance_map[0, 2] == 0.0

    def test_scalar_output(self):
        def scalar_fn(s):
            return np.float64(s.sum())

        spikes = _make_spikes(N=4)
        ci = CausalImportance(run_fn=scalar_fn)
        result = ci.explain(spikes, output_neuron=0)
        assert result.importance_map.shape == (1, 4)
