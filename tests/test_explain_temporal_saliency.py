# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTemporalSaliency from former test_explain.py

"""Focused suite: TestTemporalSaliency from former test_explain.py."""

from __future__ import annotations

from typing import Any
import numpy as np
from sc_neurocore.explain import (
    TemporalSaliency,
)
from tests.explain_support import (
    _make_spikes,
)


class TestTemporalSaliency:
    """Temporal saliency of input spikes."""

    def _run_fn(self, spikes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return spikes.sum(axis=0).astype(np.float64)

    def test_basic(self) -> None:
        """The saliency map covers every step and neuron."""
        spikes = _make_spikes()
        sal = TemporalSaliency(run_fn=self._run_fn)
        result = sal.explain(spikes, output_neuron=0)
        assert result.importance_map.shape == (20, 8)
        assert result.method == "temporal_saliency"

    def test_important_spike_detected(self) -> None:
        """Removing the only spike of a neuron is salient."""
        spikes = np.zeros((10, 4), dtype=np.int8)
        spikes[5, 0] = 1  # Only spike for neuron 0

        def run_fn(s: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return s.sum(axis=0).astype(np.float64)

        sal = TemporalSaliency(run_fn=run_fn)
        result = sal.explain(spikes, output_neuron=0)
        # Removing the only spike for neuron 0 causes max change
        assert result.importance_map[5, 0] > 0

    def test_no_spikes(self) -> None:
        """Without spikes nothing is salient."""
        spikes = np.zeros((10, 4), dtype=np.int8)
        sal = TemporalSaliency(run_fn=lambda s: s.sum(axis=0).astype(np.float64))
        result = sal.explain(spikes, output_neuron=0)
        assert result.importance_map.max() == 0.0

    def test_scalar_output(self) -> None:
        """A scalar output still gives a map per step."""

        def scalar_fn(s: np.ndarray[Any, Any]) -> Any:
            return np.float64(s.sum())

        spikes = _make_spikes(N=4)
        sal = TemporalSaliency(run_fn=scalar_fn)
        result = sal.explain(spikes, output_neuron=0)
        assert result.importance_map.shape[0] == 20
