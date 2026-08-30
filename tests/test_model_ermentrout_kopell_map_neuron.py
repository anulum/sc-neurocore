# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific test: ErmentroutKopellMapNeuron

from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.models.ermentrout_kopell_map_neuron import (
    ErmentroutKopellMapNeuron,
)

_COSIM_EVIDENCE = (
    "tests/test_cosim_ermentrout_kopell_map_neuron.py::"
    "test_q1616_class_correct_spike_count_and_circular_phase_bound"
)


def test_descriptor_silicon_evidence_resolves_to_dedicated_cosim() -> None:
    """Keep the H2 claim bound to a live model-specific pytest node."""

    descriptor = load_descriptor("ErmentroutKopellMapNeuron")
    assert descriptor is not None
    assert descriptor.silicon.cosim_evidence == _COSIM_EVIDENCE

    relative_path, node_name = _COSIM_EVIDENCE.split("::", maxsplit=1)
    assert (Path(__file__).resolve().parents[1] / relative_path).is_file()
    module_name = relative_path.removesuffix(".py").replace("/", ".")
    evidence_module = importlib.import_module(module_name)
    assert callable(getattr(evidence_module, node_name, None))


class TestErmentroutKopellMapNeuron:
    def test_defaults_and_binary_step(self) -> None:
        n = ErmentroutKopellMapNeuron()
        assert n.theta == 0.0
        assert n.theta_threshold == math.pi
        assert n.step(0.0) in (0, 1)

    def test_positive_current_advances_phase_on_circle(self) -> None:
        n = ErmentroutKopellMapNeuron()
        n.step(1.0)
        assert 0.0 <= n.theta < 2.0 * math.pi
        assert n.theta > 0.0

    def test_phase_wrap_uses_circle_geometry(self) -> None:
        n = ErmentroutKopellMapNeuron(theta=2.0 * math.pi - 0.01, dt=1.0)
        n.step(1.0)
        assert 0.0 <= n.theta < 2.0 * math.pi

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("theta", np.nan),
            ("dt", 0.0),
            ("gain", np.inf),
            ("theta_threshold", np.nan),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            ErmentroutKopellMapNeuron(**{field: value})

    def test_rejects_non_finite_current_before_state_mutation(self) -> None:
        n = ErmentroutKopellMapNeuron()
        before = n.theta
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert n.theta == before

    def test_rejects_corrupted_runtime_state_before_mutation(self) -> None:
        n = ErmentroutKopellMapNeuron()
        n.theta = np.inf
        before = n.theta
        with pytest.raises(FloatingPointError, match="phase state"):
            n.step(1.0)
        assert n.theta == before

    def test_rejects_non_finite_candidate_before_state_mutation(self) -> None:
        n = ErmentroutKopellMapNeuron(gain=1.0e308)
        before = n.theta
        with pytest.raises(FloatingPointError, match="input drive"):
            n.step(1.0e308)
        assert n.theta == before

    def test_spike_detects_upward_threshold_crossing(self) -> None:
        n = ErmentroutKopellMapNeuron(theta=math.pi - 0.01, dt=1.0)
        assert n.step(1.0) == 1
