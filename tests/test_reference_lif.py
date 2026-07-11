# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — LIF independent reference contract

"""Independent closed-form reference trace for LIF."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.reference_traces import load_reference_trace_spec
from tests.cosim_support import _closed_form_features


def test_lif_seed_features_match_independent_closed_form_solution() -> None:
    """Committed LIF features must match the closed-form RC solution, not the runner."""
    spec = load_reference_trace_spec("lif_constant_current_closed_form")

    expected = _closed_form_features(
        initial=-65.0,
        steady=-55.0,
        tau=10.0,
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)
