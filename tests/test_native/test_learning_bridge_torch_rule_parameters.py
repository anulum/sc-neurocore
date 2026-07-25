# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch learning rule-parameter contracts

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sc_neurocore._native.learning_torch_support import rule_parameters
from sc_neurocore._native.learning_validation import (
    RULE_BCM,
    RULE_ELIGENT,
    RULE_REWARD_STDP,
    RULE_STDP,
)


@pytest.mark.parametrize("rule_type", [RULE_STDP, RULE_REWARD_STDP, RULE_BCM, RULE_ELIGENT])
def test_rule_parameter_defaults_are_finite(rule_type: int) -> None:
    parameters = rule_parameters(rule_type, 0.01, 0.012, {})
    assert len(parameters) == 5
    assert np.all(np.isfinite(parameters))
    assert parameters[2] > 0.0 and parameters[3] > 0.0 and parameters[4] > 0.0


def test_rule_parameter_overrides_and_common_tau() -> None:
    common = rule_parameters(RULE_STDP, 0.1, 0.2, {"tau": 4.0, "param_a_minus": 0.3})
    split = rule_parameters(
        RULE_REWARD_STDP,
        0.1,
        0.2,
        {"tau_plus": 5.0, "tau_minus": 6.0, "tau_e": 7.0},
    )
    assert common == pytest.approx([0.1, 0.3, 4.0, 4.0, 1.0])
    assert split == pytest.approx([0.1, 0.05, 5.0, 6.0, 7.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"param_a_minus": -1.0},
        {"tau": 0.0},
        {"tau_plus": float("nan")},
        {"tau_minus": -1.0},
        {"tau_e": 0.0},
    ],
)
def test_rule_parameter_overrides_reject_invalid_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        rule_parameters(RULE_STDP, 0.1, 0.2, kwargs)
