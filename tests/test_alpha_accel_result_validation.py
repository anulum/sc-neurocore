# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha accelerator result-validation contracts

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from tests.alpha_accel_dispatch_support import PARAMETERS, baseline, normalise


@pytest.mark.parametrize("key", ("v", "a_exc", "i_exc", "a_inh", "i_inh", "spikes"))
def test_result_validator_rejects_wrong_trace_shape(key: str) -> None:
    result = baseline()
    result[key] = np.zeros(1)
    with pytest.raises(FloatingPointError, match="malformed"):
        normalise(result, n_steps=2, initial=PARAMETERS[:5])


def test_result_validator_rejects_nonfinite_and_nonbinary_traces() -> None:
    nonfinite = baseline()
    nonfinite["i_exc"] = np.asarray([0.0, np.inf])
    with pytest.raises(FloatingPointError, match="non-finite i_exc"):
        normalise(nonfinite, n_steps=2, initial=PARAMETERS[:5])

    nonbinary = baseline()
    nonbinary["spikes"] = np.asarray([0.0, 0.5])
    with pytest.raises(FloatingPointError, match="non-binary"):
        normalise(nonbinary, n_steps=2, initial=PARAMETERS[:5])


def test_result_validator_rejects_missing_trace_and_final_receipt() -> None:
    missing_trace = baseline()
    del missing_trace["a_exc"]
    with pytest.raises(FloatingPointError, match="invalid a_exc trace"):
        normalise(missing_trace, n_steps=2, initial=PARAMETERS[:5])

    missing_final = baseline()
    del missing_final["i_inh_final"]
    with pytest.raises(FloatingPointError, match="invalid i_inh_final"):
        normalise(missing_final, n_steps=2, initial=PARAMETERS[:5])


def test_result_validator_enforces_final_trace_consistency() -> None:
    result = baseline()
    result["v_final"] = cast(float, result["v_final"]) + 1.0
    with pytest.raises(FloatingPointError, match="v_final disagrees"):
        normalise(result, n_steps=2, initial=PARAMETERS[:5])


def test_result_validator_rejects_nonfinite_final_and_missing_spike_count() -> None:
    nonfinite_final = baseline()
    nonfinite_final["a_inh_final"] = np.inf
    with pytest.raises(FloatingPointError, match="non-finite a_inh_final"):
        normalise(nonfinite_final, n_steps=2, initial=PARAMETERS[:5])

    missing_count = baseline()
    del missing_count["spike_count"]
    with pytest.raises(FloatingPointError, match="invalid spike_count"):
        normalise(missing_count, n_steps=2, initial=PARAMETERS[:5])


@pytest.mark.parametrize("bad_count", (True, 1.0, np.nan, "0", -1))
def test_result_validator_requires_consistent_integral_spike_count(
    bad_count: object,
) -> None:
    result = baseline()
    result["spike_count"] = bad_count
    with pytest.raises(FloatingPointError, match="spike_count"):
        normalise(result, n_steps=2, initial=PARAMETERS[:5])


def test_result_validator_rejects_reset_receipt_drift() -> None:
    wrong = baseline()
    wrong["spikes"] = np.asarray([1.0, 0.0])
    wrong["v"] = np.asarray([-0.3, -0.3])
    wrong["v_final"] = -0.3
    wrong["spike_count"] = 1
    with pytest.raises(FloatingPointError, match="somatic v reset"):
        normalise(wrong, n_steps=2, initial=PARAMETERS[:5])
