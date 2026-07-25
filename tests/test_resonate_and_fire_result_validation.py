# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-fire result validation contracts

"""Verify native result shapes, values, receipts, counts, and reset invariants."""

import numpy as np
import pytest

from sc_neurocore.accel import resonate_and_fire as backends
from tests.resonate_and_fire_accel_dispatch_support import (
    _PARAMETERS,
    _baseline,
    _spiking_baseline,
)


@pytest.mark.parametrize("key", ("x", "y", "spikes"))
def test_result_validator_rejects_wrong_trace_shape(key: str) -> None:
    result = _baseline()
    result[key] = np.zeros(1)
    with pytest.raises(FloatingPointError, match="malformed"):
        backends.normalise_result(
            result, n_steps=2, initial=_PARAMETERS[:2], threshold=_PARAMETERS[4]
        )


def test_result_validator_rejects_nonfinite_and_nonbinary_traces() -> None:
    nonfinite = _baseline()
    nonfinite["y"] = np.asarray([0.0, np.inf])
    with pytest.raises(FloatingPointError, match="non-finite y"):
        backends.normalise_result(
            nonfinite, n_steps=2, initial=_PARAMETERS[:2], threshold=_PARAMETERS[4]
        )

    nonbinary = _baseline()
    nonbinary["spikes"] = np.asarray([0.0, 0.5])
    with pytest.raises(FloatingPointError, match="non-binary"):
        backends.normalise_result(
            nonbinary, n_steps=2, initial=_PARAMETERS[:2], threshold=_PARAMETERS[4]
        )


def test_result_validator_rejects_missing_trace_and_final_receipt() -> None:
    missing_trace = _baseline()
    del missing_trace["x"]
    with pytest.raises(FloatingPointError, match="invalid x trace"):
        backends.normalise_result(
            missing_trace, n_steps=2, initial=_PARAMETERS[:2], threshold=_PARAMETERS[4]
        )

    missing_final = _baseline()
    del missing_final["y_final"]
    with pytest.raises(FloatingPointError, match="invalid y_final"):
        backends.normalise_result(
            missing_final, n_steps=2, initial=_PARAMETERS[:2], threshold=_PARAMETERS[4]
        )


def test_result_validator_enforces_final_trace_consistency() -> None:
    result = _baseline()
    result["x_final"] = float(result["x_final"]) + 1.0
    with pytest.raises(FloatingPointError, match="x_final disagrees"):
        backends.normalise_result(
            result, n_steps=2, initial=_PARAMETERS[:2], threshold=_PARAMETERS[4]
        )


@pytest.mark.parametrize("bad_count", (True, 1.0, np.nan, "0", -1))
def test_result_validator_requires_consistent_integral_spike_count(bad_count: object) -> None:
    result = _baseline()
    result["spike_count"] = bad_count
    with pytest.raises(FloatingPointError, match="spike_count"):
        backends.normalise_result(
            result, n_steps=2, initial=_PARAMETERS[:2], threshold=_PARAMETERS[4]
        )


def test_result_validator_rejects_reset_receipt_drift() -> None:
    wrong_x = _spiking_baseline()
    wrong_x["x"] = np.asarray([0.25])
    wrong_x["x_final"] = 0.25
    with pytest.raises(FloatingPointError, match="x reset"):
        backends.normalise_result(wrong_x, n_steps=1, initial=(0.0, 0.99), threshold=1.0)

    wrong_y = _spiking_baseline()
    wrong_y["y"] = np.asarray([0.75])
    wrong_y["y_final"] = 0.75
    with pytest.raises(FloatingPointError, match="y reset"):
        backends.normalise_result(wrong_y, n_steps=1, initial=(0.0, 0.99), threshold=1.0)
