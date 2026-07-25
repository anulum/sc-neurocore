# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive-threshold accelerator result validation tests

"""Validate trace shape, finiteness, receipts, spike counts, and reset state."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel import adaptive_threshold_if as backends
from tests.adaptive_threshold_if_accel_dispatch_support import (
    _PARAMETERS,
    _baseline,
    _normalise,
    _spiking_baseline,
)


@pytest.mark.parametrize("key", ("v", "theta", "spikes"))
def test_result_validator_rejects_wrong_trace_shape(key: str) -> None:
    result = _baseline()
    result[key] = np.zeros(1)
    with pytest.raises(FloatingPointError, match="malformed"):
        _normalise(result, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_rejects_nonfinite_and_nonbinary_traces() -> None:
    nonfinite = _baseline()
    nonfinite["theta"] = np.asarray([0.0, np.inf])
    with pytest.raises(FloatingPointError, match="non-finite theta"):
        _normalise(nonfinite, n_steps=2, initial=_PARAMETERS[:2])

    nonbinary = _baseline()
    nonbinary["spikes"] = np.asarray([0.0, 0.5])
    with pytest.raises(FloatingPointError, match="non-binary"):
        _normalise(nonbinary, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_rejects_missing_trace_and_final_receipt() -> None:
    missing_trace = _baseline()
    del missing_trace["v"]
    with pytest.raises(FloatingPointError, match="invalid v trace"):
        _normalise(missing_trace, n_steps=2, initial=_PARAMETERS[:2])

    missing_final = _baseline()
    del missing_final["theta_final"]
    with pytest.raises(FloatingPointError, match="invalid theta_final"):
        _normalise(missing_final, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_enforces_final_trace_consistency() -> None:
    result = _baseline()
    result["v_final"] = float(result["v_final"]) + 1.0
    with pytest.raises(FloatingPointError, match="v_final disagrees"):
        _normalise(result, n_steps=2, initial=_PARAMETERS[:2])


@pytest.mark.parametrize("bad_count", (True, 1.0, np.nan, "0", -1))
def test_result_validator_requires_consistent_integral_spike_count(
    bad_count: object,
) -> None:
    result = _baseline()
    result["spike_count"] = bad_count
    with pytest.raises(FloatingPointError, match="spike_count"):
        _normalise(result, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_rejects_reset_receipt_drift() -> None:
    spiking_kwargs = {
        "v_reset": -65.0,
        "theta_rest": -50.0,
        "delta_theta": 5.0,
        "tau_theta": 50.0,
        "dt": 0.1,
    }
    wrong_v = _spiking_baseline()
    wrong_v["v"] = np.asarray([-60.0])
    wrong_v["v_final"] = -60.0
    with pytest.raises(FloatingPointError, match="v reset"):
        backends.normalise_result(
            wrong_v,
            n_steps=1,
            initial=(-50.5, -51.0),
            **spiking_kwargs,
        )

    wrong_theta = _spiking_baseline()
    wrong_theta["theta"] = np.asarray([-40.0])
    wrong_theta["theta_final"] = -40.0
    with pytest.raises(FloatingPointError, match="threshold shift"):
        backends.normalise_result(
            wrong_theta,
            n_steps=1,
            initial=(-50.5, -51.0),
            **spiking_kwargs,
        )
