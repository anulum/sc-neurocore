# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MPR result-validation contracts

"""Malformed native-result rejection contracts for MPR dispatch."""

from .ermentrout_kopell_pop_dispatch_support import *


@pytest.mark.parametrize("key", ("r", "v"))
def test_result_validator_rejects_wrong_trace_shape(key: str) -> None:
    result: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    result[key] = np.zeros(1)
    with pytest.raises(FloatingPointError, match="malformed"):
        backends.normalise_result(result, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_rejects_nonfinite_or_negative_rate_trace() -> None:
    nonfinite: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    nonfinite["v"] = np.asarray([0.0, np.inf])
    with pytest.raises(FloatingPointError, match="non-finite v"):
        backends.normalise_result(nonfinite, n_steps=2, initial=_PARAMETERS[:2])

    negative: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    negative["r"] = np.asarray([0.1, -0.1])
    with pytest.raises(FloatingPointError, match="negative firing-rate"):
        backends.normalise_result(negative, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_rejects_missing_trace() -> None:
    result: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    del result["r"]
    with pytest.raises(FloatingPointError, match="invalid r trace"):
        backends.normalise_result(result, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_enforces_final_trace_consistency() -> None:
    result: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    result["v_final"] = float(result["v_final"]) + 1.0
    with pytest.raises(FloatingPointError, match="v_final disagrees"):
        backends.normalise_result(result, n_steps=2, initial=_PARAMETERS[:2])


def test_result_validator_rejects_missing_or_nonfinite_final_state() -> None:
    missing: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    del missing["r_final"]
    with pytest.raises(FloatingPointError, match="invalid r_final"):
        backends.normalise_result(missing, n_steps=2, initial=_PARAMETERS[:2])

    nonfinite: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    nonfinite["v_final"] = np.inf
    with pytest.raises(FloatingPointError, match="non-finite v_final"):
        backends.normalise_result(nonfinite, n_steps=2, initial=_PARAMETERS[:2])

    negative: dict[str, Any] = dict(backends.simulate_python(*_PARAMETERS, np.asarray([1.0, 2.0])))
    negative["r_final"] = -0.1
    with pytest.raises(FloatingPointError, match="negative final firing rate"):
        backends.normalise_result(negative, n_steps=2, initial=_PARAMETERS[:2])
