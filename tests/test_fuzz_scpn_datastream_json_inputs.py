# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Property-based fuzz tests for SCPN datastream JSON

"""Property-based fuzz tests for SCPN datastream JSON payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sc_neurocore.scpn import SCPNDatastream, generate_scpn_datastream_payload

_REQUIRED_FIELDS = ("dt_s", "seed", "probabilities", "spike_train", "omega_rad_s", "knm")
_ARRAY_FIELDS = ("probabilities", "spike_train", "omega_rad_s", "knm")
_JSON_SCALAR = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-10, max_value=10)
    | st.floats(allow_nan=False, allow_infinity=False, width=32)
    | st.text(max_size=12)
)
_JSON_VALUE = st.recursive(
    _JSON_SCALAR,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=24,
)
_BAD_ARRAY_VALUE = (
    st.none()
    | st.booleans()
    | st.text(max_size=12)
    | st.dictionaries(st.text(max_size=8), _JSON_SCALAR, max_size=3)
    | st.lists(st.text(max_size=8) | st.booleans() | st.none(), min_size=1, max_size=4)
)


def _valid_payload() -> dict[str, Any]:
    return generate_scpn_datastream_payload(n_steps=3, seed=23)


@given(payload=_JSON_VALUE)
@settings(max_examples=120, deadline=None)
def test_fuzz_from_json_dict_rejects_malformed_payloads(payload: object) -> None:
    try:
        stream = SCPNDatastream.from_json_dict(cast(dict[str, Any], payload))
    except ValueError:
        return

    assert stream.n_layers == 16
    assert stream.probabilities.shape == stream.spike_train.shape


@given(missing=st.sampled_from(_REQUIRED_FIELDS))
@settings(max_examples=24, deadline=None)
def test_fuzz_from_json_dict_rejects_missing_required_fields(missing: str) -> None:
    payload = _valid_payload()
    del payload[missing]

    with pytest.raises(ValueError, match="missing required fields"):
        SCPNDatastream.from_json_dict(payload)


@given(field=st.sampled_from(_ARRAY_FIELDS), value=_BAD_ARRAY_VALUE)
@settings(max_examples=120, deadline=None)
def test_fuzz_from_json_dict_rejects_non_numeric_arrays(field: str, value: object) -> None:
    payload = _valid_payload()
    payload[field] = value

    with pytest.raises(ValueError):
        SCPNDatastream.from_json_dict(payload)


@given(value=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True))
@settings(max_examples=60, deadline=None)
def test_fuzz_from_json_dict_rejects_fractional_spike_values(value: float) -> None:
    payload = _valid_payload()
    payload["spike_train"] = deepcopy(payload["spike_train"])
    payload["spike_train"][0][0] = value

    with pytest.raises(ValueError, match="binary"):
        SCPNDatastream.from_json_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dt_s", "0.01", "numeric"),
        ("dt_s", np.inf, "finite"),
        ("seed", 1.5, "integer"),
        ("seed", True, "integer"),
    ],
)
def test_from_json_dict_rejects_bad_scalar_fields(field: str, value: object, message: str) -> None:
    payload = _valid_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        SCPNDatastream.from_json_dict(payload)


@pytest.mark.parametrize("field", ["probabilities", "omega_rad_s", "knm"])
def test_from_json_dict_rejects_nonfinite_numeric_arrays(field: str) -> None:
    payload = _valid_payload()
    payload[field] = deepcopy(payload[field])
    if field == "omega_rad_s":
        payload[field][0] = np.nan
    else:
        payload[field][0][0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        SCPNDatastream.from_json_dict(payload)
