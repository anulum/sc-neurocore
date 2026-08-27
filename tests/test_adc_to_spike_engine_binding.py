# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ADC-to-spike engine-binding contracts

"""Installed-extension contracts for decimating ADC samples into rate codes."""

from __future__ import annotations

import importlib

import numpy as np
import numpy.typing as npt
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _samples() -> npt.NDArray[np.int64]:
    return np.asarray([0, 64, 128, 255, 255, 128, 64, 0], dtype=np.int64)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_adc_to_spike_windows

    assert function.__name__ == "py_adc_to_spike_windows"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(samples, adc_width, q_int, q_frac, decimation, signed_input, threshold_q)"
    )
    assert engine.py_adc_to_spike_windows is function


def test_direct_binding_preserves_value_shape_and_dtype() -> None:
    result = extension.py_adc_to_spike_windows(_samples(), 8, 4, 4, 4, 0, 16)

    np.testing.assert_array_equal(result["window_values_q"], [-16, -16])
    np.testing.assert_array_equal(result["spike_counts"], [1, 1])
    np.testing.assert_array_equal(result["polarities"], [True, True])
    assert result["window_values_q"].dtype == np.int32
    assert result["spike_counts"].dtype == np.int32
    assert result["polarities"].dtype == np.bool_


@pytest.mark.parametrize(
    ("samples", "adc_width", "decimation", "threshold_q", "message"),
    (
        (
            np.asarray([0, 1, 2], dtype=np.int64),
            8,
            4,
            16,
            "need at least decimation=4 samples, got 3",
        ),
        (_samples(), 1, 4, 16, "adc_width must be greater than one, got 1"),
        (_samples(), 8, 0, 16, "decimation must be positive, got 0"),
        (_samples(), 8, 4, 0, "threshold_q must be positive, got 0"),
    ),
)
def test_validation_errors_are_preserved(
    samples: npt.NDArray[np.int64],
    adc_width: int,
    decimation: int,
    threshold_q: int,
    message: str,
) -> None:
    with pytest.raises(ValueError) as captured:
        extension.py_adc_to_spike_windows(
            samples,
            adc_width,
            4,
            4,
            decimation,
            0,
            threshold_q,
        )
    assert str(captured.value) == message
