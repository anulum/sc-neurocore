# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestScalarValidation from former test_validation_contracts.py

"""Focused suite: TestScalarValidation from former test_validation_contracts.py."""

from __future__ import annotations

from tests.test_bioware.validation_contracts_support import *  # noqa: F403


class TestScalarValidation:
    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_require_finite_rejects_non_finite(self, value: float) -> None:
        with pytest.raises(ValueError, match="value must be finite"):
            require_finite(value, "value")

    def test_scalar_sign_and_integer_guards(self) -> None:
        with pytest.raises(ValueError, match="value must be >= 0"):
            require_nonnegative(-1.0, "value")
        with pytest.raises(ValueError, match="value must be > 0"):
            require_positive(0.0, "value")
        with pytest.raises(TypeError, match="value must be an integer"):
            require_nonnegative_int(cast(Any, True), "value")
        with pytest.raises(ValueError, match="value must be >= 0"):
            require_nonnegative_int(-1, "value")
        with pytest.raises(ValueError, match="value must be > 0"):
            require_positive_int(0, "value")

    @pytest.mark.parametrize("value", [True, np.bool_(False), "1.0"])
    def test_scalar_guards_reject_non_real_values(self, value: Any) -> None:
        with pytest.raises(TypeError, match="value must be a real number"):
            require_finite(cast(Any, value), "value")

    @pytest.mark.parametrize(
        ("matrix", "error"),
        [
            (cast(Any, [[1.0]]), TypeError),
            (np.array([["x"]], dtype=object), TypeError),
            (np.ones(2), ValueError),
            (np.empty((0, 1)), ValueError),
            (np.empty((1, 0)), ValueError),
            (np.array([[float("inf")]]), ValueError),
        ],
    )
    def test_voltage_matrix_rejects_invalid_inputs(
        self,
        matrix: Any,
        error: type[Exception],
    ) -> None:
        with pytest.raises(error):
            validate_voltage_matrix(matrix)

    def test_voltage_matrix_rejects_channel_mismatch(self) -> None:
        with pytest.raises(ValueError, match="expected 2"):
            validate_voltage_matrix(np.ones((2, 1)), expected_channels=2)

    @pytest.mark.parametrize(
        ("bitstream", "allow_empty", "error"),
        [
            (cast(Any, [0, 1]), False, TypeError),
            (np.ones((1, 2)), False, ValueError),
            (np.array([], dtype=np.uint8), False, ValueError),
            (np.array(["x"]), False, TypeError),
            (np.array([float("nan")]), False, ValueError),
            (np.array([0, 2]), False, ValueError),
        ],
    )
    def test_bitstream_rejects_invalid_inputs(
        self,
        bitstream: Any,
        allow_empty: bool,
        error: type[Exception],
    ) -> None:
        with pytest.raises(error):
            validate_binary_bitstream(bitstream, name="bits", allow_empty=allow_empty)
