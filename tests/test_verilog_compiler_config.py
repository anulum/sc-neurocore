# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog compiler fixed-point configuration tests

"""Public-contract tests for equation-compiler fixed-point configuration."""

from collections.abc import Callable
from typing import cast

import pytest

from sc_neurocore.compiler.equation_compiler import Q88

ConfigFactory = Callable[[], Q88]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Q88(data_width=cast(int, "16")), "data_width must be an integer"),
        (lambda: Q88(fraction=cast(int, "8")), "fraction must be an integer"),
        (lambda: Q88(signed=cast(bool, 1)), "signed must be a boolean"),
        (lambda: Q88(overflow=cast(str, 1)), "overflow must be a string"),
        (lambda: Q88(rounding=cast(str, 1)), "rounding must be a string"),
    ],
)
def test_q_format_rejects_wrong_runtime_field_types(factory: ConfigFactory, message: str) -> None:
    """Runtime validation rejects booleans, strings, and integers in the wrong fields."""
    with pytest.raises(TypeError, match=message):
        factory()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Q88(data_width=0), "data_width must be positive"),
        (lambda: Q88(fraction=-1), "0 <= fraction"),
        (lambda: Q88(data_width=8, fraction=9), "0 <= fraction <= 8"),
    ],
)
def test_q_format_rejects_impossible_geometry(factory: ConfigFactory, message: str) -> None:
    """Impossible bit layouts fail closed before arithmetic is attempted."""
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize("signed", [True, False])
def test_all_fractional_word_is_valid(signed: bool) -> None:
    """Signed and unsigned words may devote every encoded bit to the fraction."""
    q = Q88(data_width=8, fraction=8, signed=signed)

    assert q.integer_bits == 0
    if signed:
        assert q.min_value == -0.5
        assert q.max_value == 127.0 / 256.0
    else:
        assert q.min_value == 0.0
        assert q.max_value == 255.0 / 256.0


def test_q_format_ranges_and_raw_words_follow_signedness() -> None:
    """Signed and unsigned configurations expose their exact word intervals."""
    signed = Q88()
    unsigned = Q88(signed=False)

    assert signed.integer_bits == 7
    assert signed.min_value == -128.0
    assert signed.max_value == 127.99609375
    assert signed.resolution == 0.00390625
    assert signed.encode(-1.0) == 65280

    assert unsigned.integer_bits == 8
    assert unsigned.min_value == 0.0
    assert unsigned.max_value == 255.99609375
    assert unsigned.encode(unsigned.max_value) == 65535


def test_signed_literal_encoding_rejects_unsigned_configuration() -> None:
    """Literal emission preserves signed words and fails closed for UQ formats."""
    signed = Q88()

    assert signed.encode_signed_literal(1.0) == "16'sd256"
    assert signed.encode_signed_literal(-1.0) == "16'sd65280"
    with pytest.raises(ValueError, match="signed=True"):
        Q88(signed=False).encode_signed_literal(1.0)


def test_range_diagnostics_name_the_actual_format() -> None:
    """Range messages distinguish Q from UQ and omit in-range values."""
    signed = Q88()
    unsigned = Q88(signed=False)

    assert signed.check_range(0.0, "state") == []
    assert "exceeds Q8.8" in signed.check_range(128.0, "state")[0]
    assert "below Q8.8" in signed.check_range(-129.0, "state")[0]
    assert "below UQ8.8" in unsigned.check_range(-1.0, "state")[0]


def test_precision_report_covers_quantised_parameters_and_overflow() -> None:
    """The report exposes timestep error, parameter codes, and range warnings."""
    report = Q88().precision_report(
        dt=1.0 / 256.0,
        params={"zero": 0.0, "too_high": 128.0},
    )

    assert "Fixed-point format: Q8.8 (16-bit signed)" in report
    assert "dt=0.00390625 → Q-value=1" in report
    assert "zero=0.0 → Q-value=0 (error=0.0%)" in report
    assert "too_high=128.0 → Q-value=32768 (error=0.0%)" in report
    assert "Overflow: too_high=128.0 exceeds Q8.8" in report


def test_unsigned_precision_report_marks_underflowing_step_and_value() -> None:
    """Unsigned reports label both a sub-LSB timestep and negative parameter."""
    report = Q88(signed=False).precision_report(dt=0.001, params={"offset": -1.0})

    assert "Fixed-point format: UQ8.8 (16-bit unsigned)" in report
    assert "error=100.0%) ✗ UNDERFLOW" in report
    assert "Underflow: offset=-1.0 below UQ8.8" in report


def test_zero_step_report_has_no_parameter_section() -> None:
    """A zero timestep remains exact when no parameter diagnostics are requested."""
    report = Q88().precision_report(dt=0.0)

    assert "dt=0.0 → Q-value=0 (actual=0.000000, error=0.0%) ✓ ZERO STEP" in report
    assert report.count("\n") == 3


def test_negative_representable_step_reports_non_negative_error() -> None:
    """Diagnostics use a magnitude denominator for a negative timestep."""
    report = Q88().precision_report(dt=-1.0 / 256.0)

    assert "dt=-0.00390625 → Q-value=-1" in report
    assert "error=0.0%) ✓" in report
