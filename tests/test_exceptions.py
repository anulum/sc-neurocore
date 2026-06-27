# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for sc_neurocore.exceptions

"""Tests for sc_neurocore.exceptions."""

import pytest

from sc_neurocore.exceptions import (
    BitstreamOverflowError,
    BitwidthMismatchError,
    CoverageGateError,
    HardwareSimMismatchError,
    IRCompilationError,
    SCCompilerError,
    SCConfigError,
    SCDependencyError,
    SCEncodingError,
    SCHardwareError,
    SCNeuroError,
    SCWeightError,
    SeedCollisionError,
)

ExceptionClass = type[SCNeuroError]


@pytest.mark.parametrize(
    "exc_cls",
    [
        BitstreamOverflowError,
        SeedCollisionError,
        BitwidthMismatchError,
        CoverageGateError,
        HardwareSimMismatchError,
        IRCompilationError,
    ],
)
def test_subclass_of_base(exc_cls: ExceptionClass) -> None:
    """All leaf exceptions inherit from the SC-NeuroCore base class."""
    assert issubclass(exc_cls, SCNeuroError)


def test_raise_and_catch() -> None:
    """The base exception catches concrete SC-NeuroCore failures."""
    with pytest.raises(SCNeuroError, match="overflow"):
        raise BitstreamOverflowError("overflow")


# ---------------------------------------------------------------------------
# Broad-exception mix-in contract (closes the test gap noted in
# exceptions.md §6 audit row 2; also exercises task #36's
# "reserved for future use" marker classes so they're at least
# constructable + catchable).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_cls",
    [SCEncodingError, SCConfigError, SCWeightError, SCCompilerError],
)
def test_value_error_mixin(exc_cls: ExceptionClass) -> None:
    """Domain exceptions are also ValueError so legacy callers keep working."""
    assert issubclass(exc_cls, ValueError)
    assert issubclass(exc_cls, SCNeuroError)
    # Catchable through both routes
    with pytest.raises(ValueError, match="probe"):
        raise exc_cls("probe")
    with pytest.raises(SCNeuroError, match="probe"):
        raise exc_cls("probe")


@pytest.mark.parametrize(
    "exc_cls",
    [SCDependencyError, SCHardwareError],
)
def test_runtime_error_mixin(exc_cls: ExceptionClass) -> None:
    """Runtime exceptions are also RuntimeError so legacy callers keep working."""
    assert issubclass(exc_cls, RuntimeError)
    assert issubclass(exc_cls, SCNeuroError)
    with pytest.raises(RuntimeError, match="probe"):
        raise exc_cls("probe")
    with pytest.raises(SCNeuroError, match="probe"):
        raise exc_cls("probe")


@pytest.mark.parametrize(
    "exc_cls",
    [
        # 6 leaf "reserved for future use" classes plus the 3 broad
        # ones whose messages are not asserted above
        SCConfigError,
        SCWeightError,
        BitstreamOverflowError,
        SeedCollisionError,
        BitwidthMismatchError,
        CoverageGateError,
        HardwareSimMismatchError,
        IRCompilationError,
    ],
)
def test_reserved_classes_are_constructable_and_catchable(exc_cls: ExceptionClass) -> None:
    """The 'reserved for future use' classes (task #36) at least work."""
    instance = exc_cls("probe")
    assert "probe" in str(instance)
    with pytest.raises(SCNeuroError):
        raise exc_cls("probe")
