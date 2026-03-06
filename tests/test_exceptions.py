# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for sc_neurocore.exceptions."""

import pytest

from sc_neurocore.exceptions import (
    SCNeuroError,
    BitstreamOverflowError,
    SeedCollisionError,
    BitwidthMismatchError,
    CoverageGateError,
    HardwareSimMismatchError,
    IRCompilationError,
)


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
def test_subclass_of_base(exc_cls):
    assert issubclass(exc_cls, SCNeuroError)


def test_raise_and_catch():
    with pytest.raises(SCNeuroError, match="overflow"):
        raise BitstreamOverflowError("overflow")
