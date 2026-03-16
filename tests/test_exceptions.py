# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for sc_neurocore.exceptions

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
