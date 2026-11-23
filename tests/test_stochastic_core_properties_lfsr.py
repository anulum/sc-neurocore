# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (lfsr) from former test_stochastic_core_properties.py

"""Property-based checks of the LFSR stream source."""

from __future__ import annotations

from hypothesis import (
    given,
    settings,
    strategies as st,
)
from sc_neurocore import (
    FixedPointLFSR,
)


@given(seed=st.integers(min_value=1, max_value=0xFFFF))
@settings(max_examples=30)
def test_lfsr_nonzero_output(seed: int) -> None:
    """Any non-zero seed gives a non-zero sequence."""
    lfsr = FixedPointLFSR(seed=seed)
    vals = [lfsr.step() for _ in range(100)]
    assert any(v != 0 for v in vals)


@given(seed=st.integers(min_value=1, max_value=0xFFFF))
@settings(max_examples=20)
def test_lfsr_deterministic(seed: int) -> None:
    """The same seed gives the same sequence."""
    a = FixedPointLFSR(seed=seed)
    b = FixedPointLFSR(seed=seed)
    assert [a.step() for _ in range(50)] == [b.step() for _ in range(50)]
