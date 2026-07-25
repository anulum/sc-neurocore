# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bitstream current-source validation contracts

"""Construction and mode validation for BitstreamCurrentSource."""

from tests.sources.bitstream_current_source_support import *


def test_source_init_mismatch_raises():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Mismatched input/weight lengths should raise ValueError."""
    with pytest.raises(ValueError):
        _ = BitstreamCurrentSource(
            x_inputs=[0.1],
            x_min=0.0,
            x_max=1.0,
            weight_values=[0.2, 0.3],
            w_min=0.0,
            w_max=1.0,
        )


def test_source_requires_at_least_one_input_in_all_sc_modes():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Bipolar mode must preserve the legacy no-empty-dot-product invariant."""
    with pytest.raises(ValueError, match="at least one"):
        _make_source(
            x_inputs=[],
            weight_values=[],
            x_min=-1.0,
            x_max=1.0,
            w_min=-1.0,
            w_max=1.0,
            sc_mode="bipolar",
        )


def test_source_rejects_unknown_sc_mode():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Unknown SC mode should fail closed instead of silently using AND semantics."""
    with pytest.raises(ValueError, match="sc_mode"):
        _make_source(sc_mode="ternary")
