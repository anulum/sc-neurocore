# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the compiler compatibility-facade import surfaces

"""Contracts for the documented compiler compatibility facades.

`compiler.quantize_core` and `compiler.adaptive_runtime_precision.validation` are
stable public import surfaces (referenced in docs/API_REFERENCE.md) that re-export
symbols from `compiler.quantizer` and `compiler.validation`. These tests pin the
re-export contract: every advertised name resolves to the same object as its
canonical source, so the facade cannot silently drift from the real implementation.
"""

from __future__ import annotations

import sc_neurocore.compiler.adaptive_runtime_precision.validation as arp_validation
import sc_neurocore.compiler.quantize_core as quantize_core
import sc_neurocore.compiler.quantizer as quantizer
import sc_neurocore.compiler.validation as validation


def test_quantize_core_reexports_match_quantizer() -> None:
    """Every symbol the quantize_core facade advertises is the canonical quantizer object."""
    assert quantize_core.__all__
    for name in quantize_core.__all__:
        assert hasattr(quantize_core, name), f"facade missing {name}"
        assert getattr(quantize_core, name) is getattr(quantizer, name), f"{name} drifted"


def test_adaptive_runtime_precision_validation_reexports_match_source() -> None:
    """The adaptive-runtime validation facade re-exports the canonical validation helpers."""
    assert arp_validation.__all__
    for name in arp_validation.__all__:
        assert hasattr(arp_validation, name), f"facade missing {name}"
        assert getattr(arp_validation, name) is getattr(validation, name), f"{name} drifted"
