# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public HDL identifier admission

"""Prevent silent module and port renaming through both native hardware backends."""

from __future__ import annotations

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder


@pytest.mark.parametrize("component", ["module", "input", "output"])
def test_predefined_macro_identifier_is_refused_before_hardware_emission(component: str) -> None:
    """Use the same authored refusal when CIRCT would change a public identifier."""
    b = ScGraphBuilder("SYNTHESIS" if component == "module" else "SYNTHESIS_MODEL")
    value = b.input("SYNTHESIS" if component == "input" else "value", "bool")
    b.output("SYNTHESIS" if component == "output" else "result", value)
    graph = b.build()
    errors = graph.verify()
    assert errors and any("reserved by CIRCT" in error for error in errors)
    for operation in [graph.emit_sv, graph.emit_mlir]:
        with pytest.raises(ValueError, match="reserved by CIRCT"):
            operation()
