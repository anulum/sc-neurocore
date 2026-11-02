# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source identity and solver-only configuration controls

"""Keep actual model configuration bytes unchanged except the selected solver."""

from __future__ import annotations

import pytest

from tests.formal_native_evidence_support import JOB, ROOT
from tools.formal_receipt_identity import solver_configuration


@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_actual_job_solver_variant_preserves_all_other_bytes(line_ending: str) -> None:
    """Keep depth, mode, script, files, comments and native line endings exact."""
    original = (ROOT / JOB).read_text().replace("\n", line_ending)
    result = solver_configuration(original, "yices-smt2")
    assert result == original.replace("smtbmc z3", "smtbmc yices")
    assert solver_configuration(result, "z3") == original


@pytest.mark.parametrize("engine", ["abc pdr", "smtbmc boolector", "smtbmc z3 extra"])
def test_unapproved_engine_variant_refuses(engine: str) -> None:
    """A solver-only selection cannot rewrite a different native proof engine."""
    original = (ROOT / JOB).read_text().replace("smtbmc z3", engine)
    with pytest.raises(ValueError, match="engine line"):
        solver_configuration(original, "yices-smt2")


def test_missing_engine_and_unregistered_solver_refuse() -> None:
    """Require the actual SMTBMC engine and one of the two approved solvers."""
    original = (ROOT / JOB).read_text()
    with pytest.raises(ValueError, match="no SMTBMC"):
        solver_configuration(original.replace("smtbmc z3", "# smtbmc z3"), "yices-smt2")
    with pytest.raises(ValueError, match="Unsupported solver-only configuration"):
        solver_configuration(original, "unknown")
