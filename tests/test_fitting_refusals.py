# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Laboratory refusal types across real scientific admission

"""Check deliberate domain validation separately from generated document faults."""

import pytest

from sc_neurocore.fitting.constraints import ParameterConstraint
from sc_neurocore.fitting.problem import ParameterDomain, problem_from_dict
from sc_neurocore.fitting.refusals import LaboratoryRefusal
from tests.test_studio_fits_routes import _exported_problem


def test_log_domain_refusal_keeps_its_deliberate_message() -> None:
    """A real search domain rejects nonpositive logarithmic bounds explicitly."""
    with pytest.raises(LaboratoryRefusal, match="a log domain of tau_m must be positive"):
        ParameterDomain("tau_m", 0.0, 1.0, "log")


def test_constraint_refusal_keeps_its_deliberate_message() -> None:
    """An actual constraint refuses an empty feasible interval explicitly."""
    with pytest.raises(LaboratoryRefusal, match="constraint bounds must be finite with low < high"):
        ParameterConstraint("sum", {"R": 1.0}, 1.0, 1.0)


def test_document_conversion_is_not_marked_as_an_authored_refusal() -> None:
    """The exported public protocol does not trust text from Python's float parser."""
    document = _exported_problem()
    document["domains"][0]["low"] = "caller-text-xyz"
    with pytest.raises(ValueError) as caught:
        problem_from_dict(document)
    assert not isinstance(caught.value, LaboratoryRefusal)
