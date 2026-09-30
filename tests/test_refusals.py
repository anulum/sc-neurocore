# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared authored-refusal compatibility at a production boundary

"""A real domain admission error retains the shared authored-message contract."""

import pytest

from sc_neurocore.fitting.problem import ParameterDomain
from sc_neurocore.refusals import AuthoredRefusal


def test_parameter_admission_preserves_value_error_and_authored_base_contract() -> None:
    """Existing ValueError consumers can still catch a deliberate domain refusal."""
    with pytest.raises(
        ValueError, match="domain of tau_m must be finite with low < high"
    ) as caught:
        ParameterDomain("tau_m", 1.0, 0.0)
    assert isinstance(caught.value, AuthoredRefusal)
