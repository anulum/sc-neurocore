# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (constructor_rejects) from former test_model_resonate_and_fire.py

"""Constructor rejections of the resonate-and-fire neuron."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.resonate_and_fire import (
    ResonateAndFireNeuron,
)
from typing import (
    cast,
)


@pytest.mark.parametrize("field", ("x", "y", "b", "omega", "threshold", "dt"))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_constructor_rejects_nonfinite_values(field: str, value: float) -> None:
    """Constructor rejects non-finite values."""
    with pytest.raises(ValueError, match="finite"):
        kwargs: dict[str, Any] = {field: value}
        ResonateAndFireNeuron(**kwargs)


@pytest.mark.parametrize("field", ("omega", "threshold", "dt"))
@pytest.mark.parametrize("value", (0.0, -1.0))
def test_constructor_rejects_nonpositive_scales(field: str, value: float) -> None:
    """Constructor rejects nonpositive scales."""
    with pytest.raises(ValueError, match=field):
        kwargs: dict[str, Any] = {field: value}
        ResonateAndFireNeuron(**kwargs)


@pytest.mark.parametrize("field", ("x", "y", "b", "omega", "threshold", "dt"))
def test_constructor_rejects_nonnumeric_values(field: str) -> None:
    """Constructor rejects nonnumeric values."""
    with pytest.raises(ValueError, match="numeric"):
        ResonateAndFireNeuron(**cast(dict[str, float], {field: object()}))
