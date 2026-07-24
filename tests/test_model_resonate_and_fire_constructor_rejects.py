# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (constructor_rejects) from former test_model_resonate_and_fire.py

from __future__ import annotations

from tests.model_resonate_and_fire_support import *  # noqa: F403

@pytest.mark.parametrize("field", ("x", "y", "b", "omega", "threshold", "dt"))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_constructor_rejects_nonfinite_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ResonateAndFireNeuron(**{field: value})


@pytest.mark.parametrize("field", ("omega", "threshold", "dt"))
@pytest.mark.parametrize("value", (0.0, -1.0))
def test_constructor_rejects_nonpositive_scales(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        ResonateAndFireNeuron(**{field: value})


@pytest.mark.parametrize("field", ("x", "y", "b", "omega", "threshold", "dt"))
def test_constructor_rejects_nonnumeric_values(field: str) -> None:
    with pytest.raises(ValueError, match="numeric"):
        ResonateAndFireNeuron(**cast(dict[str, float], {field: object()}))
