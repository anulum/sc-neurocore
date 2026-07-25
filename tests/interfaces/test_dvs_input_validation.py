# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DVS input validation contracts

"""Focused DVS input validation contracts."""

from tests.interfaces.dvs_input_support import *


@pytest.mark.parametrize("decay_tau", [0.0, -1.0, np.inf, True, "100.0"])
def test_dvs_rejects_invalid_decay_tau(decay_tau: Any) -> None:
    """Decay time constant must be finite and positive."""
    with pytest.raises(ValueError, match="decay_tau must be finite and positive"):
        DVSInputLayer(height=2, width=2, decay_tau=decay_tau)


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (0, 2),
        (2, 0),
        (-1, 2),
        (True, 2),
        (2, False),
        (1.5, 2),
    ],
)
def test_dvs_rejects_invalid_dimensions(height: Any, width: Any) -> None:
    """DVS frame dimensions must be positive integer pixel counts."""
    with pytest.raises(ValueError, match="height and width must be positive integers"):
        DVSInputLayer(height=height, width=width)
