# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McCulloch-Pitts hardware encoding contract

"""Signed Q32.0 hardware-input encoding contract."""

from .model_mcculloch_pitts_support import *


@pytest.mark.parametrize(
    ("count", "inhibited", "encoded"),
    ((0, False, 0), (7, False, 7), (_INT32_MAX, False, _INT32_MAX), (7, True, -1)),
)
def test_signed_q320_encoding_is_bijective_over_valid_logical_inputs(
    count: int,
    inhibited: bool,
    encoded: int,
) -> None:
    """The RTL input uses -1 only for inhibition and non-negative values for counts."""
    assert encode_hardware_input(count, inhibited) == encoded
