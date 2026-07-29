# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo scalar primitives for the Amari 1977 neural field

from std.math import exp


@always_inline
def amari_kernel(
    distance: Float64,
    a_exc: Float64,
    a_width: Float64,
    b_inh: Float64,
    b_width: Float64,
) -> Float64:
    """Return one difference-of-exponentials interaction coefficient."""
    return a_exc * exp(-a_width * distance) - b_inh * exp(-b_width * distance)


@always_inline
def amari_activity(state: Float64) -> Float64:
    """Return Amari's source-level Heaviside firing rate."""
    if state > 0.0:
        return 1.0
    return 0.0
