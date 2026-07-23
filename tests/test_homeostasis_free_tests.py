# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_homeostasis.py

"""Module-level tests from former test_homeostasis.py."""

from __future__ import annotations

from tests.homeostasis_support import *  # noqa: F403

def test_regulate_rejects_non_finite_thresholds() -> None:
    reg = NetworkRegulator(target_rate=0.1)
    rates = np.full(4, 0.1)
    thresholds = np.array([1.0, 1.0, np.nan, 1.0])
    # A correctly-shaped 1-D threshold vector with a non-finite entry is rejected.
    with pytest.raises(ValueError, match="thresholds must be finite"):
        reg.regulate(rates, thresholds, 0.01)
