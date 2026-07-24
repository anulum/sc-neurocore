# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_tinysc_ports.py

"""Module-level tests from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403


def test_sclayer_rejects_mismatched_weight_row_length() -> None:
    # Right number of rows but a row whose word count does not match
    # words_per_input is rejected by the SCLayer weight validator.
    with pytest.raises(ValueError, match="each weight row must match words_per_input"):
        SCLayer(n_inputs=64, n_outputs=2, weights=[[0], [0, 0, 0]])
