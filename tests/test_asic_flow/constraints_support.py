# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_constraints.py

from __future__ import annotations

"""Exercise CDC, IR-drop, IO-placement, and equivalence decks."""
from sc_neurocore.asic_flow.asic_flow import (
    CDCCheckGenerator,
    DesignParams,
    IOConstraintGenerator,
    IOPin,
    IRDropGenerator,
    LECGenerator,
    PDKConfig,
    PDKType,
)

__all__ = ['CDCCheckGenerator', 'DesignParams', 'IOConstraintGenerator', 'IOPin', 'IRDropGenerator', 'LECGenerator', 'PDKConfig', 'PDKType']
