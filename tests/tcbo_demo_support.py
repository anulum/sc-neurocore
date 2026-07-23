# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_tcbo_demo.py

from __future__ import annotations

"""Tests for TCBO Consciousness Detection Demo (UC2).

38 tests covering SyntheticEEGGenerator, core functions, TCBODemoEngine,
scenario validations, and singleton management.
"""
import unittest
import numpy as np
from sc_neurocore.experiments.tcbo_demo_engine import (
    SyntheticEEGGenerator,
    TCBODemoEngine,
    TCBODemoSnapshot,
    TCBOController,
    ScenarioName,
    _compute_order_parameter,
    _compute_p_h1_lightweight,
    get_tcbo_demo_engine,
    reset_tcbo_demo_engine,
)
from sc_neurocore.scpn.params import build_knm_matrix as _build_knm
if __name__ == "__main__":
    unittest.main()

__all__ = ['unittest', 'np', 'SyntheticEEGGenerator', 'TCBODemoEngine', 'TCBODemoSnapshot', 'TCBOController', 'ScenarioName', '_compute_order_parameter', '_compute_p_h1_lightweight', 'get_tcbo_demo_engine', 'reset_tcbo_demo_engine', '_build_knm']
