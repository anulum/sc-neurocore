# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_bci_studio.py

from __future__ import annotations

import unittest
import numpy as np
from sc_neurocore.bci_studio.bci_studio import (
    SpikeCodec,
    OnlineLearner,
    FPGAFeedbackController,
    LatencyProfiler,
    BCIStudio,
)
if __name__ == "__main__":
    unittest.main()

__all__ = ['unittest', 'np', 'SpikeCodec', 'OnlineLearner', 'FPGAFeedbackController', 'LatencyProfiler', 'BCIStudio']
