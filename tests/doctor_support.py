# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_doctor.py

from __future__ import annotations

import numpy as np
from sc_neurocore.doctor import diagnose, Diagnosis, DiagnosticReport
from sc_neurocore.doctor.diagnose import Severity

__all__ = ["np", "diagnose", "Diagnosis", "DiagnosticReport", "Severity"]
