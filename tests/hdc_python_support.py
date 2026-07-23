# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_hdc_python.py

from __future__ import annotations

"""End-to-end Python tests for HDC/VSA via the Rust BitStreamTensor backend."""
import pytest
pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built", exc_type=ImportError)
from sc_neurocore_engine import BitStreamTensor, HDCVector

__all__ = ['pytest', 'BitStreamTensor', 'HDCVector']
