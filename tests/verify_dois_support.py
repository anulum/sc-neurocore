# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_verify_dois.py

from __future__ import annotations

"""Unit tests for the network-free logic of ``tools/provenance/verify_dois.py``.

The diacritic folding and the author/year/verified classification are what keep
the DOI gate both strict (a fabricated or wrong DOI cannot pass) and free of
false positives (Llinás vs Llinas, a 2006/2007 book date). Registry routing is
checked with the HTTP layer monkeypatched, so no test here touches the network.
"""
import sys
from pathlib import Path
from typing import Any
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "provenance"))
import verify_dois  # noqa: E402

__all__ = ['sys', 'Path', 'Any', 'pytest', 'verify_dois']
