# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_gotm_brain_inline.py

from __future__ import annotations

"""Inline tests for GOTMBrain — persistence, LLM fallback, CLI."""
from pathlib import Path
import numpy as np
import pytest
from sc_neurocore.quantum_cognition.content_indexer import ContentChunk
from sc_neurocore.quantum_cognition.gotm_brain import (
    HAS_LLM,
    GOTMBrain,
    LearningStep,
)

__all__ = ["Path", "np", "pytest", "ContentChunk", "HAS_LLM", "GOTMBrain", "LearningStep"]
