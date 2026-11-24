# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_gotm_brain.py

"""Shared imports and test helpers of the GOTM brain and content-indexer suites."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Protocol, cast
import numpy as np
import pytest
from sc_neurocore.quantum_cognition.content_indexer import (
    ContentChunk,
    _chunk_text,
    _extract_python_docstrings,
    _extract_rust_doc_comments,
    _should_skip_dir,
    embed_chunks,
    embed_tfidf,
    index_file,
    index_gotm_repo,
)
from sc_neurocore.quantum_cognition.gotm_brain import (
    GOTMBrain,
    LearningStep,
)
from sc_neurocore.quantum_cognition.fisher_posner import HybridFisherPosnerLIF

_GOTM_MODULE = "sc_neurocore.quantum_cognition.gotm_brain"


class _GotmBrainModule(Protocol):
    """Typed view of a dynamically reloaded GOTM brain module."""

    HAS_LLM: bool
    GOTMBrain: type[GOTMBrain]
    _LLMEndpoint: type[object] | None


class _BlockingLLMFinder(importlib.abc.MetaPathFinder):
    """Import hook that makes the local ``llm`` module unavailable."""

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Raise on ``llm`` and ignore every other import."""
        if fullname == "llm":
            raise ModuleNotFoundError("blocked test llm module")
        return None


class _FixedSpikeNeuron:
    """Minimal neuron stub that returns a configured spike flag."""

    def __init__(self, spiked: bool) -> None:
        """Store the spike flag returned by ``step``."""
        self._spiked = spiked
        self.atp_level = 1.0

    def step(self, _current: float) -> tuple[float, bool]:
        """Return a deterministic voltage/spike pair."""
        return 0.0, self._spiked


__all__ = [
    "importlib",
    "sys",
    "types",
    "Path",
    "Protocol",
    "cast",
    "np",
    "pytest",
    "ContentChunk",
    "_chunk_text",
    "_extract_python_docstrings",
    "_extract_rust_doc_comments",
    "_should_skip_dir",
    "embed_chunks",
    "embed_tfidf",
    "index_file",
    "index_gotm_repo",
    "GOTMBrain",
    "LearningStep",
    "HybridFisherPosnerLIF",
    "_GOTM_MODULE",
    "_GotmBrainModule",
    "_BlockingLLMFinder",
    "_FixedSpikeNeuron",
    "__all__",
]
