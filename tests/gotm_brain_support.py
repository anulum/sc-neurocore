# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_gotm_brain.py

from __future__ import annotations

"""Tests for content_indexer.py and gotm_brain.py."""
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


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal fake GOTM repository for testing."""
    src = tmp_path / "src" / "example"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text('"""Example module docstring."""\n')
    (src / "core.py").write_text(
        '"""Core module.\n\n'
        "Implements the main algorithm.\n"
        '"""\n\n'
        "import numpy as np\n\n"
        "def compute(x):\n"
        '    """Compute the result."""\n'
        "    return x * 2\n"
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text(
        "# Example Repository\n\n"
        "This is a test repository for the GOTM brain.\n\n"
        "## Features\n\n"
        "- Feature 1: quantum coupling\n"
        "- Feature 2: metabolic feedback\n"
    )
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "example"\nversion = "0.1.0"\n')
    rust_src = tmp_path / "src" / "lib.rs"
    rust_src.write_text(
        "/// Main computation function.\n"
        "/// Returns the doubled input.\n"
        "pub fn compute(x: f64) -> f64 {\n"
        "    x * 2.0\n"
        "}\n"
    )
    # Add a __pycache__ dir that should be skipped
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "junk.pyc").write_bytes(b"\x00" * 100)
    return tmp_path


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
    "tmp_repo",
    "__all__",
]
