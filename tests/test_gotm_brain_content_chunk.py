# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestContentChunk from former test_gotm_brain.py

"""Focused suite: TestContentChunk from former test_gotm_brain.py."""

from __future__ import annotations

from sc_neurocore.quantum_cognition.content_indexer import (
    ContentChunk,
)


class TestContentChunk:
    """The indexed content chunk record."""

    def test_create(self) -> None:
        """A chunk keeps its repository, digest and summary."""
        c = ContentChunk(
            repo_name="TEST",
            file_path="a.py",
            chunk_index=0,
            text="hello world",
            content_type="code",
            weight=1.0,
        )
        assert c.repo_name == "TEST"
        assert len(c.sha256) == 16
        assert c.summary == "hello world"

    def test_to_dict(self) -> None:
        """A chunk converts to a dictionary with its fields."""
        c = ContentChunk(
            repo_name="R",
            file_path="b.md",
            chunk_index=1,
            text="test content",
            content_type="markdown",
            weight=1.2,
        )
        d = c.to_dict()
        assert d["repo"] == "R"
        assert d["type"] == "markdown"
        assert d["length"] == 12
