# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTextExtraction from former test_gotm_brain.py

"""Focused suite: TestTextExtraction from former test_gotm_brain.py."""

from __future__ import annotations

from tests.gotm_brain_support import *  # noqa: F403

class TestTextExtraction:
    def test_python_docstrings(self) -> None:
        code = '"""Module docstring that is long enough."""\ndef f():\n    """Function docstring text here."""\n    pass\n'
        docs = _extract_python_docstrings(code)
        assert len(docs) >= 1

    def test_python_single_quote_docstrings_and_comment_blocks(self) -> None:
        code = (
            "'''Single quoted module documentation that is long enough.'''\n"
            "# First scientific note\n"
            "# Second scientific note\n"
            "# Third scientific note\n"
            "value = 1\n"
            "# Terminal note one\n"
            "# Terminal note two\n"
            "# Terminal note three"
        )
        docs = _extract_python_docstrings(code)
        assert "Single quoted module documentation" in docs[0]
        assert any(
            "First scientific note\nSecond scientific note\nThird scientific note" in d
            for d in docs
        )
        assert any("Terminal note one\nTerminal note two\nTerminal note three" in d for d in docs)

    def test_rust_doc_comments(self) -> None:
        code = "/// First line of doc.\n/// Second line of doc.\nfn main() {}\n"
        docs = _extract_rust_doc_comments(code)
        assert len(docs) == 1
        assert "First line" in docs[0]

    def test_rust_inner_doc_comments_at_eof(self) -> None:
        code = "//! Module-level quantum cognition notes.\n//! Preserved at end of file."
        docs = _extract_rust_doc_comments(code)
        assert docs == ["Module-level quantum cognition notes.\nPreserved at end of file."]

    def test_chunk_text_short(self) -> None:
        chunks = _chunk_text("short text")
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_chunk_text_long(self) -> None:
        text = "\n\n".join([f"Paragraph {i} with some content." for i in range(50)])
        chunks = _chunk_text(text, target_size=200)
        assert len(chunks) > 1

    def test_skip_dir(self) -> None:
        assert _should_skip_dir("__pycache__")
        assert _should_skip_dir(".git")
        assert not _should_skip_dir("src")
