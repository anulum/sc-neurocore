# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestContentIndexerAdversarial from former test_qc_e2e.py

"""Focused suite: TestContentIndexerAdversarial from former test_qc_e2e.py."""

from __future__ import annotations

import os
from pathlib import (
    Path,
)
from sc_neurocore.quantum_cognition.content_indexer import (
    index_gotm_repo,
)


class TestContentIndexerAdversarial:
    """Edge cases: Unicode, empty, binary, deeply nested."""

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Mathematical and non-Latin Unicode text is indexed into at least one chunk."""
        f = tmp_path / "unicode.md"
        f.write_text("# Ĉapitro 1\n∀x∈ℝ: ∫f(x)dx = Σaₙxⁿ\n日本語テスト\n", encoding="utf-8")
        chunks = index_gotm_repo(str(tmp_path))
        assert len(chunks) >= 1

    def test_empty_files(self, tmp_path: Path) -> None:
        """Empty and whitespace-only files never yield a non-text chunk."""
        (tmp_path / "empty.md").write_text("")
        (tmp_path / "whitespace.py").write_text("   \n\n  \n")
        chunks = index_gotm_repo(str(tmp_path))
        # Empty/whitespace files may produce 0 chunks — that's OK
        for chunk in chunks:
            assert isinstance(chunk.text, str)

    def test_binary_file_skip(self, tmp_path: Path) -> None:
        """Binary files are skipped, so no chunk is typed as binary."""
        (tmp_path / "binary.bin").write_bytes(os.urandom(1024))
        (tmp_path / "real.md").write_text("# Real content\nSome math\n")
        chunks = index_gotm_repo(str(tmp_path))
        # Binary should be skipped, only .md indexed
        for chunk in chunks:
            assert chunk.content_type != "binary"

    def test_deeply_nested(self, tmp_path: Path) -> None:
        """A document ten directories deep is still found and indexed."""
        d = tmp_path
        for i in range(10):
            d = d / f"level_{i}"
        d.mkdir(parents=True)
        (d / "deep.md").write_text("# Deep theorem\n∀ε>0\n")
        chunks = index_gotm_repo(str(tmp_path))
        assert len(chunks) >= 1
