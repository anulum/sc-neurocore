# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIndexFile from former test_gotm_brain.py

"""Focused suite: TestIndexFile from former test_gotm_brain.py."""

from __future__ import annotations

from pathlib import Path
from sc_neurocore.quantum_cognition.content_indexer import index_file


class TestIndexFile:
    """Indexing single source files."""

    def test_index_python(self, tmp_repo: Path) -> None:
        """A Python file gives docstring or code chunks."""
        py_file = tmp_repo / "src" / "example" / "core.py"
        chunks = index_file(py_file, "TEST", tmp_repo)
        assert len(chunks) > 0
        types = {c.content_type for c in chunks}
        assert "docstring" in types or "code" in types

    def test_index_markdown(self, tmp_repo: Path) -> None:
        """A Markdown file gives markdown chunks."""
        md_file = tmp_repo / "docs" / "README.md"
        chunks = index_file(md_file, "TEST", tmp_repo)
        assert len(chunks) > 0
        assert chunks[0].content_type == "markdown"

    def test_index_rust(self, tmp_repo: Path) -> None:
        """A Rust file gives chunks."""
        rs_file = tmp_repo / "src" / "lib.rs"
        chunks = index_file(rs_file, "TEST", tmp_repo)
        assert len(chunks) > 0

    def test_skip_unknown_ext(self, tmp_repo: Path) -> None:
        """A file of unknown type gives no chunks."""
        unk = tmp_repo / "test.xyz"
        unk.write_text("unknown")
        chunks = index_file(unk, "TEST", tmp_repo)
        assert len(chunks) == 0

    def test_skips_empty_oversized_and_non_file_inputs(self, tmp_path: Path) -> None:
        """Indexer ignores unsupported filesystem payloads without poisoning a scan."""
        empty_md = tmp_path / "empty.md"
        empty_md.write_text("")
        assert index_file(empty_md, "TEST", tmp_path) == []

        oversized_py = tmp_path / "oversized.py"
        oversized_py.write_text('"""Large module docstring."""\n' + ("x = 1\n" * 60_000))
        assert index_file(oversized_py, "TEST", tmp_path) == []

        directory_with_supported_suffix = tmp_path / "not_a_file.md"
        directory_with_supported_suffix.mkdir()
        assert index_file(directory_with_supported_suffix, "TEST", tmp_path) == []

    def test_indexes_supported_metadata_and_hardware_files_as_code_chunks(
        self, tmp_path: Path
    ) -> None:
        """Non-doc source formats retain provenance and extension-specific weights."""
        payloads = {
            "Project.toml": 'name = "qc-indexer"\nversion = "1.0.0"\n',
            "config.yaml": "model: fisher_posner\nqubits: 12\n",
            "manifest.json": '{"repo": "SC-NEUROCORE", "pipeline": "quantum"}\n',
            "kernel.go": "package main\nfunc Step() {}\n",
            "proof.lean": "theorem posner_index : True := by trivial\n",
            "bridge.sv": "module bridge; endmodule\n",
        }
        for rel_path, text in payloads.items():
            path = tmp_path / rel_path
            path.write_text(text)
            chunks = index_file(path, "TEST", tmp_path)
            assert len(chunks) == 1
            assert chunks[0].repo_name == "TEST"
            assert chunks[0].file_path == rel_path
            assert chunks[0].content_type == "code"
            assert chunks[0].summary
