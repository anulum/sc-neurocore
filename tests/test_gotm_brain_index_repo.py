# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIndexRepo from former test_gotm_brain.py

"""Focused suite: TestIndexRepo from former test_gotm_brain.py."""

from __future__ import annotations

from tests.gotm_brain_support import *  # noqa: F403

class TestIndexRepo:
    def test_index_repo(self, tmp_repo: Path) -> None:
        chunks = index_gotm_repo(tmp_repo, "TEST-REPO")
        assert len(chunks) > 0
        repos = {c.repo_name for c in chunks}
        assert "TEST-REPO" in repos

    def test_skips_pycache(self, tmp_repo: Path) -> None:
        chunks = index_gotm_repo(tmp_repo)
        paths = {c.file_path for c in chunks}
        for p in paths:
            assert "__pycache__" not in p

    def test_nonexistent_repo(self) -> None:
        with pytest.raises(FileNotFoundError):
            index_gotm_repo("/nonexistent/path")

    def test_sorted_by_weight(self, tmp_repo: Path) -> None:
        chunks = index_gotm_repo(tmp_repo)
        weights = [c.weight for c in chunks]
        assert weights == sorted(weights, reverse=True)

    def test_skips_hidden_and_build_directories_during_walk(self, tmp_path: Path) -> None:
        visible = tmp_path / "src"
        visible.mkdir()
        (visible / "model.md").write_text("Visible quantum cognition notes.")

        hidden = tmp_path / ".cache"
        hidden.mkdir()
        (hidden / "secret.md").write_text("This hidden file must not be indexed.")

        build = tmp_path / "build"
        build.mkdir()
        (build / "artifact.md").write_text("This build artifact must not be indexed.")

        chunks = index_gotm_repo(tmp_path, "SCAN")
        paths = {chunk.file_path for chunk in chunks}
        assert paths == {"src/model.md"}
