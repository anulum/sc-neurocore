# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPackageImport from former test_gotm_brain.py

"""Focused suite: TestPackageImport from former test_gotm_brain.py."""

from __future__ import annotations

from tests.gotm_brain_support import *  # noqa: F403


class TestPackageImport:
    def test_import_new_symbols(self) -> None:
        from sc_neurocore.quantum_cognition import (
            ContentChunk,
            GOTMBrain,
            embed_chunks,
            index_gotm_repo,
        )

        assert ContentChunk is not None
        assert GOTMBrain is not None
        assert embed_chunks is not None
        assert index_gotm_repo is not None
