# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestImportNodes from former test_sonata_import.py

"""Focused suite: TestImportNodes from former test_sonata_import.py."""

from __future__ import annotations

import pytest
from pathlib import (
    Path,
)
from sc_neurocore.adapters.sonata import (
    import_sonata_nodes,
)
from tests.sonata_import_support import (
    _create_nodes_h5,
    h5py,
)


class TestImportNodes:
    """Importing SONATA nodes, and refusing a file that is not SONATA."""

    def test_basic(self, tmp_path: Path) -> None:
        """A small nodes file imports five nodes with their ids."""
        f = _create_nodes_h5(tmp_path / "nodes.h5", n=5)
        nodes = import_sonata_nodes(f)
        assert len(nodes) == 5
        assert nodes[0].node_id == 0
        assert nodes[4].node_id == 4

    def test_a_file_that_is_not_sonata_is_refused(self, tmp_path: Path) -> None:
        """A file that is not SONATA is refused."""
        p = tmp_path / "empty.h5"
        with h5py.File(p, "w"):
            pass
        with pytest.raises(ValueError, match="not a SONATA file"):
            import_sonata_nodes(p)
