# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR bridge (import, node mapping, execution)

"""Tests for nir_bridge: NIR graph → SC-NeuroCore network conversion."""

from pathlib import Path

import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.nir_bridge import from_nir

from tests.test_nir_bridge.support import make_lif_affine_graph


class TestFileIO:
    def test_from_nir_file(self, tmp_path: Path) -> None:
        graph = make_lif_affine_graph()
        path = tmp_path / "test_model.nir"
        nir.write(str(path), graph)

        net = from_nir(str(path))
        assert len(net.nodes) == 4

    def test_from_nir_path_object(self, tmp_path: Path) -> None:
        graph = make_lif_affine_graph()
        path = tmp_path / "test_model.nir"
        nir.write(str(path), graph)

        net = from_nir(path)
        assert "lif" in net.nodes

    def test_invalid_source_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected NIRGraph"):
            from_nir(42)
