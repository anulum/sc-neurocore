# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_neuroml_import.py

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from textwrap import dedent

from neuroml.utils import validate_neuroml2
import pytest

from sc_neurocore.adapters.neuroml import (
    ImportedCell,
    create_neuron,
    import_neuroml,
)
from sc_neurocore.neurons.models import Izhikevich2007Neuron
from sc_neurocore.neurons.models.adex import AdExNeuron

FIXTURES = Path(__file__).parent / "fixtures" / "neuroml"


@pytest.fixture(autouse=True)
def ensure_fixtures(tmp_path: Path) -> Iterator[Path]:
    """Create test NeuroML files in tmp_path."""
    d = tmp_path / "neuroml"
    d.mkdir()
    yield d


def _write_nml(path: Path, body: str, *, valid: bool = True) -> Path:
    """Write a NeuroML document; unless told otherwise, the upstream schema must accept it.

    Every fixture meant to be valid is checked by libNeuroML's own validator,
    so a test never exercises the importer on a document real tools reject.
    """
    header = dedent("""\
    <neuroml xmlns="http://www.neuroml.org/schema/neuroml2"
             id="test">
    """)
    path.write_text(header + body + "\n</neuroml>")
    if valid:
        validate_neuroml2(str(path))
    return path


__all__ = [
    "Path",
    "dedent",
    "pytest",
    "ImportedCell",
    "create_neuron",
    "import_neuroml",
    "Izhikevich2007Neuron",
    "AdExNeuron",
    "FIXTURES",
    "ensure_fixtures",
    "_write_nml",
]
