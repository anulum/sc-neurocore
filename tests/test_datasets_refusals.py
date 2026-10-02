# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dataset refusal provenance through public readers

"""Keep deliberate dataset guards distinct from real conversion faults."""

from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.datasets.encoders import EventBinning, encoder_from_declaration
from sc_neurocore.datasets.manifest import EventDatasetManifest, build_manifest, manifest_from_dict
from sc_neurocore.datasets.refusals import DatasetRefusal
from sc_neurocore.datasets.splits import group_split, split_plan_from_dict
from tests.event_dataset_support import write_shd


@pytest.fixture
def manifest(tmp_path: Path) -> EventDatasetManifest:
    """Read actual SHD-format recordings through the public manifest builder."""
    write_shd(tmp_path, {"train": [0, 0, 1, 1, 2, 2, 3, 3], "test": [4]})
    return build_manifest("shd", tmp_path, version="generated-format-fixture")


def test_manifest_schema_refusal_remains_value_error(manifest: EventDatasetManifest) -> None:
    """Existing ValueError handlers receive the exact authored schema message."""
    document = manifest.to_dict()
    document["schema"] = "unsupported"
    with pytest.raises(ValueError, match="manifest schema 'unsupported'") as caught:
        manifest_from_dict(document)
    assert isinstance(caught.value, DatasetRefusal)


@pytest.mark.parametrize("field", ["bytes", "index", "label"])
def test_manifest_conversion_fault_is_unmarked(manifest: EventDatasetManifest, field: str) -> None:
    """Real integer conversion errors cannot acquire caller-facing provenance."""
    document = manifest.to_dict()
    collection = document["files" if field == "bytes" else "samples"]
    collection[0][field] = "caller-text-xyz"
    with pytest.raises(ValueError, match="invalid literal") as caught:
        manifest_from_dict(document)
    assert not isinstance(caught.value, DatasetRefusal)


def test_split_refusal_keeps_authored_reason(manifest: EventDatasetManifest) -> None:
    """Public split imports preserve useful seed refusals and compatibility."""
    document = group_split(manifest, fractions={"train": 0.5, "evaluation": 0.5}, seed=7).to_dict()
    document["seed"] = "caller-text-xyz"
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 2\*\*32\)") as caught:
        split_plan_from_dict(document)
    assert isinstance(caught.value, DatasetRefusal)


def test_encoder_refusal_keeps_authored_reason() -> None:
    """A public declaration rebuild exposes its own positive-window guard."""
    declaration = EventBinning(1.0, 4, 700, 1, "merge").declaration()
    declaration["n_steps"] = 0
    with pytest.raises(ValueError, match="n_steps must be a positive integer") as caught:
        encoder_from_declaration(cast(dict[str, object], declaration))
    assert isinstance(caught.value, DatasetRefusal)
