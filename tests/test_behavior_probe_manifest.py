# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Behaviour probe manifest contracts

"""Focused behavior-probe manifest contracts."""

from typing import Any, cast
from sc_neurocore.studio.behavior_probe import BEHAVIOR_SWEEP_CURRENTS
from sc_neurocore.studio.behavior_probe import behavior_tags_for
from sc_neurocore.studio.behavior_probe import probe_all_models


def test_probe_all_models_manifest_shape() -> None:
    """The manifest carries the sweep config, digests and per-model entries."""
    manifest = cast(dict[str, Any], probe_all_models(names=["ThetaNeuron", "DendriticNMDANeuron"]))
    assert manifest["schema_version"] == "studio.behavior-probe.v1"
    assert tuple(manifest["sweep"]["currents"]) == BEHAVIOR_SWEEP_CURRENTS
    assert len(manifest["sweep_sha256"]) == 64
    assert len(manifest["result_sha256"]) == 64
    assert set(manifest["models"]) == {"ThetaNeuron", "DendriticNMDANeuron"}
    assert manifest["models"]["DendriticNMDANeuron"]["drivable"] is False


def test_behavior_tags_for_reads_manifest_entry() -> None:
    """The helper extracts a model's recorded tags and is empty for unknowns."""
    manifest = probe_all_models(names=["ThetaNeuron"])
    assert behavior_tags_for("ThetaNeuron", manifest)
    assert behavior_tags_for("NoSuchModel", manifest) == ()
