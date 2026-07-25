# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Behaviour probe manifest contracts

"""Focused behavior-probe manifest contracts."""

from .behavior_probe_support import *


def test_probe_all_models_manifest_shape() -> None:
    """The manifest carries the sweep config, digests and per-model entries."""

    manifest = probe_all_models(names=["ThetaNeuron", "DendriticNMDANeuron"])
    assert manifest["schema_version"] == "studio.behavior-probe.v1"
    assert tuple(manifest["sweep"]["currents"]) == BEHAVIOR_SWEEP_CURRENTS  # type: ignore[index,call-overload,arg-type] # Preserved JSON manifest AST
    assert len(manifest["sweep_sha256"]) == 64  # type: ignore[arg-type] # Preserved JSON manifest AST
    assert len(manifest["result_sha256"]) == 64  # type: ignore[arg-type] # Preserved JSON manifest AST
    assert set(manifest["models"]) == {"ThetaNeuron", "DendriticNMDANeuron"}  # type: ignore[arg-type] # Preserved JSON manifest AST
    assert manifest["models"]["DendriticNMDANeuron"]["drivable"] is False  # type: ignore[index,call-overload] # Preserved JSON manifest AST


def test_behavior_tags_for_reads_manifest_entry() -> None:
    """The helper extracts a model's recorded tags and is empty for unknowns."""

    manifest = probe_all_models(names=["ThetaNeuron"])
    assert behavior_tags_for("ThetaNeuron", manifest)
    assert behavior_tags_for("NoSuchModel", manifest) == ()
