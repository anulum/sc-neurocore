# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Behaviour facet catalogue gate

"""Gate for the measured behaviour facet of the descriptor corpus.

Every model's committed ``behavior_tags`` must be drawn from the controlled
vocabulary and must equal the recorded measurement in ``behavior_evidence.json``.
The gate is fast — it compares the committed tags against the recorded manifest
without re-running a single simulation; the probe's own reproducibility is
proven separately in ``test_behavior_probe.py``. A hand-edited or stale tag set
therefore cannot pass.
"""

from __future__ import annotations

from sc_neurocore.neurons.behavior_taxonomy import BEHAVIOR_TAGS, validate_behavior_tags
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.behavior_probe import (
    BEHAVIOR_PROBE_SCHEMA_VERSION,
    BEHAVIOR_SWEEP_CURRENTS,
    behavior_tags_for,
    load_behavior_evidence,
)


def test_manifest_declares_the_recorded_sweep() -> None:
    """The recorded manifest carries the current sweep schema and ladder."""

    manifest = load_behavior_evidence()
    assert manifest["schema_version"] == BEHAVIOR_PROBE_SCHEMA_VERSION
    assert tuple(manifest["sweep"]["currents"]) == BEHAVIOR_SWEEP_CURRENTS


def test_manifest_covers_every_registered_model() -> None:
    """Every registered model has a recorded behaviour measurement."""

    manifest = load_behavior_evidence()
    missing = sorted(name for name in _CLASS_TO_MODULE if name not in manifest["models"])
    assert missing == [], f"models with no recorded behaviour: {missing}"


def test_every_descriptor_tag_is_in_the_vocabulary() -> None:
    """No descriptor names a behaviour tag outside the controlled vocabulary."""

    offenders: list[str] = []
    for class_name in _CLASS_TO_MODULE:
        descriptor = load_descriptor(class_name)
        assert descriptor is not None
        for tag in descriptor.behavior_tags:
            if tag not in BEHAVIOR_TAGS:
                offenders.append(f"{class_name}.{tag}")
    assert offenders == [], f"behaviour tags outside the vocabulary: {offenders}"


def test_descriptor_tags_match_the_recorded_measurement() -> None:
    """Each descriptor's committed tags equal the recorded measurement."""

    manifest = load_behavior_evidence()
    mismatched: list[str] = []
    for class_name in _CLASS_TO_MODULE:
        descriptor = load_descriptor(class_name)
        assert descriptor is not None
        committed = validate_behavior_tags(descriptor.behavior_tags)
        measured = validate_behavior_tags(behavior_tags_for(class_name, manifest))
        if committed != measured:
            mismatched.append(f"{class_name}: {list(committed)} != {list(measured)}")
    assert mismatched == [], f"descriptor tags disagree with the measurement: {mismatched[:20]}"


def test_some_models_have_measured_behaviour() -> None:
    """The facet is populated, not vacuously empty across the catalogue."""

    tagged = 0
    for class_name in _CLASS_TO_MODULE:
        descriptor = load_descriptor(class_name)
        assert descriptor is not None
        if descriptor.behavior_tags:
            tagged += 1
    assert tagged > 100, f"only {tagged} models carry measured behaviour tags"
