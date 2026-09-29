# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event training input contracts

"""Bind a temporal training input to its files, split and encoder semantics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from sc_neurocore.datasets.encoders import EventBinning, encoder_from_declaration
from sc_neurocore.datasets.manifest import EventDatasetManifest, manifest_from_dict
from sc_neurocore.datasets.splits import SplitPlan, split_plan_from_dict, validate_split_plan

EVENT_TRAINING_SCHEMA = "sc-neurocore.studio.event-training-data.v1"


@dataclass(frozen=True, slots=True)
class EventTrainingContract:
    """Portable event data custody, with no operator filesystem path.

    Attributes
    ----------
    manifest:
        Every dataset file, sample, label, group and publisher declaration.
    split:
        Complete partition of one published source split by whole groups.
    encoder:
        Explicit event window, channel layout and late-event semantics.
    train_split, evaluation_split:
        Distinct non-empty parts of the plan used for optimisation and scoring.
    """

    manifest: EventDatasetManifest
    split: SplitPlan
    encoder: EventBinning
    train_split: str
    evaluation_split: str

    def to_dict(self) -> dict[str, Any]:
        """Return the exact portable declaration used in training checkpoints."""
        return {
            "schema": EVENT_TRAINING_SCHEMA,
            "training_cpu_threads": 1,
            "manifest": self.manifest.to_dict(),
            "split": self.split.to_dict(),
            "encoder": self.encoder.declaration(),
            "train_split": self.train_split,
            "evaluation_split": self.evaluation_split,
            "digests": {
                "manifest": self.manifest.digest,
                "split": self.split.digest,
                "encoder": self.encoder.digest,
            },
        }

    def receipt(self) -> dict[str, object]:
        """Return input digests, sample counts and temporal semantics for a run."""
        return {
            "schema": EVENT_TRAINING_SCHEMA,
            "manifest_digest": self.manifest.digest,
            "split_digest": self.split.digest,
            "encoder_digest": self.encoder.digest,
            "contract_digest": self.digest,
            "dataset": self.manifest.dataset.name,
            "dataset_version": self.manifest.version,
            "train_split": self.train_split,
            "evaluation_split": self.evaluation_split,
            "train_samples": len(self.split.assignment[self.train_split]),
            "evaluation_samples": len(self.split.assignment[self.evaluation_split]),
            "dt_ms": self.encoder.dt_ms,
            "timesteps": self.encoder.n_steps,
            "input_channels": self.encoder.channels,
            "input_layout": "timesteps,batch,channels",
            "late_events": "dropped",
            "training_cpu_threads": 1,
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 of the complete canonical data contract."""
        return "sha256:" + hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()


def resolve_event_training_contract(
    payload: object, *, dataset: str, timesteps: int
) -> EventTrainingContract:
    """Validate event data declarations before allocating a training job.

    Parameters
    ----------
    payload:
        Portable manifest, split, encoder and selected part names.
    dataset:
        Dataset requested by the training configuration.
    timesteps:
        Training window; must equal the declared encoder window.

    Returns
    -------
    EventTrainingContract
        Structurally verified declaration. Disk verification is a separate
        required admission step and is repeated by the training worker.

    Raises
    ------
    ValueError
        If the declaration, groups, geometry, polarity, time window or
        selected optimisation/evaluation parts cannot be honoured.
    """
    fields = {
        "schema",
        "manifest",
        "split",
        "encoder",
        "train_split",
        "evaluation_split",
        "digests",
        "training_cpu_threads",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("event_data must contain exactly the event training contract fields")
    if payload["schema"] != EVENT_TRAINING_SCHEMA:
        raise ValueError("unsupported event training data schema")
    if type(payload["training_cpu_threads"]) is not int or payload["training_cpu_threads"] != 1:
        raise ValueError("event training requires one CPU thread for reproducible reductions")
    manifest_data = _object(payload["manifest"], "manifest")
    manifest = manifest_from_dict(manifest_data)
    if _canonical(manifest.to_dict()) != _canonical(manifest_data):
        raise ValueError("manifest fields must retain their declared types and values")
    if manifest.dataset.name != dataset:
        raise ValueError("manifest dataset differs from the requested dataset")
    plan = split_plan_from_dict(_object(payload["split"], "split"))
    validate_split_plan(manifest, plan)
    declaration = _object(payload["encoder"], "encoder")
    if declaration.get("encoder") != "event-binning":
        raise ValueError("event training requires an event-binning encoder")
    for field in ("n_steps", "width", "height"):
        value = declaration.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"event encoder {field} must be an integer")
    dt = declaration.get("dt_ms")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise ValueError("event encoder dt_ms must be a number in milliseconds")
    # The factory dispatches the already checked event-binning name to this type.
    encoder = cast(EventBinning, encoder_from_declaration(declaration))
    geometry = manifest.dataset.geometry
    expected = geometry if len(geometry) == 2 else (geometry[0], 1)
    if (encoder.width, encoder.height) != expected:
        raise ValueError("encoder geometry differs from the manifest sensor")
    if manifest.dataset.polarities == 0 and encoder.polarity != "merge":
        raise ValueError("auditory events have no polarity and require merged channels")
    if encoder.n_steps != timesteps:
        raise ValueError("encoder window differs from training timesteps")
    train = payload["train_split"]
    evaluation = payload["evaluation_split"]
    if not isinstance(train, str) or not isinstance(evaluation, str):
        raise ValueError("training and evaluation parts must be named")
    if train == evaluation or train not in plan.assignment or evaluation not in plan.assignment:
        raise ValueError("training and evaluation must name distinct declared split parts")
    contract = EventTrainingContract(manifest, plan, encoder, train, evaluation)
    if payload["digests"] != contract.to_dict()["digests"]:
        raise ValueError("event data digests differ from the declared manifest, split or encoder")
    return contract


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
