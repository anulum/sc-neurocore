# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event training input verification and runtime startup

"""Verify event custody and start the selected decoder before Torch loading."""

from __future__ import annotations

import os
from pathlib import Path

from sc_neurocore.accel.event_recordings import decode_nmnist_recording
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.studio.event_training_budget import admit_event_training_input
from sc_neurocore.studio.event_training_contract import EventTrainingContract

DATASET_ROOT_ENV = "SC_NEUROCORE_STUDIO_DATASET_ROOT"


def verify_event_training_data(contract: EventTrainingContract) -> Path:
    """Verify dataset files and sample metadata against the declared manifest.

    Parameters
    ----------
    contract:
        Structurally resolved input declaration.

    Returns
    -------
    pathlib.Path
        Absolute operator-configured root. It is never supplied by an HTTP
        request or included in exported checkpoints.

    Raises
    ------
    ValueError
        If the operator has not configured a root, the expected files are
        unavailable, or their digests, labels or groups differ.

    Notes
    -----
    Rebuilding the manifest checks sample metadata as well as file bytes.
    This is a content check, not an immutable filesystem snapshot; the
    operator must keep the dataset unchanged while training runs.
    """
    configured = os.environ.get(DATASET_ROOT_ENV)
    if not configured:
        raise ValueError("the operator has not configured a local event dataset root")
    root = Path(configured).resolve()
    if not root.is_dir():
        raise ValueError("the configured event dataset root is unavailable")
    for record in contract.manifest.files:
        if not (root / record.path).resolve().is_relative_to(root):
            raise ValueError("event dataset file lies outside the configured root")
    try:
        actual = build_manifest(
            contract.manifest.dataset.name, root, version=contract.manifest.version
        )
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("the configured event dataset could not be verified") from exc
    if actual.digest != contract.manifest.digest:
        raise ValueError("event dataset files or sample metadata differ from the manifest")
    for record in actual.files:
        if not (root / record.path).resolve().is_relative_to(root):
            raise ValueError("event dataset file lies outside the configured root")
    return root


def _prepare_event_training_runtime(contract: EventTrainingContract, batch_size: int) -> None:
    """Verify input admission and initialise the selected N-MNIST decoder.

    Empty input uses the recording dispatcher's normal benchmark order and
    operator opt-in. The first JuliaCall import must precede Torch, including
    checkpoint deserialisation. Workers invoke this only after their grant and
    resource limits; SHD and DVS readers do not embed JuliaCall.
    """
    admit_event_training_input(contract, batch_size)
    verify_event_training_data(contract)
    if contract.manifest.dataset.name == "nmnist":
        decode_nmnist_recording(b"")
