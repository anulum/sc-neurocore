# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio trusted training weight loader

"""Trusted, fail-closed loader for portable Training Monitor weight payloads."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO

from sc_neurocore.studio.platform.training_weights import (
    STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
)


def load_training_weight_state_dict(payload: bytes) -> Mapping[str, object]:
    """Deserialize a verified torch checkpoint payload into a state dictionary.

    The loader is the trusted boundary that runs only after
    :func:`sc_neurocore.studio.platform.training_weights.materialize_training_weight_payload`
    has verified the payload digest and byte size against the restore plan. It
    restricts unpickling to tensor storages and primitive containers through
    ``weights_only=True`` and rejects any payload that does not carry the
    expected portable training checkpoint schema or a string-keyed model state
    dictionary.

    Parameters
    ----------
    payload:
        Raw bytes of a ``studio.training.torch-state-dict.v1`` checkpoint that
        was produced by the Studio Training Monitor and already passed digest
        verification.

    Returns
    -------
    Mapping[str, object]
        The string-keyed model state dictionary extracted from the checkpoint.

    Raises
    ------
    ValueError
        If the payload cannot be safely deserialized, does not carry the
        expected schema, or does not contain a string-keyed model state
        dictionary.
    """

    import torch  # local import: torch is an optional research dependency.

    try:
        loaded = torch.load(BytesIO(payload), weights_only=True, map_location="cpu")
    except Exception as exc:  # torch raises many concrete deserialization types.
        raise ValueError("Training weight payload could not be deserialized.") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("Training weight payload is not a checkpoint object.")
    if loaded.get("schema_version") != STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION:
        raise ValueError("Training weight payload schema is unsupported.")
    state_dict = loaded.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Training weight payload is missing a model state dict.")
    for key in state_dict:
        if not isinstance(key, str) or not key:
            raise ValueError("Training weight payload has an invalid state key.")
    return dict(state_dict)


__all__ = ["load_training_weight_state_dict"]
