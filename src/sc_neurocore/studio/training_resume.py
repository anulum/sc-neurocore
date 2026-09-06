# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training resume state

"""Continuing a run is not the same as starting one from its weights.

Restoring weights alone gives a **warm start**: a new run that happens to begin
from a trained network, with a fresh optimiser, a fresh generator and epoch
zero. Adam's first and second moment estimates are gone, so the first steps
after a warm start move differently from the steps that would have followed;
the shuffle order restarts; and the run reports epoch 1 of N when it is
actually further along than that.

An **exact resume** carries what a warm start drops: the optimiser's state, the
Python, NumPy and Torch generator states as they stood at the epoch boundary,
and how many epochs are already done. Resuming a run and never interrupting it
produce the same weights and the same metrics, which is the property this
module exists to make true and testable.

The two are named separately on purpose. A resume into a different
configuration or a different network is refused rather than approximated: the
state being restored belongs to a run that no longer exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any

from sc_neurocore.studio.training_contract import (
    ResolvedTrainingConfig,
    resolve_training_config,
)

#: Resume contract version. The state a resume needs is part of it, so adding
#: a required field is a new version rather than an optional extra.
TRAINING_RESUME_SCHEMA_VERSION = "studio.training-resume.v1"


class TrainingResumeMismatch(ValueError):
    """Raised when saved run state does not belong to the run being started.

    Attributes
    ----------
    field : str
        What differs — the configuration, the architecture or the schema.
    expected : str
        What the saved state describes.
    actual : str
        What the run being started describes.
    """

    def __init__(self, field: str, expected: str, actual: str) -> None:
        super().__init__(
            f"cannot resume: the saved {field} is {expected!r} but this run is {actual!r}. "
            "Start a warm run from these weights instead, which does not claim to continue."
        )
        self.field = field
        self.expected = expected
        self.actual = actual

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {
            "actual": self.actual,
            "error": "training_resume_mismatch",
            "expected": self.expected,
            "field": self.field,
            "schema_version": TRAINING_RESUME_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True)
class TrainingResumeState:
    """Everything a run needs to continue where another one stopped.

    Attributes
    ----------
    schema_version : str
        Resume contract version.
    epochs_completed : int
        How many epochs finished before this state was taken. A resume starts
        at this index, so it neither repeats nor skips one.
    architecture : str
        Layer sizes of the network the state belongs to.
    config : mapping
        The resolved configuration of the run being continued.
    optimiser_state : mapping
        The optimiser's ``state_dict``. Dropping it is what makes a warm start
        a different run: Adam's moment estimates are part of where the run is.
    rng_state : mapping
        Python, NumPy and Torch generator states at the epoch boundary, so the
        shuffle order and any stochastic layer continue rather than restart.
    dataset_fingerprint : str
        Digest of the data the run was trained on; a resume onto different
        data is a different experiment.
    """

    schema_version: str
    epochs_completed: int
    architecture: str
    config: Mapping[str, object]
    optimiser_state: Mapping[str, object]
    rng_state: Mapping[str, object]
    dataset_fingerprint: str

    def resolved_config(self) -> ResolvedTrainingConfig:
        """Return the configuration this state belongs to, re-resolved."""
        return resolve_training_config(dict(self.config))

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON-safe summary a status payload may carry.

        The optimiser and generator states are tensors and byte buffers; they
        travel in the weight checkpoint, not in a status document, so only
        what a reader can act on appears here.
        """
        return {
            "architecture": self.architecture,
            "dataset_fingerprint": self.dataset_fingerprint,
            "epochs_completed": self.epochs_completed,
            "schema_version": self.schema_version,
        }


def capture_resume_state(
    *,
    epochs_completed: int,
    architecture: str,
    config: Mapping[str, object],
    optimiser: Any,
    dataset_fingerprint: str,
) -> TrainingResumeState:
    """Take the state a later run needs in order to continue this one.

    Parameters
    ----------
    epochs_completed : int
        Epochs finished at the moment of capture.
    architecture : str
        Layer sizes of the network being trained.
    config : mapping
        The resolved configuration of the run.
    optimiser : torch.optim.Optimizer
        The live optimiser; its ``state_dict`` is copied.
    dataset_fingerprint : str
        Digest of the data the run is training on.

    Returns
    -------
    TrainingResumeState
        Captured at an epoch boundary, which is the only point where the
        generator states describe a clean position in the run.
    """
    return TrainingResumeState(
        schema_version=TRAINING_RESUME_SCHEMA_VERSION,
        epochs_completed=epochs_completed,
        architecture=architecture,
        config=dict(config),
        optimiser_state=optimiser.state_dict(),
        rng_state=_generator_states(),
        dataset_fingerprint=dataset_fingerprint,
    )


def apply_resume_state(
    state: TrainingResumeState,
    *,
    optimiser: Any,
    architecture: str,
    config: Mapping[str, object],
) -> int:
    """Restore a saved position and return the epoch to start from.

    Parameters
    ----------
    state : TrainingResumeState
        The saved position.
    optimiser : torch.optim.Optimizer
        The optimiser to load the saved state into.
    architecture : str
        Layer sizes of the network this run built.
    config : mapping
        The resolved configuration of this run.

    Returns
    -------
    int
        The epoch index to begin at.

    Raises
    ------
    TrainingResumeMismatch
        The saved state belongs to a different schema, network or
        configuration. Resuming across any of those would report a
        continuation of a run that never existed.
    """
    if state.schema_version != TRAINING_RESUME_SCHEMA_VERSION:
        raise TrainingResumeMismatch(
            "resume schema", state.schema_version, TRAINING_RESUME_SCHEMA_VERSION
        )
    if state.architecture != architecture:
        raise TrainingResumeMismatch("architecture", state.architecture, architecture)
    saved = _comparable_config(state.config)
    current = _comparable_config(config)
    if saved != current:
        raise TrainingResumeMismatch("configuration", _describe(saved), _describe(current))
    optimiser.load_state_dict(dict(state.optimiser_state))
    _restore_generator_states(state.rng_state)
    return state.epochs_completed


def resume_state_from_payload(payload: Mapping[str, object]) -> TrainingResumeState:
    """Rebuild a saved position from a weight checkpoint payload.

    Parameters
    ----------
    payload : mapping
        A loaded weight checkpoint's ``resume_state`` block.

    Returns
    -------
    TrainingResumeState
        The position the run was in when the checkpoint was written.

    Raises
    ------
    TrainingResumeMismatch
        The block is absent or was written by a different resume schema. A
        checkpoint from a build that recorded no position supports a warm
        start and nothing more, and saying so is the point.
    """
    if not isinstance(payload, Mapping) or not payload:
        raise TrainingResumeMismatch("resume state", "absent", TRAINING_RESUME_SCHEMA_VERSION)
    version = payload.get("schema_version")
    if version != TRAINING_RESUME_SCHEMA_VERSION:
        raise TrainingResumeMismatch("resume schema", str(version), TRAINING_RESUME_SCHEMA_VERSION)
    return TrainingResumeState(
        schema_version=str(version),
        epochs_completed=int(payload["epochs_completed"]),  # type: ignore[call-overload]
        architecture=str(payload["architecture"]),
        config=dict(payload["config"]),  # type: ignore[call-overload]
        optimiser_state=dict(payload["optimiser_state"]),  # type: ignore[call-overload]
        rng_state=dict(payload["rng_state"]),  # type: ignore[call-overload]
        dataset_fingerprint=str(payload["dataset_fingerprint"]),
    )


def dataset_fingerprint(loader: Any) -> str:
    """Return a digest of the data a loader will serve.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        The training loader.

    Returns
    -------
    str
        ``sha256`` over the dataset's length, the batch size, the shape and
        dtype of one sample, and the bytes of the first and last samples.

    Notes
    -----
    This is a fingerprint, not a content hash: it detects a different dataset,
    a different split boundary or a different sample layout, and it does not
    detect a change confined to the middle of a large corpus. Hashing every
    sample of a full dataset on every run would cost more than the training
    step it protects, so the boundary is stated rather than implied.
    """
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return "sha256:unavailable"
    digest = hashlib.sha256()
    length = len(dataset)
    digest.update(f"length={length}".encode())
    digest.update(f"batch_size={getattr(loader, 'batch_size', None)}".encode())
    if length == 0:
        return f"sha256:{digest.hexdigest()}"
    for index in (0, length - 1):
        sample = dataset[index]
        for element in sample if isinstance(sample, tuple) else (sample,):
            digest.update(_element_bytes(element))
    return f"sha256:{digest.hexdigest()}"


def _element_bytes(element: Any) -> bytes:
    tensor_bytes = getattr(element, "numpy", None)
    if tensor_bytes is None:
        return repr(element).encode()
    array = element.detach().cpu().numpy() if hasattr(element, "detach") else element.numpy()
    payload: bytes = array.tobytes()
    return f"{array.dtype}{array.shape}".encode() + payload


def _generator_states() -> dict[str, object]:
    """Capture the three generator states as plain data.

    Deliberately not the objects the libraries hand out: those are opaque
    tuples and arrays that only unpickle, and the weight artefact they travel
    in is loaded with ``weights_only=True`` because it arrives from a user.
    Keeping the state as integers and a hex string preserves that guarantee and
    makes a saved position readable.
    """
    import random

    import numpy as np
    import torch

    python_version, python_keys, python_gauss = random.getstate()
    numpy_state = np.random.get_state(legacy=True)
    if not isinstance(numpy_state, tuple):  # pragma: no cover - legacy=True yields a tuple
        raise TypeError("expected the legacy NumPy generator state tuple")
    return {
        "python": {
            "gauss_next": python_gauss,
            "keys": [int(key) for key in python_keys],
            "version": int(python_version),
        },
        "numpy": {
            "cached_gaussian": float(numpy_state[4]),
            "has_gauss": int(numpy_state[3]),
            "keys": [int(key) for key in numpy_state[1]],
            "kind": str(numpy_state[0]),
            "position": int(numpy_state[2]),
        },
        "torch": torch.get_rng_state().numpy().tobytes().hex(),
    }


def _restore_generator_states(states: Mapping[str, object]) -> None:
    """Put the three generators back where the captured run left them."""
    import random

    import numpy as np
    import torch

    python_state = states.get("python")
    if isinstance(python_state, Mapping):
        random.setstate(
            (
                int(python_state["version"]),
                tuple(int(key) for key in python_state["keys"]),
                python_state["gauss_next"],
            )
        )
    numpy_state = states.get("numpy")
    if isinstance(numpy_state, Mapping):
        np.random.set_state(
            (
                str(numpy_state["kind"]),
                np.array(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    torch_state = states.get("torch")
    if isinstance(torch_state, str):
        torch.set_rng_state(
            torch.frombuffer(bytearray.fromhex(torch_state), dtype=torch.uint8).clone()
        )


def _comparable_config(config: Mapping[str, object]) -> dict[str, object]:
    """Return the configuration fields a resume must match on.

    ``epochs`` is excluded deliberately: continuing a run for more epochs than
    it was first given is the ordinary reason to resume.
    """
    return {key: value for key, value in sorted(config.items()) if key != "epochs"}


def _describe(config: Mapping[str, object]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in sorted(config.items()))


__all__ = [
    "TRAINING_RESUME_SCHEMA_VERSION",
    "TrainingResumeMismatch",
    "TrainingResumeState",
    "apply_resume_state",
    "capture_resume_state",
    "dataset_fingerprint",
    "resume_state_from_payload",
]
