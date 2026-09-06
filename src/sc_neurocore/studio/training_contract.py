# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training configuration contract

"""Train what was asked for, or refuse before any work happens.

The Studio used to accept every training request and quietly train something
else. Measured through the public HTTP surface before this module existed: a run
asking for hidden widths ``[128, 64]`` on ``cifar10`` with surrogate
``not_a_real_surrogate`` completed successfully, and its exported checkpoint
recorded that request beside the architecture ``64->128->128->10`` — a network
with 128 twice and 64 nowhere, on synthetic data, using the default surrogate.
Both blocks were sealed under a ``config_sha256`` and a ``checkpoint_sha256``,
so a digest-verified artefact attested a configuration that never ran.

Every choice is therefore resolved here, before a tensor is allocated. An
unsupported dataset, an unknown surrogate, a width that is not a positive
integer or a key nobody reads is refused with the supported set named. The
resolved configuration is what the runner uses and what the checkpoint records,
so the two cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

#: Contract version. Widening the supported sets is backwards compatible;
#: changing what a field means is not.
TRAINING_CONFIG_SCHEMA_VERSION = "studio.training-config.v1"

#: Datasets the runner can actually load. ``synthetic`` is generated in
#: process; ``mnist`` is fetched through the torchvision loader.
SUPPORTED_DATASETS: tuple[str, ...] = ("synthetic", "mnist")

#: Surrogate gradients the training package exposes by name.
SUPPORTED_SURROGATES: tuple[str, ...] = (
    "fast_sigmoid",
    "superspike",
    "atan_surrogate",
    "sigmoid_surrogate",
    "straight_through",
    "triangular",
)

#: Spiking cells the training package exposes. The Studio's feedforward
#: classifier builds ``LIFCell`` layers; the rest are reachable through the
#: training package directly and are listed so a caller can see the boundary.
SUPPORTED_CELL_TYPES: tuple[str, ...] = (
    "LIFCell",
    "IFCell",
    "ALIFCell",
    "ExpIFCell",
    "AdExCell",
)

#: Every key a training request may carry, with its default.
_DEFAULTS: Mapping[str, object] = {
    "dataset": "synthetic",
    "epochs": 10,
    "batch_size": 64,
    "lr": 1e-3,
    "hidden": (128,),
    "timesteps": 25,
    "surrogate": "atan_surrogate",
    "learn_beta": False,
    "learn_threshold": False,
    "max_grad_norm": 1.0,
    "seed": 0,
}


class TrainingConfigError(ValueError):
    """Raised when a training request names something the Studio cannot run.

    Attributes
    ----------
    field : str
        The request key that was refused.
    reason : str
        What is wrong, in words a caller can act on.
    supported : tuple of str
        The accepted values, when the field has a closed set.
    """

    def __init__(self, field: str, reason: str, supported: Sequence[str] = ()) -> None:
        detail = f"{field}: {reason}"
        if supported:
            detail = f"{detail} Supported: {', '.join(supported)}."
        super().__init__(detail)
        self.field = field
        self.reason = reason
        self.supported = tuple(supported)

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {
            "error": "training_config_rejected",
            "field": self.field,
            "reason": self.reason,
            "schema_version": TRAINING_CONFIG_SCHEMA_VERSION,
            "supported": list(self.supported),
        }


@dataclass(frozen=True, slots=True)
class ResolvedTrainingConfig:
    """A training request that the runner is able to execute exactly.

    Attributes
    ----------
    dataset : str
        One of :data:`SUPPORTED_DATASETS`.
    epochs, batch_size, timesteps : int
        Positive integers.
    learning_rate, max_grad_norm : float
        Positive finite floats.
    hidden_widths : tuple of int
        One width per hidden layer, in order, each honoured as given.
    surrogate : str
        One of :data:`SUPPORTED_SURROGATES`.
    learn_beta, learn_threshold : bool
        Whether the cell parameters are learned.
    seed : int
        Seed applied to every relevant generator, so a run is replayable.
    """

    dataset: str
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_widths: tuple[int, ...]
    timesteps: int
    surrogate: str
    learn_beta: bool
    learn_threshold: bool
    max_grad_norm: float
    seed: int

    def architecture(self, n_inputs: int, n_outputs: int) -> str:
        """Return the layer sizes this configuration builds, in order.

        Parameters
        ----------
        n_inputs, n_outputs : int
            Sizes fixed by the dataset.

        Returns
        -------
        str
            ``"64->128->64->10"`` — every requested width, once each.
        """
        sizes = (n_inputs, *self.hidden_widths, n_outputs)
        return "->".join(str(size) for size in sizes)

    def to_public_dict(self) -> dict[str, object]:
        """Return the resolved configuration as the checkpoint records it."""
        return {
            "batch_size": self.batch_size,
            "dataset": self.dataset,
            "epochs": self.epochs,
            "hidden": list(self.hidden_widths),
            "learn_beta": self.learn_beta,
            "learn_threshold": self.learn_threshold,
            "lr": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "schema_version": TRAINING_CONFIG_SCHEMA_VERSION,
            "seed": self.seed,
            "surrogate": self.surrogate,
            "timesteps": self.timesteps,
        }


def resolve_training_config(payload: Mapping[str, Any]) -> ResolvedTrainingConfig:
    """Resolve a training request, refusing anything the Studio cannot run.

    Parameters
    ----------
    payload : mapping
        The request as received. Absent keys take their default; unknown keys
        are refused rather than ignored, because a silently dropped ``hiddens``
        is a request nobody honoured.

    Returns
    -------
    ResolvedTrainingConfig
        Exactly what the runner will execute.

    Raises
    ------
    TrainingConfigError
        Any field names something unsupported, is the wrong type, or is out of
        range. The refusal happens before the dataset is loaded and before a
        model is built, so a rejected request costs nothing and leaves nothing.
    """
    if not isinstance(payload, Mapping):
        raise TrainingConfigError("config", "a training request must be an object.")
    _check_schema_version(payload)
    unknown = sorted(set(payload) - set(_DEFAULTS) - {"schema_version"})
    if unknown:
        raise TrainingConfigError(
            "config",
            f"unknown request field(s) {', '.join(repr(key) for key in unknown)}.",
            tuple(sorted(_DEFAULTS)),
        )

    dataset = _choice(payload, "dataset", SUPPORTED_DATASETS)
    surrogate = _choice(payload, "surrogate", SUPPORTED_SURROGATES)
    return ResolvedTrainingConfig(
        dataset=dataset,
        epochs=_positive_int(payload, "epochs"),
        batch_size=_positive_int(payload, "batch_size"),
        learning_rate=_positive_float(payload, "lr"),
        hidden_widths=_hidden_widths(payload),
        timesteps=_positive_int(payload, "timesteps"),
        surrogate=surrogate,
        learn_beta=_flag(payload, "learn_beta"),
        learn_threshold=_flag(payload, "learn_threshold"),
        max_grad_norm=_non_negative_float(payload, "max_grad_norm"),
        seed=_seed(payload),
    )


def _check_schema_version(payload: Mapping[str, Any]) -> None:
    """Refuse a request written against a contract this build does not read.

    A resolved configuration carries its ``schema_version`` and is submitted to
    the runner unchanged, so it must resolve again — a round trip that would
    fail is a resolved configuration nobody can execute.
    """
    version = payload.get("schema_version")
    if version is None:
        return
    if version != TRAINING_CONFIG_SCHEMA_VERSION:
        raise TrainingConfigError(
            "schema_version",
            f"{version!r} is not the contract this build reads.",
            (TRAINING_CONFIG_SCHEMA_VERSION,),
        )


def _value(payload: Mapping[str, Any], key: str) -> object:
    return payload.get(key, _DEFAULTS[key])


def _choice(payload: Mapping[str, Any], key: str, supported: tuple[str, ...]) -> str:
    value = _value(payload, key)
    if not isinstance(value, str):
        raise TrainingConfigError(key, f"must be a name, got {type(value).__name__}.", supported)
    if value not in supported:
        raise TrainingConfigError(key, f"{value!r} is not supported.", supported)
    return value


def _positive_int(payload: Mapping[str, Any], key: str) -> int:
    """Return a whole count of at least one.

    No upper bound is imposed here. How much compute a run may consume is the
    bounded-admission contract's decision, and a second, weaker ceiling in this
    module would refuse legitimate long runs without protecting anything.
    """
    value = _value(payload, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingConfigError(key, f"must be an integer, got {type(value).__name__}.")
    if value < 1:
        raise TrainingConfigError(key, f"must be at least 1, got {value}.")
    return value


def _positive_float(payload: Mapping[str, Any], key: str) -> float:
    value = _value(payload, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(key, f"must be a number, got {type(value).__name__}.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise TrainingConfigError(key, f"must be a positive finite number, got {value!r}.")
    return number


def _non_negative_float(payload: Mapping[str, Any], key: str) -> float:
    """Return a finite value at or above zero.

    Zero is meaningful for gradient clipping: it clips every gradient to zero,
    which runs the loop without learning. Refusing it would remove a
    capability the Studio has always had.
    """
    value = _value(payload, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(key, f"must be a number, got {type(value).__name__}.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise TrainingConfigError(key, f"must be a finite number at or above 0, got {value!r}.")
    return number


def _flag(payload: Mapping[str, Any], key: str) -> bool:
    value = _value(payload, key)
    if not isinstance(value, bool):
        raise TrainingConfigError(key, f"must be true or false, got {type(value).__name__}.")
    return value


def _seed(payload: Mapping[str, Any]) -> int:
    value = _value(payload, "seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingConfigError("seed", f"must be an integer, got {type(value).__name__}.")
    if not 0 <= value < 2**32:
        raise TrainingConfigError("seed", f"must be in [0, 2**32), got {value}.")
    return value


def _hidden_widths(payload: Mapping[str, Any]) -> tuple[int, ...]:
    value = _value(payload, "hidden")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TrainingConfigError(
            "hidden", f"must be a list of layer widths, got {type(value).__name__}."
        )
    # An empty list is a request for the direct input-to-output layer, not a
    # mistake: it is what the Studio has always built for `hidden: []`.
    widths = tuple(value)
    resolved: list[int] = []
    for index, width in enumerate(widths):
        if isinstance(width, bool) or not isinstance(width, int):
            raise TrainingConfigError(
                "hidden", f"layer {index} width must be an integer, got {type(width).__name__}."
            )
        if width < 1:
            raise TrainingConfigError("hidden", f"layer {index} width must be at least 1.")
        resolved.append(width)
    return tuple(resolved)


__all__ = [
    "SUPPORTED_CELL_TYPES",
    "SUPPORTED_DATASETS",
    "SUPPORTED_SURROGATES",
    "TRAINING_CONFIG_SCHEMA_VERSION",
    "ResolvedTrainingConfig",
    "TrainingConfigError",
    "resolve_training_config",
]
