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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sc_neurocore.datasets.refusals import DatasetRefusal
from sc_neurocore.studio.event_training_budget import admit_event_training_input
from sc_neurocore.studio.event_training_contract import (
    EventTrainingContract,
    resolve_event_training_contract,
)
from sc_neurocore.studio.training_preregistration import (
    TrainingPreregistration,
    resolve_training_preregistration,
)
from sc_neurocore.studio.training_refusals import TrainingRefusal

#: Contract version. Widening the supported sets is backwards compatible;
#: changing what a field means is not.
TRAINING_CONFIG_SCHEMA_VERSION = "studio.training-config.v1"

#: Datasets the runner can actually load. ``synthetic`` is generated in
#: process; ``mnist`` is fetched through the torchvision loader.
SUPPORTED_DATASETS: tuple[str, ...] = ("synthetic", "mnist", "nmnist", "shd", "dvs_cifar10")

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

#: What a run trains. ``spiking`` trains the surrogate-gradient spiking
#: classifier directly. ``qcfs_conversion`` trains an ANN whose activations are
#: QCFS quantisers, converts it to a dense integrate-and-fire network and
#: judges the converted network on the validation split.
SUPPORTED_MODEL_KINDS: tuple[str, ...] = ("spiking", "qcfs_conversion")

#: Static datasets the conversion route encodes as rates in ``[0, 1]``.
CONVERSION_DATASETS: tuple[str, ...] = ("synthetic", "mnist")

#: Largest QCFS step budget every QCFS runtime accepts.
_QCFS_STEP_LIMIT = 2**32 - 1

#: Keys that configure spiking cells, which the conversion route does not build.
_SPIKING_ONLY = ("surrogate", "learn_beta", "learn_threshold")

#: Every key a training request may carry, with its default.
_DEFAULTS: Mapping[str, object] = {
    "model_kind": "spiking",
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
    "event_data": None,
    "preregistration": None,
    "target_profile": None,
}


class TrainingConfigError(TrainingRefusal):
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
        reason = reason.encode("utf-8", "backslashreplace").decode("utf-8")
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
    event_data : EventTrainingContract or None
        Manifest-bound temporal input for an event dataset; absent for static data.
    preregistration : TrainingPreregistration or None
        The acceptance criterion declared before the run; stored with the
        configuration at submission and judged on the finished run.
    model_kind : str
        One of :data:`SUPPORTED_MODEL_KINDS`. On ``qcfs_conversion``,
        ``timesteps`` is the QCFS step budget and the converted network's
        timestep budget, and the surrogate and cell flags keep their unused
        defaults.
    target_profile : str or None
        A registered hardware profile the converted network is calibrated
        for; only on ``qcfs_conversion``.
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
    event_data: EventTrainingContract | None = None
    preregistration: TrainingPreregistration | None = None
    model_kind: str = "spiking"
    target_profile: str | None = None

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
        """Return the resolved configuration as the checkpoint records it.

        A spiking run records no ``model_kind``, so its configuration and
        digest are what they were before the conversion route existed. A
        conversion run records its kind and omits the cell settings it never
        reads.
        """
        result: dict[str, object] = {
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
        if self.event_data is not None:
            result["event_data"] = self.event_data.to_dict()
        if self.preregistration is not None:
            result["preregistration"] = self.preregistration.to_public_dict()
        if self.target_profile is not None:
            result["target_profile"] = self.target_profile
        if self.model_kind != "spiking":
            result["model_kind"] = self.model_kind
            for key in _SPIKING_ONLY:
                del result[key]
        return result


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
    model_kind = _choice(payload, "model_kind", SUPPORTED_MODEL_KINDS)
    surrogate = _choice(payload, "surrogate", SUPPORTED_SURROGATES)
    timesteps = _positive_int(payload, "timesteps")
    batch_size = _positive_int(payload, "batch_size")
    if model_kind == "qcfs_conversion":
        _check_conversion_request(payload, dataset, timesteps)
    event_data = None
    if dataset not in ("synthetic", "mnist"):
        try:
            event_data = resolve_event_training_contract(
                payload.get("event_data"), dataset=dataset, timesteps=timesteps
            )
            admit_event_training_input(event_data, batch_size)
        except (TrainingRefusal, DatasetRefusal) as exc:
            raise TrainingConfigError("event_data", str(exc)) from exc
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            raise TrainingConfigError(
                "event_data", "the event data declaration is malformed."
            ) from exc
    elif payload.get("event_data") is not None:
        raise TrainingConfigError(
            "event_data", "static datasets do not accept event data contracts."
        )
    try:
        preregistration = resolve_training_preregistration(payload.get("preregistration"))
    except TrainingRefusal as exc:
        raise TrainingConfigError("preregistration", str(exc)) from exc
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        raise TrainingConfigError(
            "preregistration", "the criterion declaration is malformed."
        ) from exc
    if model_kind != "qcfs_conversion" and (
        preregistration is not None and preregistration.metric == "conversion_accuracy_drop"
    ):
        raise TrainingConfigError(
            "preregistration",
            "conversion_accuracy_drop is measured only on the qcfs_conversion route.",
        )
    return ResolvedTrainingConfig(
        dataset=dataset,
        epochs=_positive_int(payload, "epochs"),
        batch_size=batch_size,
        learning_rate=_positive_float(payload, "lr"),
        hidden_widths=_hidden_widths(payload),
        timesteps=timesteps,
        surrogate=surrogate,
        learn_beta=_flag(payload, "learn_beta"),
        learn_threshold=_flag(payload, "learn_threshold"),
        max_grad_norm=_non_negative_float(payload, "max_grad_norm"),
        seed=_seed(payload),
        event_data=event_data,
        preregistration=preregistration,
        model_kind=model_kind,
        target_profile=_target_profile(payload, model_kind),
    )


def _target_profile(payload: Mapping[str, Any], model_kind: str) -> str | None:
    """Return the registered profile a conversion run is calibrated for, if any."""
    value = payload.get("target_profile")
    if value is None:
        return None
    if model_kind != "qcfs_conversion":
        raise TrainingConfigError(
            "target_profile", "only a qcfs_conversion run has a network to calibrate."
        )
    if not isinstance(value, str):
        raise TrainingConfigError(
            "target_profile", f"must be a profile name, got {type(value).__name__}."
        )
    from sc_neurocore.compiler.platforms import get_profile

    try:
        return get_profile(value).name
    except KeyError as exc:
        raise TrainingConfigError(
            "target_profile", f"{value!r} is not a registered hardware profile."
        ) from exc


def list_target_profiles() -> list[dict[str, object]]:
    """Return the hardware profiles a conversion run can be calibrated for.

    Returns
    -------
    list of dict
        Name, vendor, family, class and fixed-point format of every
        registered profile, ordered by name.
    """
    from sc_neurocore.compiler.platforms import list_profiles

    return [
        {
            "name": profile.name,
            "vendor": profile.vendor,
            "family": profile.family,
            "platform_class": profile.platform_class,
            "q_format": profile.q_format_label,
            "data_width": profile.data_width,
            "fraction": profile.fraction,
            "signed": profile.signed,
        }
        for profile in sorted(list_profiles(), key=lambda profile: profile.name)
    ]


def _check_conversion_request(payload: Mapping[str, Any], dataset: str, timesteps: int) -> None:
    """Refuse a conversion request the conversion route would not honour exactly."""
    if dataset not in CONVERSION_DATASETS:
        raise TrainingConfigError(
            "dataset",
            f"the qcfs_conversion route encodes static samples as rates; {dataset!r} is an "
            "event dataset.",
            CONVERSION_DATASETS,
        )
    for key in _SPIKING_ONLY:
        if key in payload:
            raise TrainingConfigError(
                key, "the qcfs_conversion route builds no spiking cells to configure."
            )
    if timesteps > _QCFS_STEP_LIMIT:
        raise TrainingConfigError(
            "timesteps", f"a QCFS step budget is at most {_QCFS_STEP_LIMIT}, got {timesteps}."
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
    """Return a positive finite float, refusing unrepresentable numeric inputs."""
    value = _value(payload, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(key, f"must be a number, got {type(value).__name__}.")
    try:
        number = float(value)
    except OverflowError as exc:
        raise TrainingConfigError(key, "must be a positive finite number.") from exc
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
    try:
        number = float(value)
    except OverflowError as exc:
        raise TrainingConfigError(key, "must be a finite number at or above 0.") from exc
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
    "CONVERSION_DATASETS",
    "SUPPORTED_CELL_TYPES",
    "SUPPORTED_DATASETS",
    "SUPPORTED_MODEL_KINDS",
    "SUPPORTED_SURROGATES",
    "TRAINING_CONFIG_SCHEMA_VERSION",
    "ResolvedTrainingConfig",
    "TrainingConfigError",
    "list_target_profiles",
    "resolve_training_config",
]
