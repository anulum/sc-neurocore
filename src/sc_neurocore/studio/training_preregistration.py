# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Preregistered acceptance criterion for a training run

"""Declare how a training run will be judged before it starts, then judge it.

A metric chosen after the results are known can always be made to pass. The
criterion is therefore part of the training request: it is resolved, digested
and stored with the job's configuration at submission, before a tensor exists,
and the finished run is judged against exactly that stored criterion on its
unrounded validation metric. A run that misses its criterion still completes;
the verdict says so.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from sc_neurocore.studio.training_refusals import TrainingRefusal

PREREGISTRATION_SCHEMA_VERSION = "studio.training-preregistration.v1"

#: Judged metrics and the direction in which each one passes.
#: ``conversion_accuracy_drop`` exists only on the QCFS conversion route: the
#: source network's validation accuracy minus its converted network's.
PREREGISTERED_METRICS: Mapping[str, str] = {
    "val_accuracy": "at_least",
    "val_loss": "at_most",
    "conversion_accuracy_drop": "at_most",
}

#: Metrics whose bound is a fraction in ``[0, 1]``.
_FRACTION_METRICS = frozenset({"val_accuracy", "conversion_accuracy_drop"})

#: Longest accepted rationale, which keeps the stored configuration small.
RATIONALE_MAX_CHARACTERS = 500

_FIELDS = frozenset({"schema_version", "metric", "threshold", "rationale", "sha256"})


@dataclass(frozen=True, slots=True)
class TrainingPreregistration:
    """One criterion a finished run must meet on its validation split.

    Attributes
    ----------
    metric : str
        ``val_accuracy`` (passes at or above the threshold), ``val_loss`` or
        ``conversion_accuracy_drop`` (each passes at or below it).
    threshold : float
        Finite bound; an accuracy or accuracy-drop bound lies in ``[0, 1]``,
        a loss bound is non-negative.
    rationale : str
        The hypothesis the criterion tests, as declared before the run.
    """

    metric: str
    threshold: float
    rationale: str

    @property
    def direction(self) -> str:
        """Return ``at_least`` or ``at_most`` for the declared metric."""
        return PREREGISTERED_METRICS[self.metric]

    @property
    def sha256(self) -> str:
        """Return the digest of the criterion's canonical JSON form."""
        canonical = json.dumps(
            {
                "metric": self.metric,
                "rationale": self.rationale,
                "schema_version": PREREGISTRATION_SCHEMA_VERSION,
                "threshold": self.threshold,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_public_dict(self) -> dict[str, object]:
        """Return the stored criterion, including its digest."""
        return {
            "metric": self.metric,
            "rationale": self.rationale,
            "schema_version": PREREGISTRATION_SCHEMA_VERSION,
            "sha256": self.sha256,
            "threshold": self.threshold,
        }

    def judge(self, observed: float) -> dict[str, object]:
        """Judge an unrounded validation metric against the criterion.

        Parameters
        ----------
        observed : float
            The finished run's value of :attr:`metric`.

        Returns
        -------
        dict
            The criterion, its digest, the observed value and whether it
            passed. A non-finite observation never passes and is reported as
            ``None``, because JSON has no representation for it.
        """
        finite = math.isfinite(observed)
        if not finite:
            passed = False
        elif self.direction == "at_least":
            passed = observed >= self.threshold
        else:
            passed = observed <= self.threshold
        return {
            "direction": self.direction,
            "metric": self.metric,
            "observed": observed if finite else None,
            "passed": passed,
            "preregistration_sha256": self.sha256,
            "schema_version": PREREGISTRATION_SCHEMA_VERSION,
            "threshold": self.threshold,
        }


def resolve_training_preregistration(value: object) -> TrainingPreregistration | None:
    """Resolve an optional criterion, refusing anything that cannot be judged.

    Parameters
    ----------
    value : object
        ``None`` for no criterion, or an object with ``metric``, ``threshold``,
        an optional ``rationale`` and, when resubmitting a stored criterion,
        its ``schema_version`` and ``sha256``.

    Returns
    -------
    TrainingPreregistration or None
        The criterion exactly as it will be judged.

    Raises
    ------
    ValueError
        Unknown fields, an unknown metric, a bound outside the metric's range,
        an over-long rationale, another schema version, or a stored digest that
        does not match the criterion it accompanies.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TrainingRefusal("a preregistered criterion must be an object.")
    unknown = sorted(str(key) for key in set(value) - _FIELDS)
    if unknown:
        raise TrainingRefusal(f"unknown preregistration field(s) {', '.join(unknown)}.")
    version = value.get("schema_version", PREREGISTRATION_SCHEMA_VERSION)
    if version != PREREGISTRATION_SCHEMA_VERSION:
        raise TrainingRefusal(f"{version!r} is not the preregistration contract this build reads.")
    metric = value.get("metric")
    if not isinstance(metric, str) or metric not in PREREGISTERED_METRICS:
        raise TrainingRefusal(
            f"metric must be one of {', '.join(PREREGISTERED_METRICS)}, got {metric!r}."
        )
    threshold = value.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TrainingRefusal("threshold must be a number.")
    bound = float(threshold)
    if not math.isfinite(bound) or bound < 0.0 or (metric in _FRACTION_METRICS and bound > 1.0):
        raise TrainingRefusal(
            "threshold must be finite, in [0, 1] for val_accuracy and "
            "conversion_accuracy_drop and non-negative for val_loss."
        )
    rationale = value.get("rationale", "")
    if not isinstance(rationale, str) or len(rationale) > RATIONALE_MAX_CHARACTERS:
        raise TrainingRefusal(
            f"rationale must be text of at most {RATIONALE_MAX_CHARACTERS} characters."
        )
    resolved = TrainingPreregistration(metric=metric, threshold=bound, rationale=rationale)
    declared = value.get("sha256")
    if declared is not None and declared != resolved.sha256:
        raise TrainingRefusal("the stored preregistration digest does not match its criterion.")
    return resolved


__all__ = [
    "PREREGISTERED_METRICS",
    "PREREGISTRATION_SCHEMA_VERSION",
    "RATIONALE_MAX_CHARACTERS",
    "TrainingPreregistration",
    "resolve_training_preregistration",
]
