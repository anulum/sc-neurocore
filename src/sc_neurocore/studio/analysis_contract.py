# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Metric contracts of Studio analysis results

"""Metric contracts attached to every Studio analysis payload.

An analysis number is only evidence when the reader knows what was computed,
in which unit, under which conditions it is meaningful and where the
computation could not be carried out. The :class:`MetricContract` states
these four things next to the numbers, and the ``domain`` verdict says
whether the analysis covered its whole requested domain (``complete``), only
part of it (``partial``, with the invalid part reported) or none of it
(``empty``). An invalid part of a domain is never reported as a zero-valued
success.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

METRIC_CONTRACT_SCHEMA_VERSION = "studio.metric-contract.v1"

DomainStatus: TypeAlias = Literal["complete", "partial", "empty"]
MODEL_DEFINED_UNIT = "model-defined"

_DOMAIN_STATUSES: frozenset[str] = frozenset({"complete", "partial", "empty"})


@dataclass(frozen=True, slots=True)
class MetricContract:
    """What one analysis computed, in which units, and where it is valid.

    Parameters
    ----------
    kind:
        Stable identifier of the metric family (``"fi-curve"``,
        ``"precision-compare"``, ``"nullclines"``, …).
    definition:
        One-sentence statement of how the reported numbers are computed.
    units:
        Unit of every reported quantity, keyed by the payload field or the
        quantity name; ``"model-defined"`` when the equation system carries no
        declared unit.
    applicability:
        Conditions under which the metric is meaningful.
    limitations:
        Known limits of the computation (resolution, transients, protocol).
    domain:
        ``complete`` when every requested point was evaluated, ``partial``
        when some points were invalid and are reported, ``empty`` when no
        point could be evaluated.
    domain_detail:
        Path-free detail of the invalid part (counts, fractions, reasons).
    """

    kind: str
    definition: str
    units: Mapping[str, str]
    applicability: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    domain: DomainStatus = "complete"
    domain_detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject an empty kind or definition and an unknown domain verdict."""
        if not self.kind:
            raise ValueError("metric contract kind must not be empty")
        if not self.definition:
            raise ValueError("metric contract definition must not be empty")
        if self.domain not in _DOMAIN_STATUSES:
            raise ValueError(f"unknown metric domain {self.domain!r}")

    def to_public_dict(self) -> dict[str, Any]:
        """Return the path-free public contract block."""
        return {
            "schema_version": METRIC_CONTRACT_SCHEMA_VERSION,
            "kind": self.kind,
            "definition": self.definition,
            "units": dict(self.units),
            "applicability": list(self.applicability),
            "limitations": list(self.limitations),
            "domain": self.domain,
            "domain_detail": dict(self.domain_detail),
        }


def attach_contract(payload: dict[str, Any], contract: MetricContract) -> dict[str, Any]:
    """Set ``payload["contract"]`` and return the same payload."""
    payload["contract"] = contract.to_public_dict()
    return payload


def contract_summary(payload: Mapping[str, Any]) -> tuple[str | None, DomainStatus | None]:
    """Return ``(kind, domain)`` of a payload's contract block, or ``(None, None)``.

    The analysis manifest records these two values so an evidence bundle can
    tell a complete-domain result from a partial one without opening the
    payload.
    """
    block = payload.get("contract")
    if not isinstance(block, Mapping):
        return None, None
    kind = block.get("kind")
    domain = block.get("domain")
    if not isinstance(kind, str) or not kind:
        return None, None
    if domain not in _DOMAIN_STATUSES:
        raise ValueError(f"payload contract carries an unknown domain {domain!r}")
    return kind, domain


__all__ = [
    "METRIC_CONTRACT_SCHEMA_VERSION",
    "MODEL_DEFINED_UNIT",
    "DomainStatus",
    "MetricContract",
    "attach_contract",
    "contract_summary",
]
