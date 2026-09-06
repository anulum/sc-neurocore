# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference-trace independence adjudication

"""What a reference trace is independent *of*, decided per identity.

Backends agreeing with each other can reproduce one shared scientific error, so
a corpus is only an oracle to the extent that its expected values were produced
without executing the implementation under test — and, separately, to the extent
that the formulation itself comes from somewhere other than this repository.

Those are two different questions and this module keeps them apart:

* **Derivation** — how the expected values were produced. A re-derivation from
  the equations, a closed-form solution, values read out of the publication, a
  pinned third-party implementation, or a transcription of the project's own
  retained recurrence.
* **Attribution** — what the formulation is sourced from. A resolvable external
  citation, a publication named without a resolvable locator, or a formulation
  the project retains with no whole-model publication behind it.

A trace re-derived by hand from a recurrence this repository invented is
independent of the *implementation* but has no external source to be
independent of. Calling that "independent" without qualification is the
overstatement this adjudication exists to prevent: four traces in the current
corpus declare an independent or analytic derivation while citing a recurrence
this project retains, and the citation is what settles it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from sc_neurocore.neurons.reference_trace_contracts import ReferenceTraceSpec

#: Adjudication contract version. A change of class semantics changes this.
REFERENCE_TRACE_ADJUDICATION_VERSION = "sc-neurocore.reference-trace-adjudication.v1"

Derivation = Literal[
    "independent_rederivation",
    "analytic_solution",
    "published_data",
    "external_implementation",
    "project_transcription",
]

Attribution = Literal["resolvable_source", "named_source", "project_retained"]

IndependenceClass = Literal[
    "published_source",
    "published_data",
    "external_implementation",
    "project_formulation",
]

#: Citation prefixes that resolve to something outside this repository.
_RESOLVABLE_PREFIXES = ("doi:", "http://", "https://")

#: Citation markers that name the project's own retained formulation.
_PROJECT_MARKERS = ("project:", "sc-neurocore")

#: Every ``provenance.kind`` the corpus may declare, and how it was produced.
#:
#: Fail-closed on purpose: a trace introducing a new kind must add it here, and
#: adding it is a decision about what the trace proves rather than a string.
_DERIVATION_BY_KIND: Mapping[str, Derivation] = MappingProxyType(
    {
        "analytic_closed_form": "analytic_solution",
        "analytic_exact_flow_reference": "analytic_solution",
        "analytic_exact_integral": "analytic_solution",
        "analytic_exact_linear_flow_reference": "analytic_solution",
        "analytic_exact_relaxation_reference": "analytic_solution",
        "independent_analytic_and_exhaustive_lfsr_reference": "analytic_solution",
        "independent_appendix_euler_ou_reference": "independent_rederivation",
        "independent_coupled_rk4_reference": "independent_rederivation",
        "independent_equation_6_euler_reference": "independent_rederivation",
        "independent_equations_12_euler_reference": "independent_rederivation",
        "independent_euler_reference": "independent_rederivation",
        "independent_exact_flow_reference": "independent_rederivation",
        "independent_macrostep_gauss_seidel_reference": "independent_rederivation",
        "independent_macrostep_rk4_reference": "independent_rederivation",
        "independent_rk4_reference": "independent_rederivation",
        "map_iteration_reference": "independent_rederivation",
        "source_equation_reference": "independent_rederivation",
        "published_logical_rule": "published_data",
        "source_implementation_reference": "external_implementation",
        "project_recurrence_reference": "project_transcription",
        "project_regression": "project_transcription",
        "project_rk4_reference": "project_transcription",
        "sc_project_compatibility": "project_transcription",
    }
)

#: What each independence class does and does not license as a claim.
_CLASS_MEANING: Mapping[IndependenceClass, str] = MappingProxyType(
    {
        "published_source": (
            "Re-derived from an externally published formulation without executing the "
            "implementation under test."
        ),
        "published_data": "Values taken from the publication itself, not computed here.",
        "external_implementation": (
            "Reproduced from a pinned third-party implementation, which is independent of "
            "this repository but is an implementation rather than the publication."
        ),
        "project_formulation": (
            "Transcribed from a formulation this repository retains, with no whole-model "
            "publication to be independent of. A regression contract, not an external oracle."
        ),
    }
)


class ReferenceTraceAdjudicationError(ValueError):
    """Raised when a trace's provenance cannot be adjudicated."""


@dataclass(frozen=True, slots=True)
class SpecAdjudication:
    """How independent one reference trace is, and of what.

    Attributes
    ----------
    name : str
        Corpus identifier of the adjudicated trace.
    kind : str
        The ``provenance.kind`` the trace declares.
    derivation : str
        How the expected values were produced.
    attribution : str
        What the formulation is sourced from.
    independence : str
        The class a public claim about this trace may use.
    citation : str or None
        The citation as declared.
    """

    name: str
    kind: str
    derivation: Derivation
    attribution: Attribution
    independence: IndependenceClass
    citation: str | None

    @property
    def meaning(self) -> str:
        """Return what this trace's independence class licenses as a claim."""
        return _CLASS_MEANING[self.independence]

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-safe row for reports and generated documentation."""
        return {
            "attribution": self.attribution,
            "citation": self.citation,
            "derivation": self.derivation,
            "independence": self.independence,
            "kind": self.kind,
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class CorpusAdjudication:
    """The corpus-wide honest replication boundary.

    Attributes
    ----------
    version : str
        Adjudication contract version.
    rows : tuple of SpecAdjudication
        One adjudication per deterministic corpus trace, by name.
    """

    version: str
    rows: tuple[SpecAdjudication, ...]

    @property
    def counts(self) -> Mapping[IndependenceClass, int]:
        """Return how many traces fall in each independence class."""
        tally: dict[IndependenceClass, int] = {
            "published_source": 0,
            "published_data": 0,
            "external_implementation": 0,
            "project_formulation": 0,
        }
        for row in self.rows:
            tally[row.independence] += 1
        return MappingProxyType(tally)

    def by_class(self, independence: IndependenceClass) -> tuple[str, ...]:
        """Return the trace names in one independence class, sorted."""
        return tuple(sorted(row.name for row in self.rows if row.independence == independence))

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-safe report of the whole adjudication."""
        return {
            "counts": dict(self.counts),
            "rows": [row.to_public_dict() for row in self.rows],
            "total": len(self.rows),
            "version": self.version,
        }


def adjudicated_kinds() -> tuple[str, ...]:
    """Return every ``provenance.kind`` this build knows how to adjudicate."""
    return tuple(sorted(_DERIVATION_BY_KIND))


def derivation_of_kind(kind: str) -> Derivation:
    """Return how a declared ``provenance.kind`` produced its expected values.

    Parameters
    ----------
    kind : str
        The ``provenance.kind`` a trace declares.

    Returns
    -------
    str
        The derivation recorded for that kind.

    Raises
    ------
    ReferenceTraceAdjudicationError
        The kind is not in the adjudicated vocabulary. A new kind is a decision
        about what a trace proves, so it is declared here rather than accepted
        as free text.
    """
    derivation = _DERIVATION_BY_KIND.get(kind)
    if derivation is None:
        raise ReferenceTraceAdjudicationError(
            f"reference trace provenance kind {kind!r} is not adjudicated; "
            f"known kinds are {', '.join(adjudicated_kinds())}"
        )
    return derivation


def classify_citation(citation: str | None) -> Attribution:
    """Return what a citation string attributes the formulation to.

    Parameters
    ----------
    citation : str or None
        The declared citation.

    Returns
    -------
    str
        ``"resolvable_source"`` for a DOI or URL, ``"project_retained"`` when the
        citation names this project's own recurrence, and ``"named_source"``
        otherwise — a publication named without a locator resolvable from here.

    Raises
    ------
    ReferenceTraceAdjudicationError
        The citation is absent. Every trace states what it is derived from;
        an unstated source cannot be adjudicated, and silently treating it as
        external is the overstatement this module exists to prevent.
    """
    if citation is None or not citation.strip():
        raise ReferenceTraceAdjudicationError(
            "a reference trace must declare a citation before it can be adjudicated"
        )
    lowered = citation.strip().lower()
    if lowered.startswith(_RESOLVABLE_PREFIXES):
        return "resolvable_source"
    if any(marker in lowered for marker in _PROJECT_MARKERS):
        return "project_retained"
    return "named_source"


def adjudicate_spec(spec: ReferenceTraceSpec) -> SpecAdjudication:
    """Adjudicate one reference trace's independence.

    Parameters
    ----------
    spec : ReferenceTraceSpec
        A loaded deterministic corpus trace.

    Returns
    -------
    SpecAdjudication
        Derivation, attribution and the independence class a claim may use.

    Raises
    ------
    ReferenceTraceAdjudicationError
        The declared ``provenance.kind`` is not in the adjudicated vocabulary,
        or no citation is declared.

    Notes
    -----
    A citation naming the project's own recurrence demotes the trace to
    ``project_formulation`` whatever the declared kind says. A hand re-derivation
    of a recurrence this repository invented is independent of the
    implementation, but there is no published formulation for it to be
    independent of.
    """
    derivation = derivation_of_kind(spec.provenance.kind)
    attribution = classify_citation(spec.provenance.citation)
    independence = _independence_class(derivation, attribution)
    return SpecAdjudication(
        name=spec.name,
        kind=spec.provenance.kind,
        derivation=derivation,
        attribution=attribution,
        independence=independence,
        citation=spec.provenance.citation,
    )


def adjudicate_corpus() -> CorpusAdjudication:
    """Adjudicate every deterministic trace in the committed corpus.

    Returns
    -------
    CorpusAdjudication
        The honest replication boundary: which identities rest on a published
        source and which rest on this repository's own formulation.

    Raises
    ------
    ReferenceTraceAdjudicationError
        Any trace declares an unadjudicated kind or no citation.
    """
    from sc_neurocore.neurons.reference_trace_io import (
        list_reference_trace_specs,
        load_reference_trace_spec,
    )

    rows = tuple(
        adjudicate_spec(load_reference_trace_spec(name))
        for name in sorted(list_reference_trace_specs())
    )
    return CorpusAdjudication(version=REFERENCE_TRACE_ADJUDICATION_VERSION, rows=rows)


def _independence_class(derivation: Derivation, attribution: Attribution) -> IndependenceClass:
    if derivation == "project_transcription" or attribution == "project_retained":
        return "project_formulation"
    if derivation == "published_data":
        return "published_data"
    if derivation == "external_implementation":
        return "external_implementation"
    return "published_source"


__all__ = [
    "REFERENCE_TRACE_ADJUDICATION_VERSION",
    "Attribution",
    "CorpusAdjudication",
    "Derivation",
    "IndependenceClass",
    "ReferenceTraceAdjudicationError",
    "SpecAdjudication",
    "adjudicate_corpus",
    "adjudicate_spec",
    "adjudicated_kinds",
    "classify_citation",
    "derivation_of_kind",
]
