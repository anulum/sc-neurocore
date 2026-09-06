# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference-trace independence adjudication

"""Every reference trace must say what it is a reference *for*.

The corpus is only an oracle where its expected values came from somewhere
other than the implementation under test, and where the formulation itself came
from somewhere other than this repository. These cases hold that adjudication
to the committed corpus, including the traces whose declared kind claims more
than their citation supports.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from sc_neurocore.neurons.reference_trace_contracts import ReferenceTraceProvenance
from sc_neurocore.neurons.reference_trace_io import (
    list_reference_trace_specs,
    load_reference_trace_spec,
    reference_trace_spec_from_payload,
)
from sc_neurocore.neurons.reference_trace_provenance import (
    REFERENCE_TRACE_ADJUDICATION_VERSION,
    ReferenceTraceAdjudicationError,
    adjudicate_corpus,
    adjudicate_spec,
    adjudicated_kinds,
    classify_citation,
    derivation_of_kind,
)

#: Traces whose formulation this repository retains, with no whole-model
#: publication to be independent of. They are regression contracts, and a
#: public page may not call them independent replications.
PROJECT_FORMULATION_TRACES = (
    "sc_clipped_logistic_bursting_map_project",
    "sc_clipped_rational_recovery_map_project",
    "sc_four_state_glif_constant_current_adaptation",
    "sc_lapicque_lif_constant_current_closed_form",
    "sc_perfect_integrator_constant_current_sawtooth",
    "sc_resetting_wilson_hr_project",
    "sc_scaled_reset_adaptive_if_driven_project",
    "sc_symmetric_quadratic_if_zero_current_analytic",
    "sc_triangular_mckean_project",
)

#: Traces whose declared kind reads as independent or analytic while the
#: citation names the project's own recurrence. The citation settles it.
DEMOTED_BY_CITATION = (
    "sc_lapicque_lif_constant_current_closed_form",
    "sc_resetting_wilson_hr_project",
    "sc_symmetric_quadratic_if_zero_current_analytic",
    "sc_triangular_mckean_project",
)


class TestVocabulary:
    def test_every_committed_kind_is_adjudicated(self) -> None:
        """A trace cannot introduce an unreviewed provenance word."""
        declared = {
            load_reference_trace_spec(name).provenance.kind for name in list_reference_trace_specs()
        }

        assert declared <= set(adjudicated_kinds())

    def test_an_unknown_kind_is_refused_rather_than_accepted_as_free_text(self) -> None:
        with pytest.raises(ReferenceTraceAdjudicationError, match="not adjudicated"):
            derivation_of_kind("looks_independent_to_me")

    def test_the_loader_refuses_a_payload_with_an_unadjudicated_kind(self) -> None:
        """Fail-closed at the corpus boundary, not only in the report."""
        payload = {
            "schema_version": "sc-neurocore.reference-trace.v1",
            "name": "invented",
            "model": {"schema_name": "adex", "runner": "universal_dsl"},
            "protocol": {"dt": 0.1, "steps": 2, "inputs": {"I": 0.0}, "state_variables": ["v"]},
            "provenance": {
                "kind": "totally_independent_honest",
                "source": "nowhere",
                "equation": "none",
            },
            "expected_features": {"spike_count": 0.0},
            "tolerances": {"default": {"absolute": 1e-9, "relative": 0.0}},
        }

        with pytest.raises(ReferenceTraceAdjudicationError, match="not adjudicated"):
            reference_trace_spec_from_payload(payload)


class TestCitationAttribution:
    @pytest.mark.parametrize(
        ("citation", "expected"),
        [
            ("doi:10.1152/jn.00686.2005", "resolvable_source"),
            ("https://example.org/paper", "resolvable_source"),
            ("project:sc-neurocore-four-state-glif", "project_retained"),
            ("SC-NeuroCore retained project recurrence", "project_retained"),
            ("Lapicque 1907 schema metadata", "named_source"),
        ],
    )
    def test_a_citation_is_classified_by_what_it_resolves_to(
        self, citation: str, expected: str
    ) -> None:
        assert classify_citation(citation) == expected

    @pytest.mark.parametrize("citation", [None, "", "   "])
    def test_an_absent_citation_cannot_be_adjudicated(self, citation: str | None) -> None:
        """Silence is not an external source."""
        with pytest.raises(ReferenceTraceAdjudicationError, match="must declare a citation"):
            classify_citation(citation)


class TestAdjudication:
    def test_a_project_citation_demotes_an_independent_claim(self) -> None:
        """A hand re-derivation of our own recurrence is not a published source.

        The values were still produced without executing the candidate, so the
        derivation stands; what changes is that there is no publication for it
        to be independent of.
        """
        spec = load_reference_trace_spec("sc_triangular_mckean_project")

        adjudication = adjudicate_spec(spec)

        assert adjudication.kind == "independent_rk4_reference"
        assert adjudication.derivation == "independent_rederivation"
        assert adjudication.attribution == "project_retained"
        assert adjudication.independence == "project_formulation"
        assert "regression contract" in adjudication.meaning

    def test_a_doi_backed_rederivation_is_a_published_source(self) -> None:
        adjudication = adjudicate_spec(load_reference_trace_spec("adex_resting_adaptation_doi"))

        assert adjudication.attribution == "resolvable_source"
        assert adjudication.independence == "published_source"

    def test_the_pinned_third_party_implementation_is_not_the_publication(self) -> None:
        adjudication = adjudicate_spec(load_reference_trace_spec("iqif_a8752eb_tutorial"))

        assert adjudication.independence == "external_implementation"
        assert "implementation rather than the publication" in adjudication.meaning

    def test_published_values_are_not_a_rederivation(self) -> None:
        adjudication = adjudicate_spec(
            load_reference_trace_spec("mcculloch_pitts_1943_truth_table")
        )

        assert adjudication.derivation == "published_data"
        assert adjudication.independence == "published_data"

    def test_a_project_kind_stays_project_even_with_an_external_citation(self) -> None:
        """Transcribing our own recurrence is not independent of anything."""
        spec = load_reference_trace_spec("sc_four_state_glif_constant_current_adaptation")
        relabelled = replace(
            spec,
            provenance=ReferenceTraceProvenance(
                kind=spec.provenance.kind,
                source=spec.provenance.source,
                equation=spec.provenance.equation,
                citation="doi:10.0000/not-really",
            ),
        )

        assert adjudicate_spec(relabelled).independence == "project_formulation"


class TestCorpusBoundary:
    def test_the_whole_corpus_adjudicates(self) -> None:
        report = adjudicate_corpus()

        assert report.version == REFERENCE_TRACE_ADJUDICATION_VERSION
        assert len(report.rows) == len(list_reference_trace_specs())
        assert sum(report.counts.values()) == len(report.rows)

    def test_the_project_formulation_boundary_is_exactly_what_is_recorded(self) -> None:
        """A trace moving in or out of this list is a claim change, not a detail."""
        report = adjudicate_corpus()

        assert report.by_class("project_formulation") == PROJECT_FORMULATION_TRACES
        assert report.by_class("published_data") == ("mcculloch_pitts_1943_truth_table",)
        assert report.by_class("external_implementation") == ("iqif_a8752eb_tutorial",)

    def test_the_traces_their_citation_demotes_are_recorded(self) -> None:
        report = adjudicate_corpus()

        demoted = tuple(
            sorted(
                row.name
                for row in report.rows
                if row.derivation in {"independent_rederivation", "analytic_solution"}
                and row.attribution == "project_retained"
            )
        )

        assert demoted == DEMOTED_BY_CITATION

    def test_the_report_is_json_safe_and_carries_every_row(self) -> None:
        payload = adjudicate_corpus().to_public_dict()

        assert payload["version"] == REFERENCE_TRACE_ADJUDICATION_VERSION
        assert payload["total"] == len(list_reference_trace_specs())
        rows = payload["rows"]
        assert isinstance(rows, list)
        assert len(rows) == payload["total"]
        assert all(
            set(row) == {"attribution", "citation", "derivation", "independence", "kind", "name"}
            for row in rows
        )
