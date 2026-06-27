# Temporal Logic Verification

Verify safety and liveness properties of SNN spike trains. EU AI Act compliance.

`sc_neurocore.verification.temporal_properties` is in the scoped public
docstring policy and is covered by strict typed tests. The dedicated temporal
property suite exercises the public predicates and result formatting at 100%
isolated module coverage. This page has no polyglot or benchmark counterpart;
changes here are Python API documentation and verification-surface hardening.

::: sc_neurocore.verification.temporal_properties
    options:
      show_root_heading: true
      members:
        - fires_within
        - mutual_exclusion
        - rate_bound
        - refractory_guarantee
        - causal_order
        - bounded_activity
        - VerificationResult
        - PropertyResult
        - Counterexample
