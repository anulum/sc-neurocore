<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Model Readiness Labels

A model's panel shows two kinds of tier, and they answer different questions.

- **Declared tiers.** These are the science tiers (S0 to S5) and silicon tiers
  (H0 to H5) that follow from the flags in the model's descriptor. They show
  what the descriptor claims.
- **Verified tiers.** These are shown as, for example, `verified S3 / H0`.
  They count only facets backed by facet receipts that are still fresh, meaning
  their subjects still match the repository. The receipts are read for the
  profile Studio compiles. Verified tiers show what has actually been executed
  and still holds.

## When a model is called perfect

A model is **perfect** when it reaches S5 and the terminal silicon tier that
its deployability class declares.

Studio shows the green `perfect` label only when the *verified* tiers meet
that rule. The model detail carries this as `readiness.is_perfect_verified`.

When only the descriptor's flags meet the rule, the panel shows a muted
`declared perfect · unverified` label instead. The model detail carries this as
`readiness.is_perfect`.

At the time of writing, 49 catalogue models are declared perfect and none is
verified perfect.

## Verified readiness in an installation

In a checkout, the verified tiers are re-derived on every read: a facet receipt
counts only while every subject it recorded still has the digest it had. Some
of those subjects, such as the validator tests and benchmark scripts, are not
part of an installed distribution, so an installation cannot repeat the check.
Before this was addressed, an installed Studio found those subjects missing
and showed receipt-bound tiers as lost: Lapicque as S3 instead of S5, four
models as not enrolled on silicon instead of H1.

An installation therefore serves the verification **sealed** in the checkout
the distribution was built from (`sc_neurocore/studio/verified_readiness.json`,
written by `tools/studio_readiness_seal.py --write`). A test compares the
committed seal with a fresh derivation, so a seal that no longer matches the
receipts fails before it ships, and the distribution test checks that an
installed wheel shows every model at the same verified tiers as the checkout.

The verified block names its `source`: `receipts` when re-derived now,
`sealed` when read from the seal, and `unsealed` when the seal holds no record,
in which case nothing is shown as verified and `unsealed_reason` says why. The
model panel shows "(sealed at build)" beside sealed tiers.

## Co-simulation results

The co-simulation panel compares the generated RTL with the generated
bit-true C kernel. It does not compare the RTL with the model. An agreement is
therefore shown as `RTL = bit-true C kernel`, with the requested and stress
cycles it covers. Its tooltip states:

- the boundary that was compared;
- what the run does not cover;
- that the agreement is neither the model's scientific fidelity nor a proof.

The [hardware numeric contract](numeric-contract.md) describes the stress
schedule and the fixed-point contract that both sides share.

## Not established by Studio

Studio does not establish the following, whatever the labels show:

- timing closure, place and route, and behaviour on a board or in silicon;
- equivalence for unbounded runs or for stimuli that were not simulated;
- the RTL of RK4, exponential-Euler and Gauss-Seidel integrators or of
  sub-stepped models against a bit-true kernel, because no such kernel mirrors
  them;
- stochastic spike detection, whose LFSR the kernel does not model.
