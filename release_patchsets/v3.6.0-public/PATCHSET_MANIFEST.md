CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
Contact us: www.anulum.li  protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AFFERO GENERAL PUBLIC LICENSE v3
Commercial Licensing: Available

# SC-NeuroCore v3.6.0 Public Patchset (anulum/sc-neurocore)

## Target Base

- Recommended base ref: `v3.5.0-phase11`
- Intended branch: `release/v3.6.0`

## Patch Order

1. `0001-phase12-core-public.patch`
2. `0002-public-docs-routing-and-notice.patch`
3. `0003-public-repo-metadata.patch`

## Validation

- Verified on `2026-02-11` by replaying patches in order from `v3.5.0-phase11` in a clean sparse clone.
- Non-blocking warning: markdown lines with intentional trailing double-space trigger whitespace warnings during `git apply`.

## Included Scope

- Phase 12 runtime and bridge changes (fused dense path, xoshiro PRNG, batched dense API)
- Phase 12 tests and benchmark harness updates
- Migration and benchmark documentation updates
- Public docs/routing additions and legal notice file
- Repo metadata adjustment for public GitHub URL

## Intentionally Excluded (Internal/Non-Public)

- `SESSION_LOG_*` internal session logs
- `V3_PHASE12_CODEX_HANDOVER.md`
- `V3_PHASE12_CODE_REVIEW.md`
- Any Phase 13/14 files
- Local build artifacts / caches

## Apply Commands

```bash
git checkout -b release/v3.6.0 v3.5.0-phase11
git apply --index 0001-phase12-core-public.patch
git commit -m "feat(sc-neurocore): Phase 12 core public release payload (v3.6.0)"
git apply --index 0002-public-docs-routing-and-notice.patch
git commit -m "docs(sc-neurocore): public v3.6.0 docs and notice"
git apply --index 0003-public-repo-metadata.patch
git commit -m "chore(sc-neurocore): align README version and public repo URL for v3.6.0"
```

## Suggested Tag

```bash
git tag -a v3.6.0 -m "SC-NeuroCore v3.6.0 (Phase 12 public release)"
```
