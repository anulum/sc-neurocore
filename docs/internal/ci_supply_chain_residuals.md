<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore -->

# CI executable-provenance: pins and accepted residuals

This is the tracked record of the executable-provenance posture of
`.github/workflows/ci.yml`. It documents what is content-addressed / version
pinned and the one **accepted + tracked residual** that remains by design.

**Origin.** An independent eight-action runtime provenance audit of the CI
workflow separated *reference integrity* (is the `uses:` pin a git commit
object?) from *fetch integrity* (what does the pinned action download and
execute at runtime?). All eight action references were exact commit objects;
six admitted a runtime toolchain/executable/cache fetch with no
repository-bound digest, and the direct SBY acquisition was an unverified
`curl | tar | make`. The audit and the proportionality ruling that governs
this document are recorded in the coordination log
(`codex_2026-07-17T0523_model40_action_runtime_provenance_audit.md` and
`codex_2026-07-17T0527_model40_proportional_provenance_ruling.md`).

## Category A — bespoke fetch: FIXED (hard gate)

The SBY formal-verification tool was previously installed by downloading a
GitHub-generated source archive over `curl` and running `make install` on it
with no integrity check — the archive carried no upstream checksum/signature
and no tag points at the pinned commit. It is now **built from a pinned source
commit**: `git` fetches exactly commit
`0cb15210a857e4d20030484b37a6b9b6386bd01d`, the step asserts
`HEAD == <pinned commit>`, and fails closed (`set -euo pipefail` + explicit
`test`) on any mismatch. The installed source is content-addressed by the git
object model; there is no unverified external-archive fetch-execute left.

## Category B — official setup-* actions: exact-version pinned, residual accepted

Every official toolchain action now selects an **exact** version instead of a
mutable channel/range, which removes silent version drift:

| Action | Was | Now |
| --- | --- | --- |
| `actions/setup-go` | `1.26.3` (already exact) | `1.26.3` |
| `actions/setup-node` | `22` (major) | `22.23.1` |
| `actions/setup-python` | `3.10`–`3.14` (minor ranges) | `3.10.20`, `3.11.15`, `3.12.13`, `3.13.14`, `3.14.6` |
| `dtolnay/rust-toolchain` | `stable` | `1.97.1` (also pinned in `engine/rust-toolchain.toml`) |
| `prefix-dev/setup-pixi` | `latest` | `v0.73.0` |
| `Swatinem/rust-cache` | default key | key bound to `github.repository` + `Cargo.lock` hash |

**The accepted + tracked residual (exact invariant).** The official,
GitHub-maintained setup-* actions resolve their pinned version at runtime from
official mutable manifests — `actions/go-versions`, `actions/node-versions`,
`actions/python-versions`, the `static.rust-lang.org` dist channel, and the
`prefix-dev/pixi` releases — **without a repository-bound per-artifact
SHA-256**. The *version selection* is now deterministic (exact pin); the
residual is only the absence of a per-artifact digest on that official-manifest
fetch. `Swatinem/rust-cache` restores compiled artefacts from the Actions cache,
now bound to a repo + lockfile key so a poisoned cross-repo cache cannot be
restored.

This residual is **accepted** under the proportionality ruling: per-artifact
digest pinning is infeasible with these actions as designed, the risk tier
(official GitHub-maintained action + official manifest) is categorically lower
than a bespoke random-archive `curl`, and blocking all CI on an unachievable
absolute is disproportionate. A higher bar — fully vendored or air-gapped
toolchains — is a larger infrastructure decision to be surfaced to the owner,
not the default posture.

## Category C — checkout + upload-artifact: no current-invocation blocker

`actions/checkout` fetches repository objects by object id (content-integral)
with `lfs` and `submodules` off; its archive fallback (used only when a suitable
git is unavailable) would fetch source through the GitHub API without a
separately pinned archive digest — noted, not currently exercised.
`actions/upload-artifact` is upload-only with bundled, pinned action code.

## Maintenance

- Bump the action `uses:` pin **and** its exact version pin together; for Rust,
  keep `.github/workflows/ci.yml` `toolchain:` and `engine/rust-toolchain.toml`
  `channel` in lockstep.
- Re-run the eight-action runtime audit when adding any new action that fetches
  or executes a toolchain, binary, or executable-bearing cache at runtime.
- Exact interpreter/toolchain patch pins are chosen from the official manifests
  (verified at source at pin time); the `python-minor` matrix key keeps the
  branch-protection-required job names `test (3.10)`..`test (3.14)` stable across
  patch bumps.
