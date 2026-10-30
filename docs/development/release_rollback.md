<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Release rollback -->

# Release rollback

A `v*` tag publishes to five channels at once: PyPI (`sc-neurocore` and `sc-neurocore-engine`), crates.io
(`sc_neurocore_engine`), GHCR (`ghcr.io/anulum/sc-neurocore`) and the GitHub Release. None of them can be undone
in place. A faulty release is withdrawn and then fixed forward with a new patch version.

## Rules

- **Never delete or move a release tag.** Release tags are immutable by repository ruleset, and users and
  archives refer to them.
- **Never reuse a version number.** PyPI and crates.io refuse a re-upload of the same version anyway.
- **Record every step**, with its time and the reason, in the release record of the coordination log.

## Withdraw the faulty version

Do only the steps that apply to the fault.

1. **PyPI.** Yank both packages for the faulty version in the PyPI web interface (project, release, *Options →
   Yank*), with the reason. Yanked files stay downloadable by exact pin but are no longer chosen by resolvers.
2. **crates.io.** Run `cargo yank --version <version> sc_neurocore_engine`. Existing lockfiles keep working; new
   resolutions skip the version.
3. **GHCR.** If `latest` points at the faulty image, re-point it at the previous good release:
   `docker buildx imagetools create -t ghcr.io/anulum/sc-neurocore:latest ghcr.io/anulum/sc-neurocore:<previous>`.
   The `alias_from` / `alias_to` inputs of the Docker workflow do the same from GitHub Actions. Keep the faulty
   version tag; it is referenced by digest.
4. **GitHub Release.** Edit the release body to begin with a warning that names the fault and the replacement
   version. Mark it as a pre-release if it must stop being *Latest*. Do not delete it.

## Fix forward

1. Fix the fault on `main` and add a `## [<version>.<patch+1>]` CHANGELOG section that names the withdrawn
   version.
2. Bump every version surface to the patch version; `tests/test_tools/test_release_metadata_versions.py` lists
   them.
3. Run the release gate on the new commit, including the pre-tag dry runs of the tag-triggered workflows.
4. Tag the patch version and follow the normal release.
