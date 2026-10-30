# Conda-Forge Recipe Draft

This directory contains a local conda-forge recipe draft for `sc-neurocore`.
The package is not yet published on conda-forge: the Anaconda package API and
the `conda-forge/sc-neurocore-feedstock` repository check returned 404 on
2026-07-03, and `meta.yaml` still carries a `sha256: PLACEHOLDER` source hash.
Do not advertise a conda-forge install command until the staged recipe has been
accepted and the feedstock publishes packages.

## Submission Steps

1. Fork [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
2. Copy `meta.yaml` to `recipes/sc-neurocore/meta.yaml`
3. Replace `PLACEHOLDER` with the SHA256 of the published PyPI sdist of the
   version the recipe sets. The digest can exist only after that sdist is on
   PyPI, so the tagged tree keeps the placeholder and the filled recipe lives
   in the staged-recipes pull request or the feedstock:
   ```bash
   VERSION=$(sed -n 's/^{% set version = "\(.*\)" %}$/\1/p' conda/meta.yaml)
   DEST=$(mktemp -d)
   pip download "sc-neurocore==${VERSION}" --no-binary :all: --no-deps -d "$DEST"
   sha256sum "$DEST/sc_neurocore-${VERSION}.tar.gz"
   ```
   Compare the result with the `sha256` digest PyPI lists for that file
   before using it.
4. Open a PR against `conda-forge/staged-recipes`
5. Wait for review (typically 1-2 weeks)

## Testing Locally

```bash
conda build conda/ --no-test
conda build conda/
```

The recipe must stay aligned with the base install contract in
`pyproject.toml` (`requires-python` and `dependencies`). This page does not
repeat that list, so it cannot drift from it. `tools/install_profile_audit.py`
derives the expected `run:` list from `pyproject.toml` rather than from a copy
of this recipe, and reports any row the recipe is missing by name. Its import
tests also verify the packaged offline HDL primitive resources used by
pre-built wheels and Docker images.
