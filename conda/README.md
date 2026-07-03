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
3. Replace `PLACEHOLDER` with the SHA256 of the PyPI sdist tarball:
   ```bash
   pip download sc-neurocore==3.15.7 --no-binary :all: --no-deps -d /tmp
   sha256sum /tmp/sc_neurocore-3.15.7.tar.gz
   ```
4. Open a PR against `conda-forge/staged-recipes`
5. Wait for review (typically 1-2 weeks)

## Testing Locally

```bash
conda build conda/ --no-test
conda build conda/
```

The recipe must stay aligned with the base install contract in
`pyproject.toml`: Python >=3.10, NumPy >=1.24, SciPy >=1.10, defusedxml
>=0.7.1, and tomli only on Python <3.11. Its import tests also verify the
packaged offline HDL primitive resources used by pre-built wheels and Docker
images.
