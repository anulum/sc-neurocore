# Conda-Forge Recipe

This directory contains the conda-forge recipe for `sc-neurocore`.

## Submission Steps

1. Fork [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
2. Copy `meta.yaml` to `recipes/sc-neurocore/meta.yaml`
3. Replace `REPLACE_WITH_ACTUAL_SHA256` with the SHA256 of the PyPI sdist tarball:
   ```bash
   pip download sc-neurocore==3.10.0 --no-binary :all: --no-deps -d /tmp
   sha256sum /tmp/sc_neurocore-3.10.0.tar.gz
   ```
4. Open a PR against `conda-forge/staged-recipes`
5. Wait for review (typically 1-2 weeks)

## Testing Locally

```bash
conda build conda/ --no-test
conda build conda/
```
